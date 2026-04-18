"""
ASL Avatar Motion Diffusion Model V1 + Voting + Cross-Attention Fusion
======================================================================

Extends MotionDiffusionModelV1_CFG with VotingFusionModule.

Key difference from V1_voting:
    V1_voting:       gloss → voting gate → single condition vector (B, D) → prepend token
    V1_votingfusion: gloss → voting gate → cross-attention with motion → per-frame condition (B, T, D)

The cross-attention implicitly learns temporal alignment between gloss tokens
and motion frames, replacing the need for explicit gloss reordering.

Pipeline:
    LLM draft gloss → per-word CLIP embedding
        → VotingFusionModule.vote() → gated gloss tokens (B, K, D)
        → VotingFusionModule.fuse(motion_tokens, gated_gloss_tokens) → per-frame condition
        → transformer encoder denoiser → eps_pred
"""

import torch
import torch.nn as nn

from network.MotionDiffusionModelV1_cfg import (
    MotionDiffusionModelV1_CFG, sinusoidal_embedding
)
from network.VotingFusionModule import VotingFusionModule


class MotionDiffusionModelV1_VotingFusion(MotionDiffusionModelV1_CFG):

    def __init__(self, cfg):
        original_cond_mode = getattr(cfg, 'COND_MODE', 'votingfusion')
        cfg.COND_MODE = 'sentence'
        super().__init__(cfg)
        cfg.COND_MODE = original_cond_mode
        self.cond_mode = 'votingfusion'

        if self.text_encoder_type == 't5':
            enc_dim = self.text_encoder.config.d_model
        else:
            enc_dim = getattr(cfg, 'CLIP_DIM', 512)

        self.voting_fusion = VotingFusionModule(
            encoder_dim=enc_dim,
            model_dim=cfg.MODEL_DIM,
            n_voting_layers=getattr(cfg, 'VOTING_N_LAYERS', 2),
            n_voting_heads=getattr(cfg, 'VOTING_N_HEADS', 4),
            n_fusion_layers=getattr(cfg, 'FUSION_N_LAYERS', 2),
            n_fusion_heads=getattr(cfg, 'FUSION_N_HEADS', 8),
            ff_mult=getattr(cfg, 'VOTING_FF_MULT', 2),
            dropout=cfg.DROPOUT,
            max_words=getattr(cfg, 'VOTING_MAX_WORDS', 64),
        )

        self._voting_gates = None
        self._voting_word_mask = None

    # ---- word embedding (same as V1_voting) ----

    def _embed_gloss_words(self, gloss_strings, device):
        max_words = self.voting_fusion.pos_enc.num_embeddings
        word_lists = [g.split()[:max_words] if g.strip() else [''] for g in gloss_strings]
        lengths = [len(wl) for wl in word_lists]
        max_K = max(lengths)
        B = len(gloss_strings)

        all_words = [w for wl in word_lists for w in wl]

        with torch.no_grad():
            tokens = self.tokenizer(
                all_words, padding=True, truncation=True,
                max_length=77, return_tensors='pt',
            ).to(device)
            if self.text_encoder_type == 't5':
                hidden = self.text_encoder(**tokens).last_hidden_state
                attn = tokens['attention_mask'].unsqueeze(-1).float()
                word_embs = (hidden * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
            else:
                word_embs = self.text_encoder(**tokens).pooler_output

        enc_dim = word_embs.shape[-1]
        word_embeddings = torch.zeros(B, max_K, enc_dim, device=device, dtype=word_embs.dtype)
        word_mask = torch.ones(B, max_K, dtype=torch.bool, device=device)

        idx = 0
        for i, length in enumerate(lengths):
            word_embeddings[i, :length] = word_embs[idx:idx + length]
            word_mask[i, :length] = False
            idx += length

        return word_embeddings, word_mask

    # ---- override denoise to use cross-attention fusion ----

    def denoise(self, x_t, t, condition, padding_mask=None,
                gloss_word_embs=None, gloss_word_mask=None):
        """
        Core denoiser with cross-attention fusion.

        When gloss_word_embs is provided (training/inference with gloss):
            motion_tokens cross-attend to gated gloss tokens via VotingFusionModule.

        When gloss_word_embs is None (CFG unconditional pass):
            Falls back to parent's single-vector conditioning.
        """
        B, T, _ = x_t.shape
        device = x_t.device

        t_emb = sinusoidal_embedding(t, self.cfg.MODEL_DIM)
        t_token = self.timestep_proj(t_emb).unsqueeze(1)  # (B, 1, D)

        motion_tokens = self.pose_proj(x_t)  # (B, T, D)
        motion_tokens = self.pe(motion_tokens)

        if gloss_word_embs is not None:
            # Cross-attention fusion: motion attends to gated gloss
            # Prepend timestep token to motion before fusion
            t_prefix_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            motion_with_t = torch.cat([t_token, motion_tokens], dim=1)  # (B, 1+T, D)

            if padding_mask is not None:
                full_motion_mask = torch.cat([t_prefix_mask, padding_mask], dim=1)
            else:
                full_motion_mask = None

            fused, gates = self.voting_fusion(
                gloss_word_embs, gloss_word_mask,
                motion_with_t, full_motion_mask,
            )

            self._voting_gates = gates
            self._voting_word_mask = gloss_word_mask

            # Run through main transformer encoder
            out = self.transformer(fused, src_key_padding_mask=full_motion_mask)

            # Strip timestep token, keep motion frames
            motion_out = out[:, 1:, :]
        else:
            # Unconditional: use single condition vector (for CFG)
            c_token = condition.unsqueeze(1)
            full_seq = torch.cat([t_token, c_token, motion_tokens], dim=1)

            if padding_mask is not None:
                prefix = torch.zeros(B, 2, dtype=torch.bool, device=device)
                full_mask = torch.cat([prefix, padding_mask], dim=1)
            else:
                full_mask = None

            out = self.transformer(full_seq, src_key_padding_mask=full_mask)
            motion_out = out[:, 2:, :]

        return self.output_proj(motion_out)

    # ---- override forward ----

    def forward(self, x_t, t, cond_input, padding_mask=None, motion=None, gloss_input=None):
        assert gloss_input is not None, "gloss_input required for votingfusion model"

        device = x_t.device
        word_embs, word_mask = self._embed_gloss_words(gloss_input, device)

        # CFG: randomly drop condition
        if self.training and self.uncond_prob > 0:
            B = x_t.shape[0]
            drop_mask = torch.rand(B, device=device) < self.uncond_prob

            if drop_mask.any():
                cond_idx = (~drop_mask).nonzero(as_tuple=True)[0]
                uncond_idx = drop_mask.nonzero(as_tuple=True)[0]

                x_t_active = x_t[:, :, self.tosave_slices]
                T_active = x_t_active.shape[1]
                output = torch.zeros(B, T_active, self.input_dim,
                                     device=device, dtype=x_t_active.dtype)

                if cond_idx.numel() > 0:
                    out_cond = self.denoise(
                        x_t_active[cond_idx], t[cond_idx], None,
                        padding_mask[cond_idx] if padding_mask is not None else None,
                        gloss_word_embs=word_embs[cond_idx],
                        gloss_word_mask=word_mask[cond_idx],
                    )
                    output[cond_idx] = out_cond

                if uncond_idx.numel() > 0:
                    null_cond = self.null_cond_emb.unsqueeze(0).expand(uncond_idx.numel(), -1)
                    out_uncond = self.denoise(
                        x_t_active[uncond_idx], t[uncond_idx], null_cond,
                        padding_mask[uncond_idx] if padding_mask is not None else None,
                    )
                    output[uncond_idx] = out_uncond

                if len(self.all_slices) == len(self.tosave_slices):
                    return output
                if self.prediction_type == 'epsilon':
                    out = torch.zeros_like(motion, dtype=output.dtype)
                else:
                    out = motion.clone().to(output.dtype)
                out[:, :, self.tosave_slices] = output
                return out

        # No CFG dropout (or all samples kept): standard cross-attention path
        x_t_active = x_t[:, :, self.tosave_slices]
        output = self.denoise(x_t_active, t, None, padding_mask,
                              gloss_word_embs=word_embs, gloss_word_mask=word_mask)

        if len(self.all_slices) == len(self.tosave_slices):
            return output
        if self.prediction_type == 'epsilon':
            out = torch.zeros_like(motion, dtype=output.dtype)
        else:
            out = motion.clone().to(output.dtype)
        out[:, :, self.tosave_slices] = output
        return out

    # ---- override generate ----

    @torch.no_grad()
    def generate(self, cond_input, seq_len=100, device='cuda',
                 num_steps=50, eta=0.0, guidance_scale=None, gloss_input=None):
        self.eval()
        assert gloss_input is not None, "gloss_input required for votingfusion model"

        if guidance_scale is None:
            guidance_scale = getattr(self.cfg, 'GUIDANCE_SCALE', 1.0)

        B = len(gloss_input)
        word_embs, word_mask = self._embed_gloss_words(gloss_input, device)
        null_cond = self.null_cond_emb.unsqueeze(0).expand(B, -1)

        step_size = self.num_timesteps // num_steps
        timesteps = list(reversed(list(range(0, self.num_timesteps, step_size))))

        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, dtype=torch.long, device=device)
            alpha_t = self.alphas_cumprod[t_cur]

            if self.prediction_type == 'epsilon':
                eps_cond = self.denoise(x_t, t_batch, None, None,
                                        gloss_word_embs=word_embs,
                                        gloss_word_mask=word_mask)
                if guidance_scale != 1.0:
                    eps_uncond = self.denoise(x_t, t_batch, null_cond)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond
                x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            else:
                x_0_cond = self.denoise(x_t, t_batch, None, None,
                                         gloss_word_embs=word_embs,
                                         gloss_word_mask=word_mask)
                if guidance_scale != 1.0:
                    x_0_uncond = self.denoise(x_t, t_batch, null_cond)
                    x_0_pred = x_0_uncond + guidance_scale * (x_0_cond - x_0_uncond)
                else:
                    x_0_pred = x_0_cond
                eps = (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t)

            if i == len(timesteps) - 1:
                x_t = x_0_pred
                break

            t_next = timesteps[i + 1]
            alpha_next = self.alphas_cumprod[t_next]
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            x_t = (torch.sqrt(alpha_next) * x_0_pred
                    + torch.sqrt(1 - alpha_next - sigma**2) * eps
                    + sigma * torch.randn_like(x_t))

        B, T, D = x_t.shape
        if D < len(self.all_slices):
            out = torch.zeros(B, T, len(self.all_slices), dtype=x_t.dtype, device=device)
            out[:, :, self.tosave_slices] = x_t
        else:
            out = x_t
        return out
