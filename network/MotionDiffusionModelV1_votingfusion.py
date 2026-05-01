"""
ASL Avatar Motion Diffusion Model V1 + Cross-Attention Fusion
=============================================================

Extends MotionDiffusionModelV1_CFG with VotingFusionModule.

Key difference from V1_voting:
    V1_voting:       gloss → voting gate → single condition vector (B, D) → prepend token
    V1_votingfusion: gloss → transformer encoder → cross-attention with motion → per-frame condition (B, T, D)

Sentence conditioning modes (--sent_cond_mode):
    'none'    — gloss only (original behavior, for backward compat)
    'prefix'  — sentence as prefix token to denoiser (路径一)
    'kv_pool' — sentence in cross-attention K/V pool alongside gloss (路径二)

Pipeline:
    LLM draft gloss → per-word CLIP embedding
        → VotingFusionModule.encode_gloss() → contextualized gloss tokens (B, K, D)
        → VotingFusionModule.fuse(motion_tokens, gloss_tokens, [sent_token]) → per-frame condition
        → transformer encoder denoiser → eps_pred
"""

import torch
import torch.nn as nn

from network.MotionDiffusionModelV1_cfg import (
    MotionDiffusionModelV1_CFG, sinusoidal_embedding
)
from network.VotingFusionModule import VotingFusionModule
from network.PhonoAttributeEncoder import PhonoAttributeEncoder


class MotionDiffusionModelV1_VotingFusion(MotionDiffusionModelV1_CFG):

    def __init__(self, cfg):
        original_cond_mode = getattr(cfg, 'COND_MODE', 'votingfusion')
        cfg.COND_MODE = 'sentence'
        super().__init__(cfg)
        cfg.COND_MODE = original_cond_mode
        self.cond_mode = 'votingfusion'

        self.sent_cond_mode = getattr(cfg, 'SENT_COND_MODE', 'none')

        if self.text_encoder_type == 't5':
            enc_dim = self.text_encoder.config.d_model
        elif self.text_encoder_type == 'mclip':
            enc_dim = self.text_encoder.config.hidden_size
        else:
            enc_dim = getattr(cfg, 'CLIP_DIM', 512)

        self.use_phono = getattr(cfg, 'USE_PHONO', False)
        phono_dim = getattr(cfg, 'PHONO_DIM', 64) if self.use_phono else 0

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
            sent_cond_mode=self.sent_cond_mode,
            phono_dim=phono_dim,
        )

        # Learned null memory for CFG unconditional path (same cross-attention structure)
        self.null_gloss_memory = nn.Parameter(torch.randn(1, 1, cfg.MODEL_DIM) * 0.01)

        if self.use_phono:
            from utils.signbank_phono import SignBankPhonoLookup
            signbank_csv = getattr(cfg, 'SIGNBANK_CSV',
                                   'data/ASL_signbank/asl_signbank_dictionary-export.csv')
            self.phono_lookup = SignBankPhonoLookup(signbank_csv)
            self.phono_encoder = PhonoAttributeEncoder(
                self.phono_lookup.num_classes,
                phono_dim=phono_dim,
            )


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
            if self.text_encoder_type in ('t5', 'mclip'):
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

    def _embed_phono(self, gloss_strings, max_K, device):
        """Look up SignBank phono attributes and encode them.

        Args:
            gloss_strings: list[str] of length B
            max_K: max number of words (must match word_embeddings dim 1)
            device: target device
        Returns:
            phono_tokens: (B, max_K, phono_dim) or None if use_phono=False
        """
        if not self.use_phono:
            return None

        from utils.signbank_phono import PHONO_ATTRIBUTES
        B = len(gloss_strings)
        n_attrs = len(PHONO_ATTRIBUTES)
        attr_indices = torch.zeros(B, max_K, n_attrs, dtype=torch.long, device=device)
        found_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)

        for i, gloss_str in enumerate(gloss_strings):
            words = gloss_str.split()[:max_K] if gloss_str.strip() else ['']
            for j, w in enumerate(words):
                attrs = self.phono_lookup.get_attribute_indices(w)
                if attrs is not None:
                    found_mask[i, j] = True
                    for k, attr_name in enumerate(PHONO_ATTRIBUTES):
                        attr_indices[i, j, k] = attrs[attr_name]

        return self.phono_encoder(attr_indices, found_mask)

    def _encode_sentence(self, cond_input, device):
        """Encode sentence → projected token (B, 1, model_dim)."""
        raw = self._encode_text(cond_input, device)  # (B, clip_dim)
        projected = self.condition_proj(raw)  # (B, model_dim)
        return projected.unsqueeze(1)  # (B, 1, model_dim)

    # ---- override denoise to use cross-attention fusion ----

    def denoise(self, x_t, t, condition, padding_mask=None,
                gloss_word_embs=None, gloss_word_mask=None,
                sent_token=None, phono_tokens=None):
        """
        Core denoiser with cross-attention fusion.

        When gloss_word_embs is provided (training/inference with gloss):
            motion_tokens cross-attend to gloss tokens via VotingFusionModule.

        When gloss_word_embs is None (CFG unconditional pass):
            Falls back to parent's single-vector conditioning.

        sent_token: (B, 1, model_dim) — used by 'prefix' and 'kv_pool' modes.
        """
        B, T, _ = x_t.shape
        device = x_t.device

        t_emb = sinusoidal_embedding(t, self.cfg.MODEL_DIM)
        t_token = self.timestep_proj(t_emb).unsqueeze(1)  # (B, 1, D)

        motion_tokens = self.pose_proj(x_t)  # (B, T, D)
        motion_tokens = self.pe(motion_tokens)

        if gloss_word_embs is not None:
            # Cross-attention fusion: motion attends to gloss tokens
            # Prepend timestep token to motion before fusion
            t_prefix_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            motion_with_t = torch.cat([t_token, motion_tokens], dim=1)  # (B, 1+T, D)

            if padding_mask is not None:
                full_motion_mask = torch.cat([t_prefix_mask, padding_mask], dim=1)
            else:
                full_motion_mask = None

            # Pass sent_token for kv_pool mode
            kv_sent = sent_token if self.sent_cond_mode == 'kv_pool' else None

            fused = self.voting_fusion(
                gloss_word_embs, gloss_word_mask,
                motion_with_t, full_motion_mask,
                sent_token=kv_sent,
                phono_tokens=phono_tokens,
            )

            # prefix mode: prepend sentence token before denoiser
            if self.sent_cond_mode == 'prefix' and sent_token is not None:
                s_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
                fused = torch.cat([sent_token, fused], dim=1)  # (B, 1+1+T, D)
                if full_motion_mask is not None:
                    full_motion_mask = torch.cat([s_mask, full_motion_mask], dim=1)

            # Run through main transformer encoder
            out = self.transformer(fused, src_key_padding_mask=full_motion_mask)

            # Strip prefix tokens, keep motion frames
            if self.sent_cond_mode == 'prefix' and sent_token is not None:
                motion_out = out[:, 2:, :]  # skip sent + timestep
            else:
                motion_out = out[:, 1:, :]  # skip timestep only
        else:
            # Unconditional: use null memory through same cross-attention path
            t_prefix_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            motion_with_t = torch.cat([t_token, motion_tokens], dim=1)

            if padding_mask is not None:
                full_motion_mask = torch.cat([t_prefix_mask, padding_mask], dim=1)
            else:
                full_motion_mask = None

            null_memory = self.null_gloss_memory.expand(B, -1, -1)
            null_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

            fused = self.voting_fusion.fuse(
                motion_with_t, null_memory,
                motion_mask=full_motion_mask, gloss_mask=null_mask,
            )

            if self.sent_cond_mode == 'prefix':
                null_sent = torch.zeros(B, 1, self.cfg.MODEL_DIM, device=device)
                s_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
                fused = torch.cat([null_sent, fused], dim=1)
                if full_motion_mask is not None:
                    full_motion_mask = torch.cat([s_mask, full_motion_mask], dim=1)

            out = self.transformer(fused, src_key_padding_mask=full_motion_mask)

            if self.sent_cond_mode == 'prefix':
                motion_out = out[:, 2:, :]
            else:
                motion_out = out[:, 1:, :]

        return self.output_proj(motion_out)

    # ---- override forward ----

    def forward(self, x_t, t, cond_input, padding_mask=None, motion=None, gloss_input=None):
        assert gloss_input is not None, "gloss_input required for votingfusion model"

        device = x_t.device
        word_embs, word_mask = self._embed_gloss_words(gloss_input, device)
        phono_tokens = self._embed_phono(gloss_input, word_embs.shape[1], device)

        # Encode sentence if using sentence conditioning
        sent_token = None
        if self.sent_cond_mode != 'none':
            sent_token = self._encode_sentence(cond_input, device)

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
                    s = sent_token[cond_idx] if sent_token is not None else None
                    p = phono_tokens[cond_idx] if phono_tokens is not None else None
                    out_cond = self.denoise(
                        x_t_active[cond_idx], t[cond_idx], None,
                        padding_mask[cond_idx] if padding_mask is not None else None,
                        gloss_word_embs=word_embs[cond_idx],
                        gloss_word_mask=word_mask[cond_idx],
                        sent_token=s,
                        phono_tokens=p,
                    )
                    output[cond_idx] = out_cond.to(output.dtype)

                if uncond_idx.numel() > 0:
                    out_uncond = self.denoise(
                        x_t_active[uncond_idx], t[uncond_idx], None,
                        padding_mask[uncond_idx] if padding_mask is not None else None,
                    )
                    output[uncond_idx] = out_uncond.to(output.dtype)

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
                              gloss_word_embs=word_embs, gloss_word_mask=word_mask,
                              sent_token=sent_token, phono_tokens=phono_tokens)

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
        phono_tokens = self._embed_phono(gloss_input, word_embs.shape[1], device)

        # Encode sentence if using sentence conditioning
        sent_token = None
        if self.sent_cond_mode != 'none':
            sent_token = self._encode_sentence(cond_input, device)

        step_size = self.num_timesteps // num_steps
        timesteps = list(reversed(list(range(0, self.num_timesteps, step_size))))

        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, dtype=torch.long, device=device)
            alpha_t = self.alphas_cumprod[t_cur]

            if self.prediction_type == 'epsilon':
                eps_cond = self.denoise(x_t, t_batch, None, None,
                                        gloss_word_embs=word_embs,
                                        gloss_word_mask=word_mask,
                                        sent_token=sent_token,
                                        phono_tokens=phono_tokens)
                if guidance_scale != 1.0:
                    eps_uncond = self.denoise(x_t, t_batch, None)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond
                x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            else:
                x_0_cond = self.denoise(x_t, t_batch, None, None,
                                         gloss_word_embs=word_embs,
                                         gloss_word_mask=word_mask,
                                         sent_token=sent_token,
                                         phono_tokens=phono_tokens)
                if guidance_scale != 1.0:
                    x_0_uncond = self.denoise(x_t, t_batch, None)
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
