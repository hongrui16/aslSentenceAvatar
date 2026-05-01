"""
ASL Avatar Motion Diffusion Model V1 + Voting Module
=====================================================

Extends MotionDiffusionModelV1_CFG with a per-gloss voting gate.

Pipeline:
    LLM draft gloss → per-word CLIP embedding → VotingConditionModule
        → gated condition embedding → MDM denoiser → eps_pred

Sentence conditioning modes (--sent_cond_mode):
    'none'    — gloss only (original behavior)
    'prefix'  — sentence as additional prefix token to denoiser
    'kv_pool' — sentence token added to voting module's pool alongside gloss

Usage:
    model = MotionDiffusionModelV1_Voting(cfg)
    eps_pred = model(x_t, t, sentences, padding_mask, motion,
                     gloss_input=llm_draft_glosses)
    motion = model.generate(sentences, seq_len=200,
                            gloss_input=llm_draft_glosses)
"""

import torch
import torch.nn as nn

from network.MotionDiffusionModelV1_cfg import (
    MotionDiffusionModelV1_CFG, sinusoidal_embedding
)
from network.VotingConditionModule import VotingConditionModule
from network.PhonoAttributeEncoder import PhonoAttributeEncoder


class MotionDiffusionModelV1_Voting(MotionDiffusionModelV1_CFG):

    def __init__(self, cfg):
        original_cond_mode = getattr(cfg, 'COND_MODE', 'voting')
        cfg.COND_MODE = 'sentence'
        super().__init__(cfg)
        cfg.COND_MODE = original_cond_mode
        self.cond_mode = 'voting'

        self.sent_cond_mode = getattr(cfg, 'SENT_COND_MODE', 'none')

        if self.text_encoder_type == 't5':
            enc_dim = self.text_encoder.config.d_model
        elif self.text_encoder_type == 'mclip':
            enc_dim = self.text_encoder.config.hidden_size
        else:
            enc_dim = getattr(cfg, 'CLIP_DIM', 512)

        self.use_phono = getattr(cfg, 'USE_PHONO', False)
        phono_dim = getattr(cfg, 'PHONO_DIM', 64) if self.use_phono else 0

        self.voting_module = VotingConditionModule(
            encoder_dim=enc_dim,
            model_dim=cfg.MODEL_DIM,
            n_layers=getattr(cfg, 'VOTING_N_LAYERS', 2),
            n_heads=getattr(cfg, 'VOTING_N_HEADS', 4),
            ff_mult=getattr(cfg, 'VOTING_FF_MULT', 2),
            dropout=cfg.DROPOUT,
            max_words=getattr(cfg, 'VOTING_MAX_WORDS', 64),
            sent_cond_mode=self.sent_cond_mode,
            phono_dim=phono_dim,
        )

        if self.use_phono:
            from utils.signbank_phono import SignBankPhonoLookup
            signbank_csv = getattr(cfg, 'SIGNBANK_CSV',
                                   'data/ASL_signbank/asl_signbank_dictionary-export.csv')
            self.phono_lookup = SignBankPhonoLookup(signbank_csv)
            self.phono_encoder = PhonoAttributeEncoder(
                self.phono_lookup.num_classes,
                phono_dim=phono_dim,
            )

        self._voting_gates = None
        self._voting_word_mask = None

    def _embed_gloss_words(self, gloss_strings, device):
        max_words = self.voting_module.pos_enc.num_embeddings
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
        """Look up SignBank phono attributes and encode them."""
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

    def _encode_sentence_raw(self, cond_input, device):
        """Encode sentence → projected embedding (B, model_dim)."""
        raw = self._encode_text(cond_input, device)  # (B, clip_dim)
        return self.condition_proj(raw)  # (B, model_dim)

    def get_condition(self, cond_input, device, gloss_input=None):
        """Build condition via voting module on LLM draft gloss."""
        assert gloss_input is not None, "gloss_input required for voting model"

        word_embs, word_mask = self._embed_gloss_words(gloss_input, device)
        phono_tokens = self._embed_phono(gloss_input, word_embs.shape[1], device)

        sent_emb = None
        if self.sent_cond_mode == 'kv_pool':
            sent_emb = self._encode_sentence_raw(cond_input, device)

        condition, gates = self.voting_module(
            word_embs, word_mask, sent_emb=sent_emb, phono_tokens=phono_tokens)

        self._voting_gates = gates
        self._voting_word_mask = word_mask
        return condition

    # ---- prefix mode: override denoise to add sentence token ----

    def denoise(self, x_t, t, condition, padding_mask=None, sent_token=None):
        B, T, _ = x_t.shape
        device = x_t.device

        t_emb = sinusoidal_embedding(t, self.cfg.MODEL_DIM)
        t_token = self.timestep_proj(t_emb).unsqueeze(1)

        c_token = condition.unsqueeze(1)

        motion_tokens = self.pose_proj(x_t)
        motion_tokens = self.pe(motion_tokens)

        if self.sent_cond_mode == 'prefix' and sent_token is not None:
            # [t, sent, voting_cond, motion]
            full_seq = torch.cat([t_token, sent_token, c_token, motion_tokens], dim=1)
            n_prefix = 3
        else:
            # [t, voting_cond, motion]
            full_seq = torch.cat([t_token, c_token, motion_tokens], dim=1)
            n_prefix = 2

        if padding_mask is not None:
            prefix = torch.zeros(B, n_prefix, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix, padding_mask], dim=1)
        else:
            full_mask = None

        out = self.transformer(full_seq, src_key_padding_mask=full_mask)
        motion_out = out[:, n_prefix:, :]
        return self.output_proj(motion_out)

    # ---- override forward to wire sentence through ----

    def forward(self, x_t, t, cond_input, padding_mask=None, motion=None, gloss_input=None):
        condition = self.get_condition(cond_input, x_t.device, gloss_input=gloss_input)

        # Encode sentence for prefix mode
        sent_token = None
        if self.sent_cond_mode == 'prefix':
            sent_token = self._encode_sentence_raw(cond_input, x_t.device).unsqueeze(1)

        # CFG: randomly replace condition with null embedding during training
        if self.training and self.uncond_prob > 0:
            B = condition.shape[0]
            drop_mask = torch.rand(B, device=condition.device) < self.uncond_prob
            null_emb = self.null_cond_emb.unsqueeze(0).expand(B, -1)
            condition = torch.where(drop_mask.unsqueeze(-1), null_emb, condition)
            if sent_token is not None:
                sent_token = sent_token.masked_fill(drop_mask.view(B, 1, 1), 0.0)

        x_t = x_t[:, :, self.tosave_slices]
        output = self.denoise(x_t, t, condition, padding_mask, sent_token=sent_token)

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
        if guidance_scale is None:
            guidance_scale = getattr(self.cfg, 'GUIDANCE_SCALE', 1.0)

        B = len(cond_input) if isinstance(cond_input, (list, tuple)) else cond_input.shape[0]
        condition = self.get_condition(cond_input, device, gloss_input=gloss_input)
        null_cond = self.null_cond_emb.unsqueeze(0).expand(B, -1)

        sent_token = None
        if self.sent_cond_mode == 'prefix':
            sent_token = self._encode_sentence_raw(cond_input, device).unsqueeze(1)

        # Regression mode: single forward pass with zeros + t=0, no DDIM loop.
        if getattr(self.cfg, 'REGRESSION_MODE', False):
            t0 = torch.zeros((B,), dtype=torch.long, device=device)
            # Feed zeros of model.input_dim (the dim AFTER tosave_slice reduction).
            x0_in = torch.zeros(B, seq_len, self.input_dim, device=device)
            output = self.denoise(x0_in, t0, condition, sent_token=sent_token)
            if len(self.all_slices) == len(self.tosave_slices):
                return output
            out = torch.zeros(B, seq_len, len(self.all_slices), dtype=output.dtype, device=device)
            out[:, :, self.tosave_slices] = output
            return out

        step_size = self.num_timesteps // num_steps
        timesteps = list(reversed(list(range(0, self.num_timesteps, step_size))))

        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, dtype=torch.long, device=device)
            alpha_t = self.alphas_cumprod[t_cur]

            if self.prediction_type == 'epsilon':
                eps_cond = self.denoise(x_t, t_batch, condition, sent_token=sent_token)
                if guidance_scale != 1.0:
                    eps_uncond = self.denoise(x_t, t_batch, null_cond)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond
                x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            else:
                x_0_cond = self.denoise(x_t, t_batch, condition, sent_token=sent_token)
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
