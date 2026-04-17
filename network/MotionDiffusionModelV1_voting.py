"""
ASL Avatar Motion Diffusion Model V1 + Voting Module
=====================================================

Extends MotionDiffusionModelV1_CFG with a per-gloss voting gate.

Pipeline:
    LLM draft gloss → per-word CLIP embedding → VotingConditionModule
        → gated condition embedding → MDM denoiser → eps_pred

The VotingConditionModule is the only new trainable component. It is
trained end-to-end: the diffusion MSE loss backpropagates through the
sigmoid gates into the voting transformer.

Usage:
    model = MotionDiffusionModelV1_Voting(cfg)
    eps_pred = model(x_t, t, sentences, padding_mask, motion,
                     gloss_input=llm_draft_glosses)
    motion = model.generate(sentences, seq_len=200,
                            gloss_input=llm_draft_glosses)
"""

import torch
import torch.nn as nn

from network.MotionDiffusionModelV1_cfg import MotionDiffusionModelV1_CFG
from network.VotingConditionModule import VotingConditionModule


class MotionDiffusionModelV1_Voting(MotionDiffusionModelV1_CFG):

    def __init__(self, cfg):
        # Force cond_mode to 'voting' internally so parent skips gloss_proj
        # but still builds condition_proj for potential sentence baseline
        original_cond_mode = getattr(cfg, 'COND_MODE', 'voting')
        cfg.COND_MODE = 'sentence'
        super().__init__(cfg)
        cfg.COND_MODE = original_cond_mode
        self.cond_mode = 'voting'

        if self.text_encoder_type == 't5':
            enc_dim = self.text_encoder.config.d_model
        else:
            enc_dim = getattr(cfg, 'CLIP_DIM', 512)

        self.voting_module = VotingConditionModule(
            encoder_dim=enc_dim,
            model_dim=cfg.MODEL_DIM,
            n_layers=getattr(cfg, 'VOTING_N_LAYERS', 2),
            n_heads=getattr(cfg, 'VOTING_N_HEADS', 4),
            ff_mult=getattr(cfg, 'VOTING_FF_MULT', 2),
            dropout=cfg.DROPOUT,
            max_words=getattr(cfg, 'VOTING_MAX_WORDS', 64),
        )

        self._voting_gates = None
        self._voting_word_mask = None

    def _embed_gloss_words(self, gloss_strings, device):
        """Embed each gloss word separately through the frozen text encoder.

        Args:
            gloss_strings: list[str] of length B (space-separated gloss words)
            device: target device
        Returns:
            word_embeddings: (B, max_K, encoder_dim)
            word_mask: (B, max_K) bool, True = padding
        """
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

    def get_condition(self, cond_input, device, gloss_input=None):
        """Build condition via voting module on LLM draft gloss."""
        assert gloss_input is not None, "gloss_input required for voting model"

        word_embs, word_mask = self._embed_gloss_words(gloss_input, device)
        condition, gates = self.voting_module(word_embs, word_mask)

        self._voting_gates = gates
        self._voting_word_mask = word_mask
        return condition
