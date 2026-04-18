"""
VotingFusionModule
==================
Combines per-gloss voting gate with cross-attention fusion.

Stage 1 — Voting Gate:
    LLM draft gloss words → per-word CLIP embedding → transformer → sigmoid gate
    Same as VotingConditionModule: token-level keep/drop.

Stage 2 — Cross-Attention Fusion:
    Motion temporal features (B, T, D) cross-attend to gated gloss tokens (B, K, D).
    Attention implicitly learns temporal alignment (which frame → which gloss),
    replacing explicit gloss reordering.

Output: per-frame condition (B, T, D) instead of a single vector.
"""

import torch
import torch.nn as nn
import math


class VotingFusionModule(nn.Module):

    def __init__(self, encoder_dim, model_dim, n_voting_layers=2, n_voting_heads=4,
                 n_fusion_layers=2, n_fusion_heads=8,
                 ff_mult=2, dropout=0.1, max_words=64):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.model_dim = model_dim

        # ---- Stage 1: Voting Gate (operates in encoder_dim space) ----
        self.pos_enc = nn.Embedding(max_words, encoder_dim)

        voting_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=n_voting_heads,
            dim_feedforward=encoder_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.voting_transformer = nn.TransformerEncoder(
            voting_layer, num_layers=n_voting_layers
        )
        self.gate_head = nn.Linear(encoder_dim, 1)

        self.gloss_proj = nn.Linear(encoder_dim, model_dim)

        # ---- Stage 2: Cross-Attention Fusion (operates in model_dim space) ----
        fusion_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=n_fusion_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.fusion_decoder = nn.TransformerDecoder(
            fusion_layer, num_layers=n_fusion_layers
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_enc.weight, std=0.02)
        nn.init.constant_(self.gate_head.bias, 1.0)

    def vote(self, word_embeddings, word_mask):
        """
        Stage 1: per-word voting gate.

        Args:
            word_embeddings: (B, K, encoder_dim)
            word_mask: (B, K) bool, True = padding
        Returns:
            gated_tokens: (B, K, model_dim) — soft-gated gloss representations
            gates: (B, K) in [0, 1]
        """
        B, K, D = word_embeddings.shape

        pos_ids = torch.arange(K, device=word_embeddings.device)
        x = word_embeddings + self.pos_enc(pos_ids).unsqueeze(0)

        x = self.voting_transformer(x, src_key_padding_mask=word_mask)

        gates = torch.sigmoid(self.gate_head(x).squeeze(-1))  # (B, K)
        gates = gates.masked_fill(word_mask, 0.0)

        gated = gates.unsqueeze(-1) * x  # (B, K, encoder_dim)
        gated_tokens = self.gloss_proj(gated)  # (B, K, model_dim)

        return gated_tokens, gates

    def fuse(self, motion_tokens, gated_gloss_tokens, motion_mask=None, gloss_mask=None):
        """
        Stage 2: cross-attention fusion.

        Motion temporal features attend to gated gloss tokens.
        The attention mechanism implicitly learns temporal alignment.

        Args:
            motion_tokens: (B, T, model_dim)
            gated_gloss_tokens: (B, K, model_dim)
            motion_mask: (B, T) bool, True = padding (for tgt_key_padding_mask)
            gloss_mask: (B, K) bool, True = padding (for memory_key_padding_mask)
        Returns:
            fused: (B, T, model_dim) — per-frame conditioned features
        """
        fused = self.fusion_decoder(
            tgt=motion_tokens,
            memory=gated_gloss_tokens,
            tgt_key_padding_mask=motion_mask,
            memory_key_padding_mask=gloss_mask,
        )
        return fused

    def forward(self, word_embeddings, word_mask, motion_tokens, motion_mask=None):
        """
        Full pipeline: vote → fuse.

        Args:
            word_embeddings: (B, K, encoder_dim) from frozen CLIP per-word encoding
            word_mask: (B, K) bool, True = padding
            motion_tokens: (B, T, model_dim) projected motion features
            motion_mask: (B, T) bool, True = padding
        Returns:
            fused: (B, T, model_dim) — per-frame conditioned motion features
            gates: (B, K) — voting gate values for logging
        """
        gated_tokens, gates = self.vote(word_embeddings, word_mask)
        fused = self.fuse(motion_tokens, gated_tokens,
                          motion_mask=motion_mask, gloss_mask=word_mask)
        return fused, gates
