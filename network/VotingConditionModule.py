"""
VotingConditionModule
=====================
Per-gloss gate network for LLM draft pseudo-gloss refinement.

Input:  word-level CLIP embeddings (B, K, encoder_dim) + padding mask
Output: condition embedding (B, model_dim) + gate values (B, K)

The gate (sigmoid, 0-1) decides keep/drop for each draft gloss word.
Trained end-to-end via diffusion reconstruction loss.
"""

import torch
import torch.nn as nn
import math


class VotingConditionModule(nn.Module):

    def __init__(self, encoder_dim, model_dim, n_layers=2, n_heads=4,
                 ff_mult=2, dropout=0.1, max_words=64):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.model_dim = model_dim

        self.pos_enc = nn.Embedding(max_words, encoder_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=n_heads,
            dim_feedforward=encoder_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.gate_head = nn.Linear(encoder_dim, 1)

        self.output_proj = nn.Sequential(
            nn.Linear(encoder_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_enc.weight, std=0.02)
        # bias gate toward keeping (init bias > 0 so sigmoid > 0.5)
        nn.init.constant_(self.gate_head.bias, 1.0)

    def forward(self, word_embeddings, word_mask):
        """
        Args:
            word_embeddings: (B, K, encoder_dim) from frozen CLIP per-word encoding
            word_mask:       (B, K) bool, True = padding position
        Returns:
            condition: (B, model_dim)
            gates:     (B, K) in [0, 1], 0 at padding positions
        """
        B, K, D = word_embeddings.shape

        pos_ids = torch.arange(K, device=word_embeddings.device)
        x = word_embeddings + self.pos_enc(pos_ids).unsqueeze(0)

        x = self.transformer(x, src_key_padding_mask=word_mask)

        gates = torch.sigmoid(self.gate_head(x).squeeze(-1))  # (B, K)
        gates = gates.masked_fill(word_mask, 0.0)

        # weighted mean pool
        gate_sum = gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = (gates.unsqueeze(-1) * x).sum(dim=1) / gate_sum  # (B, D)

        condition = self.output_proj(pooled)
        return condition, gates
