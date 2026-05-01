"""
VotingConditionModule
=====================
Per-gloss gate network for LLM draft pseudo-gloss refinement.

Input:  word-level CLIP embeddings (B, K, encoder_dim) + padding mask
Output: condition embedding (B, model_dim) + gate values (B, K)

The gate (sigmoid, 0-1) decides keep/drop for each draft gloss word.
Trained end-to-end via diffusion reconstruction loss.

Sentence conditioning modes (sent_cond_mode):
    'none'    — gloss only (original behavior)
    'kv_pool' — sentence token added to the gloss pool before gating
"""

import torch
import torch.nn as nn
import math


class VotingConditionModule(nn.Module):

    def __init__(self, encoder_dim, model_dim, n_layers=2, n_heads=4,
                 ff_mult=2, dropout=0.1, max_words=64,
                 sent_cond_mode='none', phono_dim=0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.model_dim = model_dim
        self.sent_cond_mode = sent_cond_mode

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

        self.phono_dim = phono_dim
        if phono_dim > 0:
            self.gloss_phono_proj = nn.Linear(encoder_dim + phono_dim, encoder_dim)

        if sent_cond_mode == 'kv_pool':
            self.sent_proj = nn.Linear(model_dim, encoder_dim)
            self.type_emb_sent = nn.Parameter(torch.randn(encoder_dim) * 0.02)
            self.type_emb_gloss = nn.Parameter(torch.randn(encoder_dim) * 0.02)

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

    def forward(self, word_embeddings, word_mask, sent_emb=None, phono_tokens=None):
        """
        Args:
            word_embeddings: (B, K, encoder_dim) from frozen CLIP per-word encoding
            word_mask:       (B, K) bool, True = padding position
            sent_emb:        (B, model_dim) projected sentence embedding (for kv_pool mode)
            phono_tokens:    (B, K, phono_dim) optional phono attribute vectors
        Returns:
            condition: (B, model_dim)
            gates:     (B, K) in [0, 1], 0 at padding positions
        """
        B, K, D = word_embeddings.shape

        pos_ids = torch.arange(K, device=word_embeddings.device)
        x = word_embeddings + self.pos_enc(pos_ids).unsqueeze(0)

        if self.sent_cond_mode == 'kv_pool' and sent_emb is not None:
            sent_token = self.sent_proj(sent_emb).unsqueeze(1)  # (B, 1, encoder_dim)
            sent_token = sent_token + self.type_emb_sent
            x = x + self.type_emb_gloss

            x = torch.cat([sent_token, x], dim=1)  # (B, 1+K, D)
            sent_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([sent_mask, word_mask], dim=1)  # (B, 1+K)
        else:
            full_mask = word_mask

        x = self.transformer(x, src_key_padding_mask=full_mask)

        # Fuse phono attributes after transformer, before gating
        if self.phono_dim > 0 and phono_tokens is not None:
            if self.sent_cond_mode == 'kv_pool' and sent_emb is not None:
                # x is (B, 1+K, D) — only fuse phono with gloss positions [1:]
                sent_part = x[:, :1, :]  # (B, 1, D)
                gloss_part = x[:, 1:, :]  # (B, K, D)
                fused = torch.cat([gloss_part, phono_tokens], dim=-1)
                gloss_part = self.gloss_phono_proj(fused)
                x = torch.cat([sent_part, gloss_part], dim=1)
            else:
                fused = torch.cat([x, phono_tokens], dim=-1)
                x = self.gloss_phono_proj(fused)

        gates = torch.sigmoid(self.gate_head(x).squeeze(-1))  # (B, 1+K) or (B, K)
        gates = gates.masked_fill(full_mask, 0.0)

        # weighted mean pool
        gate_sum = gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = (gates.unsqueeze(-1) * x).sum(dim=1) / gate_sum  # (B, D)

        condition = self.output_proj(pooled)

        if self.sent_cond_mode == 'kv_pool' and sent_emb is not None:
            gloss_gates = gates[:, 1:]  # strip sentence gate for logging
        else:
            gloss_gates = gates

        return condition, gloss_gates
