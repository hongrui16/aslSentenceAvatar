"""
VotingFusionModule
==================
Cross-attention fusion for gloss-conditioned motion generation.

Stage 1 — Gloss Encoding:
    LLM draft gloss words → per-word CLIP embedding → transformer encoder
    Contextualizes gloss tokens before cross-attention.

Stage 2 — Cross-Attention Fusion:
    Motion temporal features (B, T, D) cross-attend to gloss tokens (B, K, D).
    Attention implicitly learns both token selection and temporal alignment.

Sentence conditioning modes (sent_cond_mode):
    'none'    — gloss only, no sentence (original behavior)
    'prefix'  — sentence token prepended to motion before denoiser (路径一)
    'kv_pool' — sentence token concatenated into K/V pool alongside gloss (路径二)

Output: per-frame condition (B, T, D) instead of a single vector.
"""

import torch
import torch.nn as nn
import math


class VotingFusionModule(nn.Module):

    def __init__(self, encoder_dim, model_dim, n_voting_layers=2, n_voting_heads=4,
                 n_fusion_layers=2, n_fusion_heads=8,
                 ff_mult=2, dropout=0.1, max_words=64,
                 sent_cond_mode='none', phono_dim=0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.model_dim = model_dim
        self.sent_cond_mode = sent_cond_mode

        # ---- Stage 1: Gloss Encoder (operates in encoder_dim space) ----
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

        # ---- kv_pool mode: type embeddings to distinguish sentence vs gloss ----
        if sent_cond_mode == 'kv_pool':
            self.type_emb_sent = nn.Parameter(torch.randn(model_dim) * 0.02)
            self.type_emb_gloss = nn.Parameter(torch.randn(model_dim) * 0.02)

        # ---- phono fusion: concat gloss + phono → project back to model_dim ----
        self.phono_dim = phono_dim
        if phono_dim > 0:
            self.gloss_phono_proj = nn.Linear(model_dim + phono_dim, model_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_enc.weight, std=0.02)

    def encode_gloss(self, word_embeddings, word_mask, phono_tokens=None):
        """
        Stage 1: contextualize gloss tokens, optionally fuse with phono.

        Args:
            word_embeddings: (B, K, encoder_dim)
            word_mask: (B, K) bool, True = padding
            phono_tokens: (B, K, phono_dim) optional phono attribute vectors
        Returns:
            gloss_tokens: (B, K, model_dim)
        """
        B, K, D = word_embeddings.shape

        pos_ids = torch.arange(K, device=word_embeddings.device)
        x = word_embeddings + self.pos_enc(pos_ids).unsqueeze(0)

        x = self.voting_transformer(x, src_key_padding_mask=word_mask)

        gloss_tokens = self.gloss_proj(x)  # (B, K, model_dim)

        if self.phono_dim > 0 and phono_tokens is not None:
            fused = torch.cat([gloss_tokens, phono_tokens], dim=-1)  # (B, K, model_dim + phono_dim)
            gloss_tokens = self.gloss_phono_proj(fused)  # (B, K, model_dim)

        return gloss_tokens

    def fuse(self, motion_tokens, gloss_tokens, motion_mask=None, gloss_mask=None,
             sent_token=None):
        """
        Stage 2: cross-attention fusion.

        Args:
            motion_tokens: (B, T, model_dim)
            gloss_tokens: (B, K, model_dim)
            motion_mask: (B, T) bool, True = padding
            gloss_mask: (B, K) bool, True = padding
            sent_token: (B, 1, model_dim) projected sentence embedding (for kv_pool mode)
        Returns:
            fused: (B, T, model_dim) — per-frame conditioned features
        """
        if self.sent_cond_mode == 'kv_pool' and sent_token is not None:
            B = gloss_tokens.shape[0]
            device = gloss_tokens.device

            gloss_kv = gloss_tokens + self.type_emb_gloss
            sent_kv = sent_token + self.type_emb_sent  # (B, 1, D)

            memory = torch.cat([sent_kv, gloss_kv], dim=1)  # (B, 1+K, D)

            sent_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            if gloss_mask is not None:
                memory_mask = torch.cat([sent_mask, gloss_mask], dim=1)
            else:
                memory_mask = None
        else:
            memory = gloss_tokens
            memory_mask = gloss_mask

        fused = self.fusion_decoder(
            tgt=motion_tokens,
            memory=memory,
            tgt_key_padding_mask=motion_mask,
            memory_key_padding_mask=memory_mask,
        )
        return fused

    def forward(self, word_embeddings, word_mask, motion_tokens, motion_mask=None,
                sent_token=None, phono_tokens=None):
        """
        Full pipeline: encode → fuse.

        Args:
            word_embeddings: (B, K, encoder_dim) from frozen CLIP per-word encoding
            word_mask: (B, K) bool, True = padding
            motion_tokens: (B, T, model_dim) projected motion features
            motion_mask: (B, T) bool, True = padding
            sent_token: (B, 1, model_dim) projected sentence embedding (for kv_pool mode)
            phono_tokens: (B, K, phono_dim) optional phono attribute vectors
        Returns:
            fused: (B, T, model_dim) — per-frame conditioned motion features
        """
        gloss_tokens = self.encode_gloss(word_embeddings, word_mask, phono_tokens=phono_tokens)
        fused = self.fuse(motion_tokens, gloss_tokens,
                          motion_mask=motion_mask, gloss_mask=word_mask,
                          sent_token=sent_token)
        return fused
