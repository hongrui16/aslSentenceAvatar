"""
BackTranslationModel  (v2 — anti-posterior-collapse)
=====================================================
Pose sequence → English sentence

Changes from v1:
    1. Temporal pooling: compress T pose tokens → N_POOL tokens (default 24)
       before feeding to T5 cross-attention
    2. Decoder token dropout: randomly mask teacher-forced tokens so the
       decoder cannot rely on autoregressive LM alone
    3. Two-stage freeze helpers: freeze / unfreeze T5 on demand

Architecture:
    Pose Encoder (Transformer) → Temporal Pool → Linear → T5 Decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput


# ═══════════════════════════════════════════════════════════════════════
# Positional encoding
# ═══════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # (1, max_len, d)

    def forward(self, x):                                     # x: (B, T, d)
        return x + self.pe[:, : x.size(1)]


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════
class BackTranslationModel(nn.Module):
    """
    Extra cfg fields (all have defaults):
        N_POOL          : number of tokens after temporal pooling (default 24)
        TOKEN_DROP_RATE : fraction of decoder tokens to mask      (default 0.0)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        input_dim  = cfg.INPUT_DIM
        model_dim  = getattr(cfg, "MODEL_DIM",     512)
        n_heads    = getattr(cfg, "N_HEADS",       8)
        n_layers   = getattr(cfg, "N_LAYERS",      4)
        dropout    = getattr(cfg, "DROPOUT",       0.1)
        t5_name    = getattr(cfg, "T5_MODEL_NAME", "t5-base")
        freeze_t5  = getattr(cfg, "FREEZE_T5",     False)

        self.n_pool         = getattr(cfg, "N_POOL",          24)
        self.token_drop_rate = getattr(cfg, "TOKEN_DROP_RATE", 0.0)

        # ── 1. Pose input projection + positional encoding ────────────
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pe         = PositionalEncoding(model_dim)
        self.encoder_dropout = nn.Dropout(dropout)

        # ── 2. Pose Transformer Encoder ───────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.pose_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers
        )

        # ── 3. Temporal pooling: T tokens → n_pool tokens ────────────
        #    Learnable query tokens + cross-attention (Perceiver-style)
        self.pool_queries = nn.Parameter(
            torch.randn(1, self.n_pool, model_dim) * 0.02
        )
        self.pool_cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_ln  = nn.LayerNorm(model_dim)
        self.pool_ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.Dropout(dropout),
        )
        self.pool_ln2 = nn.LayerNorm(model_dim)

        # ── 4. Project to T5 hidden dim ───────────────────────────────
        print(f"Loading T5: {t5_name} ...")
        self.t5     = T5ForConditionalGeneration.from_pretrained(t5_name)
        t5_dim      = self.t5.config.d_model               # 768 for t5-base

        self.pose_to_t5 = nn.Sequential(
            nn.Linear(model_dim, t5_dim),
            nn.LayerNorm(t5_dim),
        )

        # ── 5. Freeze / unfreeze T5 ──────────────────────────────────
        if freeze_t5:
            self.freeze_t5()

        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name)

        self._init_weights()

    # ─────────────────────────────────────────────────────────── init
    def _init_weights(self):
        for m in [self.input_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # pose_to_t5 is now Sequential, init first child
        nn.init.xavier_uniform_(self.pose_to_t5[0].weight)
        nn.init.zeros_(self.pose_to_t5[0].bias)

    # ─────────────────────────────────────────────────────────── freeze helpers
    def freeze_t5(self):
        for p in self.t5.parameters():
            p.requires_grad = False
        print("[BackTranslationModel] T5 weights FROZEN.")

    def unfreeze_t5(self):
        for p in self.t5.parameters():
            p.requires_grad = True
        print("[BackTranslationModel] T5 weights UNFROZEN.")

    @property
    def t5_frozen(self):
        return not next(self.t5.parameters()).requires_grad

    # ─────────────────────────────────────────────────────────── encoder
    def encode_pose(self, pose, padding_mask=None):
        """
        Args
            pose         : (B, T, D)
            padding_mask : (B, T)  True where padded
        Returns
            pooled       : (B, n_pool, t5_dim)
            pool_mask    : (B, n_pool)  all-False (no padding after pooling)
        """
        B = pose.size(0)

        x = self.input_proj(pose)                            # (B, T, model_dim)
        x = self.pe(x)
        x = self.encoder_dropout(x)
        x = self.pose_encoder(
            x, src_key_padding_mask=padding_mask
        )                                                    # (B, T, model_dim)

        # ── Perceiver-style cross-attention pooling ───────────────
        queries = self.pool_queries.expand(B, -1, -1)        # (B, n_pool, model_dim)

        # key_padding_mask for cross-attn: True = ignore
        pooled, _ = self.pool_cross_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
        )                                                    # (B, n_pool, model_dim)
        pooled = self.pool_ln(pooled + queries)              # residual + LN
        pooled = self.pool_ln2(pooled + self.pool_ffn(pooled))

        # Project to T5 dim
        pooled = self.pose_to_t5(pooled)                     # (B, n_pool, t5_dim)

        # After pooling there is no padding
        pool_mask = torch.zeros(
            B, self.n_pool, dtype=torch.bool, device=pose.device
        )
        return pooled, pool_mask

    # ─────────────────────────────────────────────────────────── forward
    def forward(self, pose, labels, padding_mask=None,
                token_drop_rate=None):
        """
        Args
            pose            : (B, T, D)
            labels          : (B, L)  with -100 for padding
            padding_mask    : (B, T)  True where padded
            token_drop_rate : override self.token_drop_rate for this call

        Returns
            loss : scalar
        """
        encoder_hidden, pool_mask = self.encode_pose(pose, padding_mask)

        # Attention mask for T5: 1 = attend, 0 = ignore
        encoder_attention_mask = (~pool_mask).long()          # (B, n_pool)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden
        )

        # ── Decoder token dropout ─────────────────────────────────
        drop_rate = token_drop_rate if token_drop_rate is not None \
                    else self.token_drop_rate

        decoder_input_ids = None
        if self.training and drop_rate > 0:
            # Build shifted decoder inputs manually
            decoder_input_ids = self.t5._shift_right(
                labels.clamp(min=0)
            )
            mask = torch.rand_like(decoder_input_ids.float()) < drop_rate
            mask[:, 0] = False                                # keep start token
            decoder_input_ids[mask] = self.t5.config.pad_token_id

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,              # None → T5 builds from labels
            labels=labels,
        )
        return outputs.loss

    # ─────────────────────────────────────────────────────────── generate
    @torch.no_grad()
    def generate(self, pose, padding_mask=None,
                 max_new_tokens=64, num_beams=4, early_stopping=True):
        """Returns List[str], length B."""
        self.eval()

        encoder_hidden, pool_mask = self.encode_pose(pose, padding_mask)
        encoder_attention_mask = (~pool_mask).long()

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden
        )

        token_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=True
        )