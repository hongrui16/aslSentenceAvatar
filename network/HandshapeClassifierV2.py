"""
Handshape Classifier: ResNet50 + Temporal Attention + MLP
==========================================================
- ResNet50 (pretrained) extracts per-frame features (2048-d)
- Temporal self-attention aggregates across frames
- MLP head outputs class logits

Supports both multi-frame (B, T, C, H, W) and single-frame (B, C, H, W) input.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class TemporalAttentionPool(nn.Module):
    """Single-layer self-attention over the temporal dimension, then weighted pool."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, T, D) → (B, D)"""
        out, _ = self.attn(x, x, x)
        out = self.norm(out + x)      # (B, T, D)
        return out.mean(dim=1)        # (B, D)


class HandshapeClassifierV2(nn.Module):
    """
    Args:
        num_classes:  number of handshape classes
        dropout:      dropout rate (default 0.3)
    """

    def __init__(self, num_classes, dropout=0.3, **kwargs):
        super().__init__()

        # --- ResNet50 backbone (remove final fc) ---
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 2048, 1, 1)
        self.cnn_dim = 2048

        # Freeze conv1 + bn1 + relu + maxpool + layer1 + layer2
        for name, param in self.cnn.named_parameters():
            # 0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1, 5=layer2
            if name.startswith(('0.', '1.', '2.', '3.', '4.', '5.')):
                param.requires_grad = False

        # --- Temporal attention ---
        self.temporal_attn = TemporalAttentionPool(self.cnn_dim, num_heads=4, dropout=dropout)

        # --- MLP classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) or (B, C, H, W)

        Returns:
            logits: (B, num_classes)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            features = self.cnn(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)
            features = features.reshape(B, T, -1)           # (B, T, 2048)
            features = self.temporal_attn(features)          # (B, 2048)
        else:
            features = self.cnn(x).squeeze(-1).squeeze(-1)  # (B, 2048)

        logits = self.classifier(features)
        return logits
