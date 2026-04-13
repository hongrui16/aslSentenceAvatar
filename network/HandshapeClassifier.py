"""
Handshape Classifier: ResNet34 + MLP
=====================================
- ResNet34 (pretrained) extracts per-frame features (512-d)
- Mean pooling over frames
- MLP head outputs class logits

Supports both multi-frame (B, T, C, H, W) and single-frame (B, C, H, W) input.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class HandshapeClassifier(nn.Module):
    """
    Args:
        num_classes:  number of handshape classes
        dropout:      dropout rate (default 0.3)
    """

    def __init__(self, num_classes, dropout=0.3, **kwargs):
        super().__init__()

        # --- ResNet34 backbone (remove final fc) ---
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1)
        self.cnn_dim = 512

        # Freeze early layers (conv1 + layer1), fine-tune the rest
        for name, param in self.cnn.named_parameters():
            if name.startswith('0.') or name.startswith('1.') or name.startswith('2.') or name.startswith('3.'):
                param.requires_grad = False
            elif name.startswith('4.'):
                param.requires_grad = False

        # --- MLP classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
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
            features = self.cnn(x).squeeze(-1).squeeze(-1)  # (B*T, 512)
            features = features.reshape(B, T, -1)           # (B, T, 512)
            features = features.mean(dim=1)                 # (B, 512)
        else:
            features = self.cnn(x).squeeze(-1).squeeze(-1)  # (B, 512)

        logits = self.classifier(features)
        return logits
