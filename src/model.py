"""
Multi-backbone feature head。

特徴量抽出フェーズ（extract_features.py）で以下の 3 バックボーンを使用:
  1. CLIP ViT-L-14          : 768-dim （LAION Aesthetics 学習済み）
  2. NIMA Aesthetic         : 1536-dim（InceptionResNetV2, AVA データセット）
  3. NIMA Technical (KonIQ) : 1536-dim（InceptionResNetV2, KonIQ データセット）
  連結: 768 + 1536 + 1536 = 3840-dim

学習フェーズ（train.py）では事前抽出済みの 3840-dim ベクトルで AestheticsHead のみ学習。
"""

import torch
import torch.nn as nn


class AestheticsHead(nn.Module):
    """Binary classifier MLP head。
    高次元特徴（3840-dim）に対して過学習を防ぐため、
    入力 BN + 1 hidden layer + 強い Dropout の軽量構成。
    """

    def __init__(self, embed_dim: int = 3840, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
