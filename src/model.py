"""
Multi-backbone feature head。

特徴量抽出フェーズ（extract_features.py）で以下の 3 バックボーンを使用:
  1. CLIP ViT-L-14          : 768-dim （LAION Aesthetics 学習済み）
  2. NIMA Aesthetic         : 1536-dim（InceptionResNetV2, AVA データセット）
  3. NIMA Technical (KonIQ) : 1536-dim（InceptionResNetV2, KonIQ データセット）
  連結: 768 + 1536 + 1536 = 3840-dim

学習フェーズ（train.py）では事前抽出済みの 3840-dim ベクトルで AestheticsHead のみ学習。

アーキテクチャ設計:
  - 各バックボーンを独立した投影層でスライスして融合
  - clip_out=256, nima_out=128 により CLIP:NIMA = 50:50 の発言権に揃える
    （旧構造: CLIP 20%, NIMA 合計 80% → NIMA ノイズが支配的だった）
  - 融合後: BN(512) → Dropout → Linear(512→64) → ReLU → Dropout → Linear(64→1)
"""

import torch
import torch.nn as nn


class AestheticsHead(nn.Module):
    """各バックボーンを独立投影してから融合するhead。"""

    def __init__(
        self,
        clip_dim: int = 768,
        nima_dim: int = 1536,
        clip_out: int = 256,
        nima_out: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.nima_dim = nima_dim

        self.clip_proj = nn.Sequential(nn.Linear(clip_dim, clip_out), nn.ReLU())
        self.nima_aes_proj = nn.Sequential(nn.Linear(nima_dim, nima_out), nn.ReLU())
        self.nima_tech_proj = nn.Sequential(nn.Linear(nima_dim, nima_out), nn.ReLU())

        fused_dim = clip_out + nima_out * 2  # 256 + 128 + 128 = 512
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clip = x[:, : self.clip_dim]
        nima_aes = x[:, self.clip_dim : self.clip_dim + self.nima_dim]
        nima_tech = x[:, self.clip_dim + self.nima_dim :]

        fused = torch.cat(
            [self.clip_proj(clip), self.nima_aes_proj(nima_aes), self.nima_tech_proj(nima_tech)],
            dim=1,
        )
        return self.classifier(fused)


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
