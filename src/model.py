"""
NIMA（Neural Image Assessment）backbone（frozen）+ binary classifier head。

Backbone: pyiqa の NIMA — InceptionResNetV2（AVA データセットで事前学習済み）。
Head: 1536→512→256→128→64→1 の MLP。
保存・ロード対象は head の state_dict のみ（backbone 重みは変化しないため）。
"""

import pyiqa
import torch
import torch.nn as nn


class AestheticsHead(nn.Module):
    """Binary classifier MLP head"""

    def __init__(self, embed_dim: int = 1536, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NIMAModel(nn.Module):
    """NIMA InceptionResNetV2（常に frozen + eval）+ 学習可能な AestheticsHead"""

    def __init__(self, backbone: nn.Module, global_pool: nn.Module, head: AestheticsHead):
        super().__init__()
        self.backbone = backbone
        self.global_pool = global_pool
        self.head = head

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()   # backbone は常に eval（BN 統計を更新しない）
        self.global_pool.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(images)
            pooled = self.global_pool(feats[-1])          # [B, 1536, 1, 1]
            features = pooled.view(pooled.size(0), -1)    # [B, 1536]
        return self.head(features.float())


def build_model(embed_dim: int = 1536, dropout: float = 0.2) -> nn.Module:
    nima_net = pyiqa.create_metric("nima", device=torch.device("cpu")).net
    backbone = nima_net.base_model
    global_pool = nima_net.global_pool
    # backbone を凍結
    backbone.eval()
    global_pool.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    head = AestheticsHead(embed_dim=embed_dim, dropout=dropout)
    return NIMAModel(backbone, global_pool, head)


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
