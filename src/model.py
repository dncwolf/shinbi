"""
CLIP ViT-L-14（frozen）+ LAION Aesthetics Predictor V2 スタイルの MLP head。
学習時は logits を出力（BCEWithLogitsLoss に渡す）。
推論時は sigmoid を適用して確率に変換する。
保存・ロード対象は head の state_dict のみ（CLIP 重みは変化しないため）。
"""

import open_clip
import torch
import torch.nn as nn


class AestheticsHead(nn.Module):
    """LAION Aesthetics Predictor V2 と同じ MLP 構造（768→512→256→128→64→1）"""

    def __init__(self, embed_dim: int = 768, dropout: float = 0.2):
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


class CLIPAestheticsModel(nn.Module):
    """CLIP encoder（常に frozen + eval）+ 学習可能な AestheticsHead"""

    def __init__(self, clip_model: nn.Module, head: AestheticsHead):
        super().__init__()
        self.clip = clip_model
        self.head = head

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # CLIP は常に eval（BN 統計を更新しない）
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.clip.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
        return self.head(features.float())


def build_model(
    backbone: str = "ViT-L-14",
    pretrained: str = "openai",
    embed_dim: int = 768,
    dropout: float = 0.2,
) -> nn.Module:
    clip_model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    head = AestheticsHead(embed_dim=embed_dim, dropout=dropout)
    return CLIPAestheticsModel(clip_model, head)


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
