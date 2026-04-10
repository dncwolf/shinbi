"""
EfficientNet-B0 の最終層を付け替えた二値分類モデル。
学習時は logits を出力（BCEWithLogitsLoss に渡す）。
推論時は sigmoid を適用して確率に変換する。
"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    # 最終層: Dropout → Linear(1280, 1) → logit
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, 1),
    )
    return model


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
