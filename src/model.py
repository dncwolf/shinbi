"""
EfficientNet-B0 の最終層を付け替えた二値分類モデル。
学習時は logits を出力（BCEWithLogitsLoss に渡す）。
推論時は sigmoid を適用して確率に変換する。
"""

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b3,
)

# モデル別の classifier 入力次元
_CLASSIFIER_IN_FEATURES = {
    "efficientnet_b0": 1280,
    "efficientnet_b3": 1536,
}

_MODEL_REGISTRY = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
}


def build_model(pretrained: bool = True, dropout: float = 0.3, base: str = "efficientnet_b3") -> nn.Module:
    factory, default_weights = _MODEL_REGISTRY[base]
    weights = default_weights if pretrained else None
    model = factory(weights=weights)
    in_features = _CLASSIFIER_IN_FEATURES[base]
    # 最終層: Dropout → Linear(in_features, 1) → logit
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
