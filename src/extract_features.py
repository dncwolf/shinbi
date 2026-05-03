"""
CLIP ViT-L-14 + NIMA Aesthetic + NIMA Technical の 3 バックボーンで特徴量を抽出・連結して保存。

  CLIP ViT-L-14     (768-dim, L2 正規化)
  NIMA Aesthetic    (1536-dim, L2 正規化, AVA 学習済み)
  NIMA Technical    (1536-dim, L2 正規化, KonIQ 学習済み)
  ─────────────────────────────────────────────────────
  連結ベクトル        3840-dim

各バックボーンは常に frozen / eval。train.py より前に一度だけ実行すること。
"""

from pathlib import Path

import numpy as np
import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import get_raw_dataset
from model import get_device

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
NIMA_MEAN = [0.5, 0.5, 0.5]
NIMA_STD = [0.5, 0.5, 0.5]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_backbones(device: torch.device):
    """3 バックボーンをロードして frozen eval で返す。"""
    # CLIP ViT-L-14（laion_aes 経由でロード）
    laion_model = pyiqa.create_metric("laion_aes", device=device)
    clip_model: nn.Module = laion_model.net.clip_model[0].to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # NIMA Aesthetic (AVA)
    nima_aes_net = pyiqa.create_metric("nima", device=device).net
    nima_aes_backbone = nima_aes_net.base_model.to(device)
    nima_aes_pool = nima_aes_net.global_pool.to(device)
    nima_aes_backbone.eval()
    nima_aes_pool.eval()
    for p in list(nima_aes_backbone.parameters()) + list(nima_aes_pool.parameters()):
        p.requires_grad = False

    # NIMA Technical (KonIQ)
    nima_tech_net = pyiqa.create_metric("nima-koniq", device=device).net
    nima_tech_backbone = nima_tech_net.base_model.to(device)
    nima_tech_pool = nima_tech_net.global_pool.to(device)
    nima_tech_backbone.eval()
    nima_tech_pool.eval()
    for p in list(nima_tech_backbone.parameters()) + list(nima_tech_pool.parameters()):
        p.requires_grad = False

    return clip_model, (nima_aes_backbone, nima_aes_pool), (nima_tech_backbone, nima_tech_pool)


clip_norm = transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
nima_norm = transforms.Normalize(NIMA_MEAN, NIMA_STD)


@torch.no_grad()
def extract(
    clip_model,
    nima_aes: tuple,
    nima_tech: tuple,
    dataset,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """3 バックボーンから特徴量を抽出して連結・返却。"""
    nima_aes_backbone, nima_aes_pool = nima_aes
    nima_tech_backbone, nima_tech_pool = nima_tech

    all_features, all_labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for raw_images, labels in loader:
        raw_images = raw_images.to(device)

        # CLIP: OPENAI 統計で正規化
        clip_images = clip_norm(raw_images)
        clip_emb = clip_model.encode_image(clip_images)
        clip_emb = F.normalize(clip_emb.float(), p=2, dim=-1)  # (B, 768)

        # NIMA: 両モデルとも 0.5/0.5 で正規化
        nima_images = nima_norm(raw_images)

        aes_feats = nima_aes_backbone(nima_images)
        aes_pooled = nima_aes_pool(aes_feats[-1])
        aes_emb = F.normalize(aes_pooled.view(aes_pooled.size(0), -1).float(), p=2, dim=-1)  # (B, 1536)

        tech_feats = nima_tech_backbone(nima_images)
        tech_pooled = nima_tech_pool(tech_feats[-1])
        tech_emb = F.normalize(tech_pooled.view(tech_pooled.size(0), -1).float(), p=2, dim=-1)  # (B, 1536)

        combined = torch.cat([clip_emb, aes_emb, tech_emb], dim=1)  # (B, 3840)
        all_features.append(combined.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_features), np.array(all_labels)


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["device"])
    print(f"デバイス: {device}")
    print("バックボーン: CLIP ViT-L-14 (768) + NIMA Aesthetic (1536) + NIMA Technical KonIQ (1536) = 3840-dim")

    print("バックボーンをロード中...")
    clip_model, nima_aes, nima_tech = build_backbones(device)

    processed_dir = cfg["data"]["processed_dir"]
    image_size = cfg["data"]["image_size"]
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        print(f"\n{split} セットを処理中...")
        ds = get_raw_dataset(processed_dir, split, image_size)
        features, labels = extract(clip_model, nima_aes, nima_tech, ds, device)
        save_path = features_dir / f"{split}.npz"
        np.savez(save_path, features=features, labels=labels)
        print(f"  保存: {save_path}  ({len(labels)}件, shape: {features.shape})")

    print("\n特徴量抽出完了。次は uv run python src/train.py を実行してください。")


if __name__ == "__main__":
    main()
