"""
NIMA InceptionResNetV2 backbone で data/processed/ 以下の全画像を一度だけ推論し、
埋め込みベクトルを data/features/ に保存する。

train.py よりも前に一度だけ実行すること。
保存後は NIMA backbone のロードが不要になり、学習が数十倍高速になる。
"""

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import build_model, get_device


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract(model, dataset, device, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_features, all_labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for images, labels in loader:
        images = images.to(device)
        feats = model.backbone(images)
        pooled = model.global_pool(feats[-1])
        features = pooled.view(pooled.size(0), -1).float().cpu().numpy()
        all_features.append(features)
        all_labels.extend(labels.numpy())
    return np.concatenate(all_features), np.array(all_labels)


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["device"])
    print(f"デバイス: {device}")

    model_cfg = cfg["model"]
    model = build_model(
        embed_dim=model_cfg.get("embed_dim", 1536),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)

    processed_dir = cfg["data"]["processed_dir"]
    image_size = cfg["data"]["image_size"]
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        print(f"\n{split} セットを処理中...")
        ds = get_dataset(processed_dir, split, image_size, pre_resized=True)
        features, labels = extract(model, ds, device)
        save_path = features_dir / f"{split}.npz"
        np.savez(save_path, features=features, labels=labels)
        print(f"  保存: {save_path}  ({len(labels)}件, shape: {features.shape})")

    print("\n特徴量抽出完了。次は uv run python src/train.py を実行してください。")


if __name__ == "__main__":
    main()
