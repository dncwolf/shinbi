"""
単一画像のお気に入り確率を推論する。

使い方:
    uv run python src/predict.py --image path/to/photo.jpg
"""

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image

from dataset import get_transform
from model import build_model, get_device


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def predict(image_path: str, cfg: dict) -> float:
    device = get_device(cfg["device"])
    model_path = cfg["train"]["model_save_path"]
    image_size = cfg["data"]["image_size"]

    dropout = cfg["model"].get("dropout", 0.3)
    base = cfg["model"].get("base", "efficientnet_b3")
    model = build_model(pretrained=False, dropout=dropout, base=base).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_transform("val", image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    return prob


def main() -> None:
    parser = argparse.ArgumentParser(description="お気に入り確率を推論する")
    parser.add_argument("--image", required=True, help="推論する画像のパス")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"エラー: 画像ファイルが見つかりません: {args.image}")
        raise SystemExit(1)

    cfg = load_config(args.config)
    prob = predict(args.image, cfg)
    print(f"お気に入り確率: {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
