"""
config.yaml を読み込んで学習を実行する。
class_weight・early stopping・best model 保存に対応。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import build_model, get_device


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_labels(dataset) -> list[int]:
    return [label for _, label in dataset.samples]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["device"])
    print(f"デバイス: {device}")

    # データセット
    processed_dir = cfg["data"]["processed_dir"]
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["train"]["batch_size"]

    train_ds = get_dataset(processed_dir, "train", image_size)
    val_ds = get_dataset(processed_dir, "val", image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    print(f"train: {len(train_ds)}枚, val: {len(val_ds)}枚")
    print(f"クラスマップ: {train_ds.class_to_idx}")

    # class_weight（favoritクラスのインデックスに注意）
    labels = get_labels(train_ds)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    # ImageFolder はアルファベット順に class_to_idx を割り当てる
    # favorite=0, not_favorite=1 の場合は pos_weight = weights[0] / weights[1]
    fav_idx = train_ds.class_to_idx.get("favorite", 0)
    not_fav_idx = 1 - fav_idx
    pos_weight = torch.tensor(weights[fav_idx] / weights[not_fav_idx], dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # モデル
    model = build_model(pretrained=cfg["model"]["pretrained"]).to(device)
    lr = cfg["train"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学習
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["early_stopping_patience"]
    save_path = Path(cfg["train"]["model_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            mark = " ✓"
        else:
            no_improve += 1
            mark = ""

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}{mark} | "
            f"patience={no_improve}/{patience}"
        )

        if no_improve >= patience:
            print(f"Early stopping（{patience}epoch 改善なし）")
            break

    print(f"学習完了。best val_loss={best_val_loss:.4f}")
    print(f"モデル保存先: {save_path}")


if __name__ == "__main__":
    main()
