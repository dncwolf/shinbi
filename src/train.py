"""
config.yaml を読み込んで学習を実行する。

改善点:
  - Phase 1: backbone 凍結、classifier head のみ学習（freeze_epochs）
  - Phase 2: backbone 解凍、layer-wise LR（backbone_lr < head_lr）
  - ReduceLROnPlateau で val_loss 停滞時に lr 自動削減
  - Dropout による正則化（model.py 側）
  - num_workers=0（MPS との相性問題回避）
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


def freeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = True


def make_optimizer(model: nn.Module, backbone_lr: float, head_lr: float) -> torch.optim.Adam:
    return torch.optim.Adam([
        {"params": model.features.parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ])


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

    train_ds = get_dataset(processed_dir, "train", image_size, pre_resized=True)
    val_ds = get_dataset(processed_dir, "val", image_size, pre_resized=True)

    # num_workers=0: MPS 環境での multiprocessing semaphore 問題を回避
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"train: {len(train_ds)}枚, val: {len(val_ds)}枚")
    print(f"クラスマップ: {train_ds.class_to_idx}")

    # class_weight
    labels = get_labels(train_ds)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    fav_idx = train_ds.class_to_idx.get("favorite", 0)
    not_fav_idx = 1 - fav_idx
    pos_weight = torch.tensor(weights[fav_idx] / weights[not_fav_idx], dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # モデル
    dropout = cfg["model"].get("dropout", 0.3)
    model = build_model(pretrained=cfg["model"]["pretrained"], dropout=dropout).to(device)

    backbone_lr = cfg["train"]["backbone_lr"]
    head_lr = cfg["train"]["head_lr"]
    freeze_epochs = cfg["train"]["freeze_epochs"]

    # Phase 1: backbone 凍結（head のみ学習）
    freeze_backbone(model)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=head_lr)
    print(f"Phase 1: backbone 凍結（{freeze_epochs} epoch）")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["train"]["scheduler_patience"]
    )

    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["early_stopping_patience"]
    save_path = Path(cfg["train"]["model_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):

        # Phase 2 への切り替え
        if epoch == freeze_epochs + 1:
            unfreeze_backbone(model)
            optimizer = make_optimizer(model, backbone_lr, head_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=cfg["train"]["scheduler_patience"]
            )
            print(f"Phase 2: backbone 解凍（backbone_lr={backbone_lr}, head_lr={head_lr}）")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[-1]["lr"]
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            mark = " ✓"
        else:
            no_improve += 1
            mark = ""

        phase = 1 if epoch <= freeze_epochs else 2
        print(
            f"[P{phase}] Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}{mark} | "
            f"lr={current_lr:.2e} | "
            f"patience={no_improve}/{patience}"
        )

        if no_improve >= patience:
            print(f"Early stopping（{patience}epoch 改善なし）")
            break

    print(f"学習完了。best val_loss={best_val_loss:.4f}")
    print(f"モデル保存先: {save_path}")


if __name__ == "__main__":
    main()
