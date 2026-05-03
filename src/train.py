"""
config.yaml を読み込んで学習を実行する。

前提: extract_features.py で data/features/ を事前生成しておくこと。
NIMA backbone（frozen）の事前抽出済み特徴量で AestheticsHead のみを学習する。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from model import AestheticsHead, get_device


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_features(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    return (
        torch.tensor(data["features"], dtype=torch.float32),
        torch.tensor(data["labels"], dtype=torch.long),
    )


def train_one_epoch(
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    head.train()
    total_loss = 0.0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    head.eval()
    total_loss = 0.0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        logits = head(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["device"])
    print(f"デバイス: {device}")

    features_dir = Path("data/features")
    if not features_dir.exists():
        print("エラー: data/features/ が見つかりません。")
        print("先に: uv run python src/extract_features.py を実行してください。")
        raise SystemExit(1)

    train_features, train_labels = load_features(features_dir / "train.npz")
    val_features, val_labels = load_features(features_dir / "val.npz")

    batch_size = cfg["train"]["batch_size"]
    train_ds = TensorDataset(train_features, train_labels)
    val_ds = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"train: {len(train_ds)}件, val: {len(val_ds)}件")

    labels_np = train_labels.numpy()
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_np)
    n_fav = int((labels_np == 1).sum())
    n_not = int((labels_np == 0).sum())
    print(f"favorite: {n_fav}, not_favorite: {n_not}")
    pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model_cfg = cfg["model"]
    head = AestheticsHead(
        embed_dim=model_cfg.get("embed_dim", 1536),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)

    lr = cfg["train"]["learning_rate"]
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["train"]["scheduler_patience"]
    )

    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["early_stopping_patience"]
    save_path = Path(cfg["train"]["model_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("モデル: CLIP ViT-L-14 + NIMA Aesthetic + NIMA Technical (all frozen) + AestheticsHead (head のみ学習)")

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(head, train_loader, optimizer, criterion, device)
        val_loss = validate(head, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(head.state_dict(), save_path)
            mark = " ✓"
        else:
            no_improve += 1
            mark = ""

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}{mark} | "
            f"lr={current_lr:.2e} | "
            f"patience={no_improve}/{patience}"
        )

        if no_improve >= patience:
            print(f"Early stopping（{patience}epoch 改善なし）")
            break

    print(f"学習完了。best val_loss={best_val_loss:.4f}")
    print(f"Head 保存先: {save_path}")


if __name__ == "__main__":
    main()
