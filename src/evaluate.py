"""
test セットで学習済みモデルを評価する。
accuracy / precision / recall / F1 / AUC-ROC と混同行列を出力。
"""

import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import build_model, get_device


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[list, list]:
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.tolist())
    return all_probs, all_labels


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["device"])
    processed_dir = cfg["data"]["processed_dir"]
    image_size = cfg["data"]["image_size"]
    model_path = cfg["train"]["model_save_path"]

    test_ds = get_dataset(processed_dir, "test", image_size)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    print(f"test: {len(test_ds)}枚, クラスマップ: {test_ds.class_to_idx}")

    dropout = cfg["model"].get("dropout", 0.3)
    model = build_model(pretrained=False, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"モデルロード: {model_path}")

    probs, labels = run_inference(model, test_loader, device)

    # dataset.py の target_transform で favorite=1, not_favorite=0 に反転済み。
    # prob = sigmoid(logit) = P(favorite)、pos_label=1 で favorite を正例として評価。
    preds = [1 if p > 0.5 else 0 for p in probs]

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(labels, probs)

    cm = confusion_matrix(labels, preds, labels=[1, 0])

    print("\n=== 評価結果 ===")
    print(f"Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}  ← お気に入りの見逃し率に注目")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    print("\n=== 目標値チェック ===")
    print(f"Accuracy >= 75%: {'OK' if acc >= 0.75 else 'NG'} ({acc*100:.1f}%)")
    print(f"AUC-ROC >= 0.80: {'OK' if auc >= 0.80 else 'NG'} ({auc:.4f})")
    print(f"Recall  >= 70%: {'OK' if rec >= 0.70 else 'NG'} ({rec*100:.1f}%)")

    print("\n=== 混同行列（行=実際, 列=予測）===")
    print(f"{'':>14} {'予測:favorite':>14} {'予測:not_fav':>14}")
    print(f"{'実際:favorite':>14} {cm[0][0]:>14} {cm[0][1]:>14}")
    print(f"{'実際:not_fav':>14} {cm[1][0]:>14} {cm[1][1]:>14}")

    print("\n=== 詳細レポート ===")
    print(classification_report(labels, preds, labels=[1, 0], target_names=["favorite", "not_favorite"]))


if __name__ == "__main__":
    main()
