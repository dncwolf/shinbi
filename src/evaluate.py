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
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4)
    print(f"test: {len(test_ds)}枚, クラスマップ: {test_ds.class_to_idx}")

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"モデルロード: {model_path}")

    probs, labels = run_inference(model, test_loader, device)

    # favorite のインデックスを確認
    fav_idx = test_ds.class_to_idx.get("favorite", 0)
    # fav_idx=0 なら prob > 0.5 → favorite=0 の予測なので反転が必要か確認
    # BCEWithLogitsLoss は label=fav_idx=0 or 1 に合わせているので要注意
    # train.py では favorite ラベルを pos_weight で強調しているため、
    # sigmoid(logit) > 0.5 → positive (= fav_idx のクラス) を意味する
    preds = [1 if p > 0.5 else 0 for p in probs]

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=fav_idx, zero_division=0)
    rec = recall_score(labels, preds, pos_label=fav_idx, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=fav_idx, zero_division=0)

    # AUC: probs がお気に入りの確率として扱う
    if fav_idx == 1:
        auc_probs = probs
    else:
        auc_probs = [1 - p for p in probs]
    binary_labels = [1 if l == fav_idx else 0 for l in labels]
    auc = roc_auc_score(binary_labels, auc_probs)

    cm = confusion_matrix(labels, preds)

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

    print("\n=== 混同行列 ===")
    class_names = [k for k, v in sorted(test_ds.class_to_idx.items(), key=lambda x: x[1])]
    header = "         " + "  ".join(f"{n:>12}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>12}" for v in row)
        print(f"{class_names[i]:>8} {row_str}")

    print("\n=== 詳細レポート ===")
    print(classification_report(labels, preds, target_names=class_names))


if __name__ == "__main__":
    main()
