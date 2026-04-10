"""
test セットで学習済みモデルを評価する。
accuracy / precision / recall / F1 / AUC-ROC と混同行列を出力。

--threshold で判定閾値を指定（デフォルト 0.5）。
--sweep  で 0.3〜0.7 の閾値スイープを実行。
"""

import argparse

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


def evaluate_at_threshold(probs, labels, threshold: float) -> dict:
    preds = [1 if p > threshold else 0 for p in probs]
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "auc": roc_auc_score(labels, probs),
        "preds": preds,
    }


def print_results(metrics: dict, labels: list) -> None:
    preds = metrics["preds"]
    cm = confusion_matrix(labels, preds, labels=[1, 0])

    print(f"\n=== 評価結果（threshold={metrics['threshold']:.2f}）===")
    print(f"Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}  ← お気に入りの見逃し率に注目")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"AUC-ROC  : {metrics['auc']:.4f}")

    print("\n=== 目標値チェック ===")
    print(f"Accuracy >= 75%: {'OK' if metrics['accuracy'] >= 0.75 else 'NG'} ({metrics['accuracy']*100:.1f}%)")
    print(f"AUC-ROC >= 0.80: {'OK' if metrics['auc'] >= 0.80 else 'NG'} ({metrics['auc']:.4f})")
    print(f"Recall  >= 70%: {'OK' if metrics['recall'] >= 0.70 else 'NG'} ({metrics['recall']*100:.1f}%)")

    print("\n=== 混同行列（行=実際, 列=予測）===")
    print(f"{'':>14} {'予測:favorite':>14} {'予測:not_fav':>14}")
    print(f"{'実際:favorite':>14} {cm[0][0]:>14} {cm[0][1]:>14}")
    print(f"{'実際:not_fav':>14} {cm[1][0]:>14} {cm[1][1]:>14}")

    print("\n=== 詳細レポート ===")
    print(classification_report(labels, preds, labels=[1, 0], target_names=["favorite", "not_favorite"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5, help="判定閾値（デフォルト: 0.5）")
    parser.add_argument("--sweep", action="store_true", help="閾値スイープを実行")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["device"])
    processed_dir = cfg["data"]["processed_dir"]
    image_size = cfg["data"]["image_size"]
    model_path = cfg["train"]["model_save_path"]

    test_ds = get_dataset(processed_dir, "test", image_size)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    print(f"test: {len(test_ds)}枚, クラスマップ: {test_ds.class_to_idx}")

    dropout = cfg["model"].get("dropout", 0.3)
    base = cfg["model"].get("base", "efficientnet_b3")
    model = build_model(pretrained=False, dropout=dropout, base=base).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"モデルロード: {model_path}")

    probs, labels = run_inference(model, test_loader, device)

    if args.sweep:
        print("\n=== 閾値スイープ ===")
        print(f"{'threshold':>10} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
        print("-" * 55)
        best = None
        for t in [i / 10 for i in range(3, 8)]:
            m = evaluate_at_threshold(probs, labels, t)
            meets_recall = m["recall"] >= 0.70
            flag = " ← Recall OK" if meets_recall else ""
            print(
                f"{t:>10.2f} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} {m['f1']:>10.4f}{flag}"
            )
            # Recall >= 70% を満たしつつ Accuracy が最大のものを best とする
            if meets_recall and (best is None or m["accuracy"] > best["accuracy"]):
                best = m
        if best:
            print(f"\n最良閾値: {best['threshold']:.2f}（Recall≥70% 条件下で Accuracy 最大）")
            print_results(best, labels)
        else:
            print("\nRecall >= 70% を満たす閾値が見つかりませんでした。")
    else:
        metrics = evaluate_at_threshold(probs, labels, args.threshold)
        print_results(metrics, labels)


if __name__ == "__main__":
    main()
