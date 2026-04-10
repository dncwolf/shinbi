# shinbi

## 概要

手持ちの写真にお気に入りの印をつけたデータをもとに、
新しい画像がお気に入りになる確率（0〜100%）を出力する二値分類モデル。

---

## 技術スタック

| 項目 | 内容 |
|------|------|
| 言語 | Python 3.11 |
| パッケージ管理 | uv |
| 学習フレームワーク | PyTorch |
| モデル | EfficientNet-B0（転移学習） |
| デバイス | Apple Silicon MPS（CPU fallback） |
| 設定管理 | config.yaml |

---

## データ仕様

### 構成

| クラス | 枚数 | 内容 |
|--------|------|------|
| お気に入り | 549枚 | ラベル1 |
| 非お気に入り | 4303枚 → 1000-1200枚に間引き | ラベル0 |
| 合計目安 | 1700枚 | 比率1:2〜1:3 |

### 非お気に入りサンプルの構成比

| 種類 | 割合 |
|------|------|
| 同シーンで選ばなかった写真 | 70〜80% |
| 異なるシーンのランダム写真 | 20〜30% |

### フォルダ構成

```
data/
├── raw/
│   ├── favorite/
│   └── not_favorite/
└── processed/
    ├── train/
    │   ├── favorite/
    │   └── not_favorite/
    ├── val/
    │   ├── favorite/
    │   └── not_favorite/
    └── test/
        ├── favorite/
        └── not_favorite/
```

### データ分割比率

| セット | 割合 |
|--------|------|
| train | 70% |
| val | 15% |
| test | 15% |

---

## 前処理・Data Augmentation

### train（水増しあり）

```python
transforms.Resize((224, 224)),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(15),
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]),
```

### val / test（水増しなし）

```python
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]),
```

---

## モデル仕様

### アーキテクチャ

- ベース：EfficientNet-B0（ImageNet pretrained）
- 最終層：`nn.Linear → nn.Sigmoid`
- 出力：0〜1のスカラー値（お気に入り確率）

```python
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Sequential(
    nn.Linear(1280, 1),
    nn.Sigmoid()
)
```

### 学習設定

| パラメータ | 値 |
|------|------|
| optimizer | Adam |
| learning_rate | 1e-4 |
| epochs | 50 |
| batch_size | 16 |
| loss | BCEWithLogitsLoss（class_weight付き） |
| early_stopping | val_lossが10epoch改善しなければ停止 |

### class_weight

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=[0, 1], y=labels)
pos_weight = torch.tensor(weights[1] / weights[0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## config.yaml

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  split:
    train: 0.70
    val: 0.15
    test: 0.15
  image_size: 224

model:
  base: efficientnet_b0
  pretrained: true

train:
  epochs: 50
  batch_size: 16
  learning_rate: 0.0001
  early_stopping_patience: 10
  model_save_path: models/best_model.pth

device: mps
```

---

## スクリプト構成

### src/split_data.py

- `data/raw/` を読み込み
- 非お気に入りをランダム間引き（比率調整）
- train / val / test に分割して `data/processed/` に保存

### src/dataset.py

- `ImageFolder` ベースの Dataset クラス
- transform を train / val / test で切り替え

### src/model.py

- EfficientNet-B0 のロードと最終層の付け替え
- デバイス（MPS / CPU）の自動判定

### src/train.py

- config.yaml を読み込み
- class_weight 計算
- 学習ループ（train / val）
- best_model.pth の保存
- early stopping

### src/evaluate.py

- test セットで最終評価
- accuracy / precision / recall / F1 / AUC を出力
- 混同行列を表示

### src/predict.py

- 画像パスを受け取り確率（%）を返す
- 使い方：

```bash
uv run python src/predict.py --image path/to/photo.jpg
# → お気に入り確率: 83.4%
```

---

## 評価指標

| 指標 | 目標値 |
|------|------|
| Accuracy | 75%以上 |
| AUC-ROC | 0.80以上 |
| Recall（お気に入り） | 70%以上（見逃しを減らす） |

Recallを重視する理由：お気に入りを「お気に入りじゃない」と判定するミスの方がストレスが大きいため。

---

## 実行フロー

```
1. data/raw/ に画像を配置
        ↓
2. uv run python src/split_data.py
        ↓
3. uv run python src/train.py
        ↓
4. uv run python src/evaluate.py
        ↓
5. uv run python src/predict.py --image xxx.jpg
```

---
