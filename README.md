# shinbi

手持ちの写真にお気に入りの印をつけたデータをもとに、画像がお気に入りになる確率（0〜100%）を出力する二値分類モデル。

## 技術スタック

| 項目 | 内容 |
|------|------|
| 言語 | Python 3.13 |
| パッケージ管理 | uv |
| フレームワーク | PyTorch |
| モデル | EfficientNet-B3（ImageNet 転移学習） |
| デバイス | Apple Silicon MPS |
| 設定 | config.yaml |

## セットアップ

```bash
uv sync
```

## 実行フロー

```bash
# 1. データを配置
#    data/raw/fav/        ← お気に入り画像
#    data/raw/not_fav/    ← 非お気に入り画像

# 2. 前処理（リサイズ・JPEG 圧縮）
uv run python src/preprocess_images.py

# 3. train / val / test に分割
uv run python src/split_data.py

# 4. 学習
uv run python src/train.py

# 5. 評価
uv run python src/evaluate.py

# 6. 推論
uv run python src/predict.py --image path/to/photo.jpg
# → お気に入り確率: 83.4%
```

## データ仕様

| クラス | 目安枚数 |
|--------|----------|
| お気に入り（fav） | 549 枚 |
| 非お気に入り（not_fav） | 約 1100 枚（1:2 比率） |

分割比率：train 70% / val 15% / test 15%

## 学習設定（config.yaml）

| パラメータ | 値 |
|------------|-----|
| epochs | 50 |
| batch_size | 16 |
| freeze_epochs | 10（head のみ学習） |
| backbone_lr | 1e-5 |
| head_lr | 1e-4 |
| dropout | 0.3 |
| early_stopping_patience | 15 |

## 評価目標

| 指標 | 目標 |
|------|------|
| Accuracy | 75% 以上 |
| AUC-ROC | 0.80 以上 |
| Recall（お気に入り） | 70% 以上 |

お気に入りを見逃す方がストレスが大きいため、Recall を重視する。
