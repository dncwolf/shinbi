# shinbi

手持ちの写真にお気に入りの印をつけたデータをもとに、画像がお気に入りになる確率（0〜100%）を出力する二値分類モデル。

## 技術スタック

| 項目 | 内容 |
|------|------|
| 言語 | Python 3.13 |
| パッケージ管理 | uv |
| フレームワーク | PyTorch |
| バックボーン | CLIP ViT-L-14 + NIMA Aesthetic + NIMA Technical（3バックボーン融合） |
| デバイス | Apple Silicon MPS |
| 設定 | config.yaml |

## アーキテクチャ

特徴量抽出（frozen）と分類ヘッドの学習を分離した 2フェーズ方式。

```
CLIP ViT-L-14          →  768-dim（意味的特徴、L2正規化）
NIMA Aesthetic         → 1536-dim（美的品質、AVA学習済み、L2正規化）
NIMA Technical (KonIQ) → 1536-dim（技術的品質、KonIQ学習済み、L2正規化）
                          ─────────────
                           3840-dim 連結
                               ↓
              BN → Linear(128) → ReLU → Dropout(0.5) → Linear(1)
```

## セットアップ

```bash
uv sync
```

## 実行フロー

```bash
# 1. データを配置
#    data/raw/fav/        ← お気に入り画像
#    data/raw/not_fav/    ← 非お気に入り画像

# 2. 前処理（224×224 JPEG にリサイズ・圧縮）
uv run python src/preprocess_images.py

# 3. train / val / test に分割
uv run python src/split_data.py

# 4. 特徴量抽出（3バックボーンで一度だけ実行）
uv run python src/extract_features.py

# 5. 学習（事前抽出済み特徴量で分類ヘッドのみ学習）
uv run python src/train.py

# 6. 評価
uv run python src/evaluate.py --sweep

# 7. 推論
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
| embed_dim | 3840 |
| dropout | 0.5 |
| epochs | 50 |
| batch_size | 32 |
| learning_rate | 1e-4 |
| weight_decay | 1e-3 |
| early_stopping_patience | 15 |

## 評価目標・達成状況

| 指標 | 目標 | 最良結果（実行 #8） |
|------|------|---------------------|
| Accuracy | 75% 以上 | 59.7% |
| AUC-ROC | 0.80 以上 | **0.7218**（全実行最高） |
| Recall（お気に入り） | 70% 以上 | **71.1%** ✓ |

お気に入りを見逃す方がストレスが大きいため、Recall を重視する。

## 実験履歴

| 実行 | バックボーン | AUC-ROC |
|------|--------------|---------|
| #1〜#5 | EfficientNet-B0/B3 | 0.61〜0.65 |
| #6 | CLIP ViT-L-14 | 0.6773 |
| #7 | NIMA Aesthetic のみ | 0.5915 |
| **#8** | **CLIP + NIMA Aesthetic + NIMA Technical** | **0.7218** |

詳細は [log.md](log.md) を参照。
