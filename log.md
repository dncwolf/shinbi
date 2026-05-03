# 学習ログ

## 実行 #1 — 2026-04-10

### 設定

| パラメータ | 値 |
|---|---|
| モデル | EfficientNet-B0（ImageNet pretrained） |
| optimizer | Adam |
| learning_rate | 1e-4 |
| batch_size | 16 |
| epochs（予定） | 50 |
| early_stopping_patience | 10 |
| loss | BCEWithLogitsLoss（pos_weight=2.0052） |
| デバイス | MPS |

### データ

| セット | favorite | not_favorite | 合計 |
|---|---|---|---|
| train | 384 | 770 | 1154 |
| val | 82 | 165 | 247 |
| test | 83 | 165 | 248 |

### 結果（手動停止：Epoch 8）

| Epoch | train_loss | val_loss | patience |
|---|---|---|---|
| 1 | 0.9062 | **0.8301** ✓ | 0/10 |
| 2 | 0.7395 | 0.8316 | 1/10 |
| 3 | 0.6996 | 0.8334 | 2/10 |
| 4 | 0.6208 | 0.8616 | 3/10 |
| 5 | 0.6314 | 0.8900 | 4/10 |
| 6 | 0.5839 | 0.8871 | 5/10 |
| 7 | 0.5507 | 0.9868 | 6/10 |
| 8 | 0.5298 | 1.0070 | 7/10 |

- best val_loss: **0.8301**（Epoch 1）
- train_loss は順調に低下、val_loss は Epoch 1 以降一度も改善せず
- **過学習**が顕著

---

## 改善策

### 問題の診断

Epoch 1 で val_loss が最良になり、以降は train_loss が下がる一方で val_loss が上昇し続けた。
典型的な「転移学習初期の過学習」パターン。

考えられる原因：

1. **learning_rate が高すぎる** — 1e-4 は全パラメータ更新には大きく、pretrained weight を壊している
2. **Dropout がない** — EfficientNet-B0 の classifier に Dropout がなく正則化が弱い
3. **Data Augmentation が弱い** — val に対して train データが少なく、汎化しにくい
4. **全層を同じ lr で更新している** — backbone は小さな lr で fine-tune すべき

### 改善案（優先順位順）

#### 1. Backbone を凍結してから段階的に解凍する（最優先）

最初の数 epoch は classifier 層だけ学習し、backbone は凍結。
その後 backbone を解凍して全体を小さな lr で fine-tune する。

```python
# Phase 1: classifier のみ学習（Epoch 1-5）
for param in model.features.parameters():
    param.requires_grad = False

# Phase 2: 全層を解凍（Epoch 6〜）
for param in model.parameters():
    param.requires_grad = True
```

#### 2. Learning Rate を下げる＋Layer-wise LR を設定する

```python
optimizer = torch.optim.Adam([
    {"params": model.features.parameters(), "lr": 1e-5},  # backbone: 小さく
    {"params": model.classifier.parameters(), "lr": 1e-4}, # head: 大きく
])
```

#### 3. Dropout を追加する

```python
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 1),
)
```

#### 4. LR Scheduler を導入する（ReduceLROnPlateau）

val_loss が改善しない場合に lr を自動で下げる。

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
# val 後に呼ぶ
scheduler.step(val_loss)
```

#### 5. Data Augmentation を強化する

```python
transforms.RandomGrayscale(p=0.1),
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
```

### 次回実行計画

1. Backbone 凍結 + Layer-wise LR（改善案 1+2）を先に試す
2. val_loss が改善しなければ Dropout 追加（改善案 3）
3. それでも駄目なら LR Scheduler + Augmentation 強化（改善案 4+5）

---

## 実行 #2 — 2026-04-10

### 実施した改善

| 項目 | 内容 |
|---|---|
| Backbone 凍結 | Phase 1（Epoch 1-5）は classifier のみ学習 |
| Layer-wise LR | backbone=1e-5、head=1e-4 |
| Dropout | classifier head に Dropout(0.3) を追加 |
| LR Scheduler | ReduceLROnPlateau（patience=3, factor=0.5） |
| Augmentation | GaussianBlur・RandomGrayscale を追加 |
| 画像軽量化 | 224x224 JPEG に事前変換（8813 MB → 15.3 MB、99.8%削減） |
| num_workers | 0 に変更（MPS との semaphore 問題を回避） |

### バグ修正

- `compute_class_weight` に `numpy.ndarray` が必要なのにリストを渡していた
- MPS は float64 非対応 → `pos_weight` を `dtype=torch.float32` に変換
- `ImageFolder` がアルファベット順で `favorite=0, not_favorite=1` と割り当てるため、`BCEWithLogitsLoss` の `pos_weight` が not_favorite（多数派）を強調してしまっていた → `target_transform=lambda y: 1-y` で **favorite=1（正例）** に反転して修正
- `.jpeg` 拡張子の画像が `preprocess_images.py` で削除されず残存し、DataLoader でサイズ不一致エラー → 修正済み

### 学習結果（best val_loss=0.7946 / Epoch 25 で early stopping）

| Epoch | Phase | train_loss | val_loss | lr | patience |
|---|---|---|---|---|---|
| 1 | P1 | 1.0031 | **0.9512** ✓ | 1e-4 | 0/10 |
| 2 | P1 | 0.8920 | **0.9006** ✓ | 1e-4 | 0/10 |
| 3 | P1 | 0.8731 | **0.8790** ✓ | 1e-4 | 0/10 |
| 4 | P1 | 0.8665 | **0.8783** ✓ | 1e-4 | 0/10 |
| 5 | P1 | 0.8610 | **0.8545** ✓ | 1e-4 | 0/10 |
| 6 | P2 | 0.8407 | **0.8340** ✓ | 1e-4 | 0/10 |
| 7 | P2 | 0.7914 | **0.8269** ✓ | 1e-4 | 0/10 |
| 8 | P2 | 0.8016 | **0.8089** ✓ | 1e-4 | 0/10 |
| 9 | P2 | 0.7831 | 0.8093 | 1e-4 | 1/10 |
| 10 | P2 | 0.7725 | 0.8095 | 1e-4 | 2/10 |
| 11 | P2 | 0.7756 | 0.8102 | 1e-4 | 3/10 |
| 12 | P2 | 0.7510 | **0.8057** ✓ | 1e-4 | 0/10 |
| 13 | P2 | 0.7560 | **0.8020** ✓ | 1e-4 | 0/10 |
| 14 | P2 | 0.7378 | 0.8062 | 1e-4 | 1/10 |
| 15 | P2 | 0.7368 | **0.7946** ✓ | 1e-4 | 0/10 |
| 16〜25 | P2 | 〜0.70 | 0.80〜0.81 | 5e-5→2.5e-5 | 〜10/10 |
| 25 | P2 | 0.6960 | 0.8118 | 2.5e-5 | 10/10 → **Early stopping** |

- best val_loss: **0.7946**（前回比 -0.0355 改善）
- val_loss が Epoch 15 まで着実に改善 → 改善策が有効

### evaluate 結果（ラベルバグ修正前のモデルで実行）

| 指標 | 結果 | 目標 |
|---|---|---|
| Accuracy | 66.5% | 75%以上 |
| AUC-ROC | 0.6508 | 0.80以上 |
| Recall（favorite） | 1.2% | 70%以上 |

→ **ラベル反転バグ（pos_weight が not_favorite を強調していた）が原因で Recall が壊滅的だった。**
　修正後のモデル（実行 #3）で再評価済み（下記）。

---

## 実行 #3 — 2026-04-10（実行中）

### 修正内容

- `dataset.py` に `target_transform=lambda y: 1-y` を追加
  - favorite=1（正例）、not_favorite=0 に統一
  - `pos_weight=2.0` が正しく favorite の loss を強調するようになった
- `evaluate.py` の評価ロジックを `pos_label=1` に統一、`--sweep` オプションで閾値スイープ追加

### 学習結果（best val_loss=0.8449 / Epoch 23 で early stopping）

### evaluate 結果（閾値スイープ込み）

| threshold | accuracy | recall | 備考 |
|---|---|---|---|
| 0.30 | 45.6% | 97.6% | Recall OK |
| 0.40 | 46.8% | 92.8% | Recall OK |
| **0.50** | **57.7%** | **68.7%** | Recall NG |
| 0.60 | 65.3% | 24.1% | — |

- AUC-ROC: 0.6494
- **閾値調整だけでは Recall≥70% と Accuracy≥75% を両立できない**
- Recall を 70% 以上にするには閾値を 0.4 以下にする必要があり、Accuracy が 47% に落ちる

---

## 実行 #4 — 2026-04-10

### 変更点（実行 #3 からの差分）

| 項目 | 前回 | 今回 |
|---|---|---|
| freeze_epochs | 5 | 10 |
| early_stopping_patience | 10 | 15 |

### 学習結果（best val_loss=0.8522 / Epoch 16、Epoch 31 で early stopping）

val_loss が Epoch 16 で 0.8522 まで改善した後、15 epoch 改善なく終了。
前回（#3）の 0.8449 より悪化。freeze を長くすると backbone の fine-tune 開始が遅れ、
結果的に収束が浅くなった可能性がある。

### evaluate 結果（閾値スイープ込み）

| threshold | accuracy | recall | 備考 |
|---|---|---|---|
| 0.30 | 46.0% | 98.8% | Recall OK |
| 0.40 | 46.0% | 95.2% | Recall OK |
| **0.50** | **50.8%** | **80.7%** | **Recall OK** |
| 0.60 | 62.1% | 34.9% | — |

- AUC-ROC: 0.6073（前回比 -0.04）
- threshold=0.5 でも Recall 80.7% を達成したが、Accuracy は 50.8%

---

## 総括・根本課題

複数回の試行を通じて、**AUC が 0.60〜0.65 付近で頭打ち**になっている。
ハイパーパラメータの調整では改善しておらず、より根本的な対策が必要。

### 考えられる原因

1. **データの識別困難性** — 同じシーンで選ばなかった写真（not_fav の 70-80%）と favorite は視覚的に非常に似ており、EfficientNet-B0 では特徴が足りない
2. **データ量不足** — favorite が 549 枚（train 384 枚）は転移学習でも少ない
3. **ラベルノイズ** — お気に入りの基準が主観的で一貫していない可能性

### 次の改善候補（根本的な対策）

| 優先度 | 対策 | 期待効果 |
|---|---|---|
| ★★★ | EfficientNet-B3/B4 に変更 | より高い表現能力で AUC 改善 |
| ★★★ | not_fav サンプリング戦略の見直し（同シーン比率を下げる） | 識別困難なペアを減らす |
| ★★☆ | TTA（Test Time Augmentation） | 推論精度を若干改善 |
| ★★☆ | val + train を合わせた学習（最終モデル） | データ量を増やす |
| ★☆☆ | 誤分類画像の目視確認 | 問題の本質を理解する |

---

## 実行 #5 — 2026-04-10

### 変更点

| 項目 | 値 |
|---|---|
| モデル | EfficientNet-B3（B0 から変更） |
| image_size | 300×300（B3 推奨サイズ） |
| classifier 入力次元 | 1536（B0: 1280） |
| その他 | 実行 #4 の設定を継承（freeze=10, patience=15） |

### 学習結果（best val_loss=0.8663 / Epoch 26、Epoch 41 で early stopping）

### evaluate 結果（閾値スイープ込み）

| threshold | accuracy | recall | 備考 |
|---|---|---|---|
| 0.50 | 54.0% | 78.3% | Recall OK |
| 0.60 | 64.9% | 41.0% | — |

- AUC-ROC: **0.6362**

### 結論：モデルサイズはボトルネックではない

B3（パラメータ数 B0 比 約3倍）に変えても AUC が 0.63〜0.65 の範囲から出ない。
**問題はモデルの表現力ではなくデータにある。**

not_fav の 70〜80% が「同シーンで選ばなかった写真」であり、favorite と視覚的に
ほぼ区別がつかない画像が多数混在している。これがモデルの識別能力の上限を規定している。

### 最優先の次手：not_fav のサンプリング戦略を変える

same-scene 比率を 70-80% → 30-40% に下げ、ランダム写真を増やす。
これにより識別困難なペアが減り、モデルが「選ばれる写真の特徴」を学びやすくなる。

### 学習結果（best val_loss=0.8449 / Epoch 23 で early stopping）

val_loss は実行 #2 より高め。pos_weight が正しく favorite を強調するようになった分、
favorite の誤検出ペナルティが増えて loss が高止まりした可能性がある。

### evaluate 結果

| 指標 | 結果 | 目標 | 前回比 |
|---|---|---|---|
| Accuracy | 57.7% | 75%以上 | - |
| AUC-ROC | 0.6494 | 0.80以上 | ≒同等 |
| Recall（favorite） | **68.7%** | 70%以上 | **+67.5%** ← 大幅改善 |

混同行列：

|  | 予測:favorite | 予測:not_fav |
|---|---|---|
| 実際:favorite | 57 | 26 |
| 実際:not_fav | 79 | 86 |

### 考察

- **ラベル修正は有効**：Recall が 1.2% → 68.7% と劇的に改善
- **AUC が低い（0.65）**：モデル自体の識別能力がまだ弱い。val_loss が 0.84 止まりで、Accuracy も低いことからモデルが favorite の特徴を十分に捉えられていない
- **false positive が多い（79件）**：閾値 0.5 では favorite と判定しすぎている

### 次の改善候補

| 優先度 | 対策 | 理由 |
|---|---|---|
| ★★★ | 決定閾値を 0.5 → 0.6〜0.7 に調整 | false positive を減らして Accuracy/Precision を改善 |
| ★★☆ | early_stopping_patience を 15 に延ばす | val_loss の改善余地がまだある可能性 |
| ★★☆ | freeze_epochs を 10 に延ばす | head をより十分に学習してから backbone を解凍 |
| ★☆☆ | pos_weight を手動で大きくする（例: 3.0） | Recall 70% を確実に超えるため |

---

## 実行 #6 — 2026-05-03

### アーキテクチャ全面変更：LAION Aesthetics Predictor アプローチ

EfficientNet-B3 をやめ、LAION Aesthetics Predictor V2 と同一の構成に全面書き換え。

| 項目 | 変更内容 |
|---|---|
| Backbone | CLIP ViT-L-14（OpenAI pretrained、4.28億パラメータ、常に frozen） |
| Head | LAION Aesthetics V2 と同一 MLP（768→512→256→128→64→1） |
| 学習可能パラメータ | 約 566,273（head のみ） |
| 正規化統計 | ImageNet → CLIP（mean/std 変更） |
| image_size | 300 → 224 |
| batch_size | 16 → 32 |
| 依存追加 | open-clip-torch 3.3.0 |

### 実行フロー変更

CLIP ViT-L-14 の MPS 推論が 1 epoch あたり約 20 分と極端に遅かったため、
特徴量を一度だけ事前抽出する 2 フェーズ方式に変更。

```
uv run python src/extract_features.py  # CLIP で全画像を一度だけ推論・保存
uv run python src/train.py             # 事前抽出済みベクトルで MLP のみ学習（数秒/epoch）
```

| ファイル | 内容 |
|---|---|
| `data/features/train.npz` | 1154件 × 768次元 |
| `data/features/val.npz` | 247件 × 768次元 |
| `data/features/test.npz` | 248件 × 768次元 |

### 学習結果（best val_loss=0.8813 / Epoch 8、Epoch 23 で early stopping）

| Epoch | train_loss | val_loss | lr | patience |
|---|---|---|---|---|
| 1 | 0.9264 | **0.9253** ✓ | 1e-4 | 0/15 |
| 2 | 0.9256 | **0.9246** ✓ | 1e-4 | 0/15 |
| 3 | 0.9243 | **0.9233** ✓ | 1e-4 | 0/15 |
| 4 | 0.9223 | **0.9209** ✓ | 1e-4 | 0/15 |
| 5 | 0.9167 | **0.9147** ✓ | 1e-4 | 0/15 |
| 6 | 0.9037 | **0.9028** ✓ | 1e-4 | 0/15 |
| 7 | 0.8768 | **0.8939** ✓ | 1e-4 | 0/15 |
| 8 | 0.8464 | **0.8813** ✓ | 1e-4 | 0/15 |
| 9〜23 | 0.72〜0.84 | 0.88〜0.90 | 1e-4→1.25e-5 | 〜15/15 |
| 23 | 0.7185 | 0.9007 | 1.25e-5 | 15/15 → **Early stopping** |

- best val_loss: **0.8813**（Epoch 8）

### evaluate 結果（閾値スイープ込み）

| threshold | accuracy | precision | recall | f1 | 備考 |
|---|---|---|---|---|---|
| 0.30 | 45.6% | 37.9% | 97.6% | 54.6% | Recall OK |
| 0.40 | 48.0% | 38.4% | 91.6% | 54.1% | Recall OK |
| **0.50** | **56.9%** | **42.6%** | **83.1%** | **56.3%** | **Recall OK ← 最良** |
| 0.60 | 64.9% | 47.4% | 43.4% | 45.3% | — |
| 0.70 | 66.5% | 0.0% | 0.0% | 0.0% | — |

- AUC-ROC: **0.6773**（過去最高）
- threshold=0.5 で Recall 83.1%（目標 70% 達成）

混同行列（threshold=0.50）：

|  | 予測:favorite | 予測:not_fav |
|---|---|---|
| 実際:favorite | 69 | 14 |
| 実際:not_fav | 93 | 72 |

### 目標値チェック

| 指標 | 結果 | 目標 | 判定 |
|---|---|---|---|
| Accuracy | 56.9% | 75%以上 | NG |
| AUC-ROC | **0.6773** | 0.80以上 | NG（過去最高） |
| Recall（favorite） | **83.1%** | 70%以上 | **OK** |

### 考察

- **AUC が 0.68 に改善**：EfficientNet-B3 の 0.63 から向上。CLIP の汎用視覚特徴は有効。
- **Accuracy が低い（57%）**：false positive が多い（93件）。モデルが大半の画像を「favorite」と判定する傾向。
- **pos_weight=2.0 による bias**：favorite を positive にするクラス重みが強すぎる可能性。
- **Data Augmentation なし**：特徴量を事前抽出するため、学習時の augmentation が効かない。固定 1154 件のベクトルだけで学習しており多様性が乏しい。

### 根本課題：変わらず

AUC が 0.65〜0.68 で頭打ち。CLIP の汎用特徴でも「同シーンで選ばなかった写真」と「選んだ写真」の区別は困難。

### 次の改善候補

| 優先度 | 対策 | 期待効果 |
|---|---|---|
| ★★★ | pos_weight を下げる（例: 1.5）か threshold を上げる（0.6） | false positive を減らして Accuracy 改善 |
| ★★☆ | not_fav の same-scene 比率を下げる | 識別困難なペアを減らす |
| ★★☆ | Augmentation 付きで特徴量を複数回抽出して訓練データを増やす | 過学習抑制 |
| ★☆☆ | CLIP ViT-B/16 に変更して速度と品質のバランスを確認 | 実験速度向上 |

---

## 実行 #7 — 2026-05-03

### アーキテクチャ全面変更：NIMA（Neural Image Assessment）

| 項目 | 変更内容 |
|---|---|
| Backbone | NIMA InceptionResNetV2（AVA 255k 枚で事前学習済み、5430万パラメータ、常に frozen） |
| Head | 1536→512→256→128→64→1 の MLP（学習可能: 約 95.9万パラメータ） |
| 正規化統計 | mean=0.5, std=0.5（NIMA 固有） |
| embed_dim | 768（CLIP）→ 1536（NIMA InceptionResNetV2） |
| 依存 | open-clip-torch を削除、pyiqa 0.1.15 を追加 |

### 実行フロー（CLIP 方式を継承）

```
uv run python src/extract_features.py  # NIMA backbone で全画像を推論・保存（1154件×1536次元）
uv run python src/train.py             # AestheticsHead のみ学習（数秒/epoch）
```

### 学習結果（best val_loss=0.8845 / Epoch 10、Epoch 25 で early stopping）

val_loss が不安定（0.88〜1.09 の幅で変動）で、CLIP 時（0.88〜0.90）より荒れた。

| Epoch | train_loss | val_loss | lr | patience |
|---|---|---|---|---|
| 1〜6 | 0.93→0.90 | **0.92→0.90** ✓ | 1e-4 | 0/15 |
| 7〜8 | 0.88→0.86 | 0.93→0.95 | 1e-4 | 1〜2/15 |
| 9 | 0.85 | **0.8872** ✓ | 1e-4 | 0/15 |
| 10 | 0.85 | **0.8845** ✓ | 1e-4 | 0/15 |
| 11〜25 | 0.78→0.76 | 0.93〜1.01 | 1e-4→1.25e-5 | 〜15/15 |
| 25 | 0.7644 | 1.0050 | 1.25e-5 | 15/15 → **Early stopping** |

- best val_loss: **0.8845**（Epoch 10）

### evaluate 結果（閾値スイープ込み）

| threshold | accuracy | precision | recall | f1 | 備考 |
|---|---|---|---|---|---|
| 0.30 | 45.2% | 37.7% | 97.6% | 54.4% | Recall OK |
| 0.40 | 45.6% | 37.9% | 97.6% | 54.6% | Recall OK |
| **0.50** | **46.0%** | **37.8%** | **95.2%** | **54.1%** | **Recall OK ← 最良** |
| 0.60 | 53.6% | 37.9% | 60.2% | 46.5% | — |
| 0.70 | 66.5% | 0.0% | 0.0% | 0.0% | — |

- AUC-ROC: **0.5915**

混同行列（threshold=0.50）：

|  | 予測:favorite | 予測:not_fav |
|---|---|---|
| 実際:favorite | 79 | 4 |
| 実際:not_fav | 130 | 35 |

### 目標値チェック

| 指標 | 結果 | 目標 | 判定 |
|---|---|---|---|
| Accuracy | 46.0% | 75%以上 | NG |
| AUC-ROC | 0.5915 | 0.80以上 | NG |
| Recall（favorite） | 95.2% | 70%以上 | OK |

### モデル比較（全実行通算）

| 実行 | backbone | AUC-ROC | Recall@0.5 | Accuracy@0.5 |
|---|---|---|---|---|
| #1〜#5 | EfficientNet-B0/B3 | 0.61〜0.65 | 1〜81% | 46〜66% |
| **#6** | **CLIP ViT-L-14** | **0.6773** | **83.1%** | **56.9%** |
| #7 | NIMA InceptionResNetV2 | 0.5915 | 95.2% | 46.0% |

### 考察：NIMA が CLIP より劣った理由

- **AVA vs 個人的嗜好**：NIMA は AVA（大衆的な美的品質評価）で学習されており、「自分が選んだ写真」という個人的な好みとは異なる次元を評価している。AVA 的に高品質でも「お気に入り」でない写真は多く存在する。
- **val_loss の不安定性**：NIMA 特徴量が本タスクに対して汎化しにくく、head が過学習しやすい。
- **モデルが "全部 favorite" に倒れる**：AUC 0.59 はランダム（0.50）に近く、NIMA スコアがお気に入り予測に有効な信号をほとんど持っていない。

### 結論

個人的写真お気に入り予測において **CLIP > EfficientNet > NIMA** の順で有効。
NIMA はバックボーンとして不適切であるため、次回は CLIP ViT-L-14 に戻して
データ戦略（same-scene 比率削減・augmentation 多様化）を改善する方向で進める。

---

## 実行 #8 — 2026-05-03

### アーキテクチャ：CLIP + NIMA Aesthetic + NIMA Technical 3バックボーン融合

#6（CLIP 単体, 0.6773）と #7（NIMA 単体, 0.5915）の両バックボーンを組み合わせ、
それぞれが持つ異なる表現空間の情報を最大限に活用する。

| 項目 | 変更内容 |
|---|---|
| Backbone 1 | CLIP ViT-L-14（768-dim, L2 正規化） |
| Backbone 2 | NIMA Aesthetic — InceptionResNetV2, AVA 学習済み（1536-dim, L2 正規化） |
| Backbone 3 | NIMA Technical — InceptionResNetV2, KonIQ 学習済み（1536-dim, L2 正規化） |
| 連結ベクトル | 768 + 1536 + 1536 = **3840-dim** |
| Head 構成 | BN(3840) → Linear(3840→128) → ReLU → Dropout(0.5) → Linear(128→1) |
| 学習可能パラメータ | 約 491,905（head のみ） |
| optimizer | Adam（lr=1e-4, weight_decay=1e-3） |
| 全バックボーン | 常に frozen / eval |

#### Head 設計の経緯

最初に深い MLP（3840→512→256→128→64→1）を試したところ、約 210 万パラメータが
1154 サンプルに対して多すぎ、Epoch 3 以降で即座に過学習した（train_loss=0.15, val_loss=1.53）。
AUC は 0.6876 と#6 を上回ったものの不安定な学習であったため、
**BN + 1 hidden layer(128) + Dropout(0.5)** の軽量構成に再設計。

### 学習結果（best val_loss=0.8536 / Epoch 2、Epoch 17 で early stopping）

| Epoch | train_loss | val_loss | lr | patience |
|---|---|---|---|---|
| 1 | 0.8591 | **0.9052** ✓ | 1e-4 | 0/15 |
| 2 | 0.7591 | **0.8536** ✓ | 1e-4 | 0/15 |
| 3〜17 | 0.73→0.49 | 0.86〜0.92 | 1e-4→1.25e-5 | 〜15/15 |
| 17 | 0.5078 | 0.9205 | 1.25e-5 | 15/15 → **Early stopping** |

- best val_loss: **0.8536**（Epoch 2）

### evaluate 結果（閾値スイープ込み）

| threshold | accuracy | precision | recall | f1 | 備考 |
|---|---|---|---|---|---|
| 0.30 | 45.6% | 37.9% | 97.6% | 54.6% | Recall OK |
| 0.40 | 47.6% | 38.7% | 96.4% | 55.2% | Recall OK |
| **0.50** | **59.7%** | **43.7%** | **71.1%** | **54.1%** | **Recall OK ← 最良** |
| 0.60 | 68.5% | 60.0% | 18.1% | 27.8% | — |
| 0.70 | 66.5% | 0.0% | 0.0% | 0.0% | — |

- AUC-ROC: **0.7218**（全実行通算の新記録）
- threshold=0.5 で Recall 71.1%（目標 70% 達成）

混同行列（threshold=0.50）：

|  | 予測:favorite | 予測:not_fav |
|---|---|---|
| 実際:favorite | 59 | 24 |
| 実際:not_fav | 76 | 89 |

### 目標値チェック

| 指標 | 結果 | 目標 | 判定 | 前回比（#6） |
|---|---|---|---|---|
| Accuracy | 59.7% | 75%以上 | NG | +2.8% |
| AUC-ROC | **0.7218** | 0.80以上 | NG（**全実行最高**） | **+0.0445** |
| Recall（favorite） | **71.1%** | 70%以上 | **OK** | ≒同等 |

### モデル比較（全実行通算）

| 実行 | backbone | AUC-ROC | Recall@0.5 | Accuracy@0.5 |
|---|---|---|---|---|
| #1〜#5 | EfficientNet-B0/B3 | 0.61〜0.65 | 1〜81% | 46〜66% |
| #6 | CLIP ViT-L-14 | 0.6773 | 83.1% | 56.9% |
| #7 | NIMA Aesthetic only | 0.5915 | 95.2% | 46.0% |
| **#8** | **CLIP + NIMA Aes + NIMA Tech** | **0.7218** | **71.1%** | **59.7%** |

### 考察

- **3 バックボーン融合が有効**：CLIP 単体（0.68）と NIMA 単体（0.59）の単純な性能より、
  融合後（0.72）の方が高い AUC を達成。各バックボーンが捉える特徴が相補的に機能した。
- **NIMA Technical（KonIQ）の寄与**：技術的画質の情報が、美的評価（CLIP・NIMA Aesthetic）だけでは
  拾えない「撮影品質の良い写真を好む」傾向を補足している可能性。
- **Accuracy はまだ低い（59.7%）**：false positive（76件）が多く、モデルが
  positive（favorite）に偏りやすい。pos_weight=2.0 による bias が影響。
- **val_loss が Epoch 2 で頭打ち**：3840次元の特徴量でもサンプル数（1154件）に対しては
  過学習が早い。より多くのデータがあれば、このアーキテクチャの真価が発揮できる可能性。

### 次の改善候補

| 優先度 | 対策 | 期待効果 |
|---|---|---|
| ★★★ | not_fav の same-scene 比率を下げる | 識別困難なペアを減らし AUC を根本改善 |
| ★★☆ | pos_weight を 1.5 に下げる | false positive を抑えて Accuracy 改善 |
| ★★☆ | threshold を 0.55〜0.60 に調整 | Recall 70% 維持しつつ Accuracy 改善 |
| ★☆☆ | NIMA SPAQ を 4 つ目のバックボーンとして追加 | さらなる AUC 改善（over-fitting に注意） |

---

## 実行 #9 — 2026-05-03

### 変更内容：NIMA 配点調整 + Head アーキテクチャ見直し

#### 問題の特定

実行 #8 の Head は `Linear(3840→128)` で 3840-dim を一括処理しており、
次元の比率で NIMA（1536×2=3072）が CLIP（768）の **4倍の発言権** を持っていた。

| バックボーン | 旧配点 | 新配点 |
|---|---|---|
| CLIP（AUC 0.68, 有効） | 768/3840 = **20%** | 256/512 = **50%** |
| NIMA Aes（AUC 0.59, ほぼランダム） | 1536/3840 = **40%** | 128/512 = **25%** |
| NIMA Tech | 1536/3840 = **40%** | 128/512 = **25%** |

#### 新 Head アーキテクチャ

```
CLIP(768)     → Linear(768→256) → ReLU ─┐
NIMA Aes(1536) → Linear(1536→128) → ReLU ─┤→ concat(512) → BN → Dropout(0.5)
NIMA Tech(1536) → Linear(1536→128) → ReLU ─┘              → Linear(512→64) → ReLU
                                                            → Dropout(0.5)
                                                            → Linear(64→1)
```

- 特徴量の再抽出なし（既存 3840-dim npz をそのまま使用）
- Head 内でスライスして各バックボーンを独立投影

### 学習結果（best val_loss=0.8449 / Epoch 5、Epoch 20 で early stopping）

| Epoch | train_loss | val_loss | lr | patience |
|---|---|---|---|---|
| 1 | 0.8982 | **0.9190** ✓ | 1e-4 | 0/15 |
| 2 | 0.8379 | **0.8785** ✓ | 1e-4 | 0/15 |
| 3 | 0.8156 | **0.8574** ✓ | 1e-4 | 0/15 |
| 4 | 0.8069 | **0.8496** ✓ | 1e-4 | 0/15 |
| 5 | 0.7819 | **0.8449** ✓ | 1e-4 | 0/15 |
| 6〜20 | 0.77→0.65 | 0.84〜0.87 | 1e-4→1.25e-5 | 〜15/15 |
| 20 | 0.6489 | 0.8724 | 1.25e-5 | 15/15 → **Early stopping** |

- best val_loss: **0.8449**（実行 #8 の 0.8536 から改善）
- 収束が Epoch 2 → Epoch 5 に遅延（過学習が若干抑制された）

### evaluate 結果（閾値スイープ込み）

| threshold | accuracy | precision | recall | f1 | 備考 |
|---|---|---|---|---|---|
| 0.30 | 45.6% | 37.9% | 97.6% | 54.6% | Recall OK |
| 0.40 | 46.4% | 38.0% | 95.2% | 54.3% | Recall OK |
| **0.50** | **58.5%** | **44.0%** | **88.0%** | **58.6%** | **Recall OK ← 最良** |
| 0.60 | 68.5% | 53.3% | 49.4% | 51.2% | — |
| 0.70 | 69.8% | 83.3% | 12.1% | 21.1% | — |

- AUC-ROC: **0.7233**（実行 #8 の 0.7218 から微改善）
- threshold=0.5 で Recall **88.0%**（実行 #8 の 71.1% から大幅改善）

混同行列（threshold=0.50）：

|  | 予測:favorite | 予測:not_fav |
|---|---|---|
| 実際:favorite | 73 | 10 |
| 実際:not_fav | 93 | 72 |

### 目標値チェック

| 指標 | 結果 | 目標 | 判定 | 前回比（#8） |
|---|---|---|---|---|
| Accuracy | 58.5% | 75%以上 | NG | ≒同等 |
| AUC-ROC | **0.7233** | 0.80以上 | NG（全実行最高更新） | +0.0015 |
| Recall（favorite） | **88.0%** | 70%以上 | **OK** | **+16.9%** |

### モデル比較（全実行通算）

| 実行 | backbone / Head | AUC-ROC | Recall@0.5 | Accuracy@0.5 |
|---|---|---|---|---|
| #1〜#5 | EfficientNet-B0/B3 | 0.61〜0.65 | 1〜81% | 46〜66% |
| #6 | CLIP ViT-L-14（単体） | 0.6773 | 83.1% | 56.9% |
| #7 | NIMA Aesthetic（単体） | 0.5915 | 95.2% | 46.0% |
| #8 | CLIP+NIMA×2、Linear(3840→128) | 0.7218 | 71.1% | 59.7% |
| **#9** | **CLIP+NIMA×2、独立投影 (CLIP=50%)** | **0.7233** | **88.0%** | **58.5%** |

### 考察

- **NIMA 配点調整は有効**：Recall が 71.1% → 88.0% と大幅改善。AUC も微改善。
- **Accuracy の変化は小さい**：false positive（93件）が依然として多い。根本的にはデータ問題。
- **val_loss の収束が遅延**：Epoch 2 → Epoch 5 に改善。正則化が若干強化されていることを示す。
- **AUC の頭打ちが継続**：0.72 から大きく改善しない。「同シーンで選ばなかった写真」という識別困難なサンプルが上限を規定。

### 次の改善候補

| 優先度 | 対策 | 期待効果 |
|---|---|---|
| ★★★ | not_fav の same-scene 比率を下げる（70-80% → 30-40%） | AUC の根本改善（最も重要） |
| ★★☆ | threshold を 0.60 に調整 | Recall 約 50%、Accuracy 約 69% でバランス改善 |
| ★★☆ | pos_weight を 1.5 に下げる | false positive を抑えて Accuracy 改善 |
