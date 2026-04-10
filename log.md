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
　修正後のモデル（実行 #3）で再評価予定。

---

## 実行 #3 — 2026-04-10（実行中）

### 修正内容

- `dataset.py` に `target_transform=lambda y: 1-y` を追加
  - favorite=1（正例）、not_favorite=0 に統一
  - `pos_weight=2.0` が正しく favorite の loss を強調するようになった
- `evaluate.py` の評価ロジックを `pos_label=1` に統一
