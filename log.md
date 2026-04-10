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
