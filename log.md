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
