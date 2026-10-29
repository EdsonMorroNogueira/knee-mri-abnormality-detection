"""Train/validation loop: 20 epochs, BCEWithLogitsLoss with pos_weight, mean AUC on the holdout.

pos_weight compensates for rare positives. Mean AUC over the 12 classes is
computed each epoch and the best epoch is tracked. Classes whose holdout labels
are all one value are skipped, since roc_auc_score is undefined for them.
Introduced in Stage 5.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

N_EPOCHS = 20
ENCODER_LR = 1e-5
HEAD_LR = 1e-3

train_study_ids = train_df["StudyInstanceUID"].tolist()
val_study_ids = val_df["StudyInstanceUID"].tolist()
train_labels = torch.tensor(train_df[TARGET_COLUMNS].values, dtype=torch.float32)
val_labels = torch.tensor(val_df[TARGET_COLUMNS].values, dtype=torch.float32)

print(f"train: {len(train_study_ids)} studies, validation: {len(val_study_ids)} studies")

train_cache, train_mask = build_pixel_cache(train_study_ids)
val_cache, val_mask = build_pixel_cache(val_study_ids)

model = build_finetune_model()
pool = MultiQueryAttentionPool(dim=model.config.hidden_size, n_slot=6, n_out=12, hidden=64)

optimizer = torch.optim.AdamW(
    [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": ENCODER_LR},
        {"params": pool.parameters(), "lr": HEAD_LR},
    ]
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_auc = -1.0
best_epoch = -1

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    logits, _ = forward_batch(model, pool, train_cache, train_mask)
    loss = criterion(logits, train_labels)
    loss.backward()
    optimizer.step()

    # Evaluate on the holdout without gradient, then resume training mode.
    model.eval()
    pool.eval()
    with torch.no_grad():
        val_logits, _ = forward_batch(model, pool, val_cache, val_mask)
    model.train()
    pool.train()
    val_probs = torch.sigmoid(val_logits).numpy()

    aucs = []
    for i in range(12):  # skip classes with a single label value in the holdout
        try:
            auc = roc_auc_score(val_labels[:, i].numpy(), val_probs[:, i])
            aucs.append(auc)
        except ValueError:
            pass

    mean_auc = sum(aucs) / len(aucs)
    print(f"epoch {epoch + 1}/{N_EPOCHS}: loss={loss.item():.4f} val_auc={mean_auc:.4f}")
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_epoch = epoch + 1

print(f"\nBest epoch: {best_epoch}/{N_EPOCHS}, mean AUC = {best_auc:.4f}")
