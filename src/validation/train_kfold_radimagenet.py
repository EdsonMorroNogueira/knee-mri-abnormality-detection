"""
Train the RadImageNet model over 5 grouped folds and score out-of-fold (OOF) predictions.

Each fold validates on a different slice of the gold studies. Concatenating the
per-fold predictions gives every study one prediction from a model that never
saw it, so the macro AUC is computed over all 58 studies instead of a single
12-study holdout. That makes a far more stable CV estimate (trust your CV, not
one tiny holdout).

Expects these pieces already defined in the session: MultiQueryAttentionPool,
build_pixel_cache, gold_df, TARGET_COLUMNS, build_finetune_radimagenet_backbone,
group_kfold_split, IMAGENET_MEAN/IMAGENET_STD.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

N_EPOCHS = 40
ENCODER_LR = 1e-5
HEAD_LR = 1e-3
SEED = 0


def forward_batch_timm(model, pool, cache_uint8: np.ndarray, mask: np.ndarray):
    """Embed all present slices with the backbone, then pool per study."""
    n_studies, n_slot = mask.shape
    dim = model.num_features
    present = mask.reshape(-1) > 0.5

    flat_pixels = cache_uint8.reshape(n_studies * n_slot, 3, 224, 224)[present]
    batch = (flat_pixels.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    batch_tensor = torch.tensor(batch, dtype=torch.float32)

    pooled = model(batch_tensor)

    embeddings_flat = torch.zeros(n_studies * n_slot, dim)
    embeddings_flat[present] = pooled
    embeddings = embeddings_flat.reshape(n_studies, n_slot, dim)

    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    return pool(embeddings, mask_tensor)


def compute_pos_weight(train_df):
    """Per-class (1 - rate) / rate, recomputed per fold since each fold has a different train_df."""
    weights = []
    for col in TARGET_COLUMNS:
        rate = (train_df[col] > 0).mean()
        weight = 1.0 if rate == 0 else (1 - rate) / rate
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


folds = group_kfold_split(gold_df)
print(f"{len(folds)} folds over {len(gold_df)} gold studies")

oof_study_ids = []
oof_probs = []
oof_labels = []
fold_best_aucs = []

for fold_idx, (train_df, val_df) in enumerate(folds):
    print(f"\n--- fold {fold_idx} ---")
    train_study_ids = train_df["StudyInstanceUID"].tolist()
    val_study_ids = val_df["StudyInstanceUID"].tolist()
    train_labels = torch.tensor(train_df[TARGET_COLUMNS].values, dtype=torch.float32)
    val_labels = torch.tensor(val_df[TARGET_COLUMNS].values, dtype=torch.float32)
    pos_weight = compute_pos_weight(train_df)

    train_cache, train_mask = build_pixel_cache(train_study_ids)
    val_cache, val_mask = build_pixel_cache(val_study_ids)

    torch.manual_seed(SEED)
    model = build_finetune_radimagenet_backbone()
    pool = MultiQueryAttentionPool(dim=model.num_features, n_slot=6, n_out=12, hidden=64)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": ENCODER_LR},
            {"params": pool.parameters(), "lr": HEAD_LR},
        ]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_val_probs = None

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        logits, _ = forward_batch_timm(model, pool, train_cache, train_mask)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        pool.eval()
        with torch.no_grad():
            val_logits, _ = forward_batch_timm(model, pool, val_cache, val_mask)
        model.train()
        pool.train()
        val_probs = torch.sigmoid(val_logits).numpy()

        aucs = []
        for i in range(12):
            try:
                auc = roc_auc_score(val_labels[:, i].numpy(), val_probs[:, i])
            except ValueError:
                continue
            # In ~12-study folds a rare class can have all-identical labels; this
            # sklearn version then returns NaN (with a warning) instead of raising.
            # NaN compares False, which would leave best_auc stuck at -1.0.
            if not np.isnan(auc):
                aucs.append(auc)
        epoch_mean_auc = sum(aucs) / len(aucs) if aucs else -1.0

        if epoch_mean_auc > best_auc:
            best_auc = epoch_mean_auc
            best_val_probs = val_probs.copy()

    if best_val_probs is None:
        # No epoch had a valid class (very unlikely); fall back to the last epoch.
        best_val_probs = val_probs.copy()

    print(f"fold {fold_idx}: best mean AUC (on the fold's own validation) = {best_auc:.4f}")
    fold_best_aucs.append(best_auc)

    oof_study_ids.extend(val_study_ids)
    oof_probs.append(best_val_probs)
    oof_labels.append(val_labels.numpy())

oof_probs = np.concatenate(oof_probs, axis=0)
oof_labels = np.concatenate(oof_labels, axis=0)

oof_aucs = []
for i in range(12):
    try:
        auc = roc_auc_score(oof_labels[:, i], oof_probs[:, i])
    except ValueError:
        continue
    if not np.isnan(auc):
        oof_aucs.append(auc)
oof_mean_auc = sum(oof_aucs) / len(oof_aucs) if oof_aucs else float("nan")

print(f"\nMean AUC per fold (each on its own holdout): {[f'{a:.4f}' for a in fold_best_aucs]}")
print(f"OUT-OF-FOLD macro AUC (all {len(oof_study_ids)} gold studies, each predicted once): {oof_mean_auc:.4f}")
print("Reference: single-split baseline (12 studies) = 0.7907 on holdout, 0.741 on the leaderboard.")
