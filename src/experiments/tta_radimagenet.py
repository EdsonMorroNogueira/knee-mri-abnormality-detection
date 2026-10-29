"""
Evaluate horizontal-flip test-time augmentation (TTA) on the RadImageNet model.

At prediction time the study is run as-is and mirrored left-right, and the two
probability sets are averaged. No retraining: only inference changes. Each epoch
is scored with and without TTA on the same checkpoint so the two are directly
comparable. In this project flip-TTA hurt (holdout AUC 0.7907 to 0.7698): the
data pipeline already normalizes laterality, so mirroring again swaps medial and
lateral anatomy.

Expects these pieces already defined in the session: MultiQueryAttentionPool,
build_pixel_cache, TARGET_COLUMNS, train_df/val_df, pos_weight,
build_finetune_radimagenet_backbone, IMAGENET_MEAN/IMAGENET_STD.
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


def predict_probs_tta(model, pool, cache_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean probability over the original and the horizontally flipped images.

    The cache layout is (n_studies, n_slot, 3, 224, 224), so axis -1 is width.
    """
    logits_normal, _ = forward_batch_timm(model, pool, cache_uint8, mask)
    probs_normal = torch.sigmoid(logits_normal).numpy()

    flipped_cache = np.flip(cache_uint8, axis=-1).copy()
    logits_flipped, _ = forward_batch_timm(model, pool, flipped_cache, mask)
    probs_flipped = torch.sigmoid(logits_flipped).numpy()

    return (probs_normal + probs_flipped) / 2


train_study_ids = train_df["StudyInstanceUID"].tolist()
val_study_ids = val_df["StudyInstanceUID"].tolist()
train_labels = torch.tensor(train_df[TARGET_COLUMNS].values, dtype=torch.float32)
val_labels = torch.tensor(val_df[TARGET_COLUMNS].values, dtype=torch.float32)

print(f"train: {len(train_study_ids)} studies, validation: {len(val_study_ids)} studies")

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

best_auc_no_tta = -1.0
best_auc_tta = -1.0

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
        val_probs_no_tta = torch.sigmoid(val_logits).numpy()
        val_probs_tta = predict_probs_tta(model, pool, val_cache, val_mask)
    model.train()
    pool.train()

    def epoch_auc(probs):
        aucs = []
        for i in range(12):
            try:
                aucs.append(roc_auc_score(val_labels[:, i].numpy(), probs[:, i]))
            except ValueError:
                pass
        return sum(aucs) / len(aucs)

    auc_no_tta = epoch_auc(val_probs_no_tta)
    auc_tta = epoch_auc(val_probs_tta)

    best_auc_no_tta = max(best_auc_no_tta, auc_no_tta)
    best_auc_tta = max(best_auc_tta, auc_tta)

print(f"\nBest AUC WITHOUT TTA: {best_auc_no_tta:.4f}")
print(f"Best AUC WITH TTA (horizontal flip): {best_auc_tta:.4f}")
print("Known reference: seed 0 without TTA = 0.7907 on holdout, 0.741 on the leaderboard.")
