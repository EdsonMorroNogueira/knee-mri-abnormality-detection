"""
Train a RadImageNet student on gold labels plus teacher soft labels (distillation).

Training mixes the 58 gold studies (hard 0/1 labels, weight 1.0) with unlabeled
studies whose targets are soft labels (continuous probabilities, weight
SOFT_LABEL_WEIGHT < 1.0) from a strong teacher image ensemble (~0.932 on the
leaderboard alone). Unlike LLM-derived weak labels from report text, the
teacher always returns all 12 probabilities, so no per-class NaN mask is needed.

Outcome (Stage 19): a pilot with 300 soft-labeled studies gave +0.02 on the
holdout; the full-scale run with 4349 studies gave ~0.00, which is within the
noise of a 12-study holdout. The distillation gain was not confirmed.

Scale notes: the forward pass is chunked into blocks of studies with gradient
accumulation (one optimizer step per epoch), mathematically identical to one
full-batch forward but small enough to fit in GPU memory at ~4.4k studies.
Model and tensors are moved to the GPU (earlier versions silently ran on CPU,
which was fine for tens of studies but far too slow at scale).

Expects these pieces already defined in the session: MultiQueryAttentionPool,
build_pixel_cache, TARGET_COLUMNS, train_df/val_df, pos_weight,
build_finetune_radimagenet_backbone, IMAGENET_MEAN/IMAGENET_STD, and the teacher
soft-label CSV (raptor_soft_labels.csv).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}")

N_EPOCHS = 40
ENCODER_LR = 1e-5
HEAD_LR = 1e-3
SEED = 0
SOFT_LABEL_WEIGHT = 0.5  # below gold (1.0): the teacher is strong but not ground truth
TRAIN_BATCH_STUDIES = 64  # studies per forward chunk in training
VAL_BATCH_STUDIES = 64  # same in validation (no gradients, but activations still use memory)


def forward_batch_timm(model, pool, cache_uint8: np.ndarray, mask: np.ndarray):
    """Embed all present slices with the backbone, then pool per study."""
    n_studies, n_slot = mask.shape
    dim = model.num_features
    present = mask.reshape(-1) > 0.5

    flat_pixels = cache_uint8.reshape(n_studies * n_slot, 3, 224, 224)[present]
    batch = (flat_pixels.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=dev)

    pooled = model(batch_tensor)

    embeddings_flat = torch.zeros(n_studies * n_slot, dim, device=dev)
    embeddings_flat[present] = pooled
    embeddings = embeddings_flat.reshape(n_studies, n_slot, dim)

    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=dev)
    return pool(embeddings, mask_tensor)


soft_labels_df = pd.read_csv("/kaggle/working/raptor_soft_labels.csv")
print(f"Teacher soft labels available: {len(soft_labels_df)} studies")

gold_part = train_df[["StudyInstanceUID"] + TARGET_COLUMNS].copy()
gold_part["sample_weight"] = 1.0

soft_part = soft_labels_df[["StudyInstanceUID"] + TARGET_COLUMNS].copy()
soft_part["sample_weight"] = SOFT_LABEL_WEIGHT

combined_train_df = pd.concat([gold_part, soft_part], ignore_index=True)
print(f"\nCombined train set: {len(combined_train_df)} studies "
      f"({(combined_train_df['sample_weight'] == 1.0).sum()} gold + "
      f"{(combined_train_df['sample_weight'] < 1.0).sum()} teacher soft labels)")

train_study_ids = combined_train_df["StudyInstanceUID"].tolist()
val_study_ids = val_df["StudyInstanceUID"].tolist()
train_labels = torch.tensor(combined_train_df[TARGET_COLUMNS].values, dtype=torch.float32, device=dev)
sample_weights = torch.tensor(combined_train_df["sample_weight"].values, dtype=torch.float32, device=dev)
val_labels = torch.tensor(val_df[TARGET_COLUMNS].values, dtype=torch.float32)

train_cache, train_mask = build_pixel_cache(train_study_ids)
val_cache, val_mask = build_pixel_cache(val_study_ids)

torch.manual_seed(SEED)
model = build_finetune_radimagenet_backbone().to(dev)
pool = MultiQueryAttentionPool(dim=model.num_features, n_slot=6, n_out=12, hidden=64).to(dev)

optimizer = torch.optim.AdamW(
    [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": ENCODER_LR},
        {"params": pool.parameters(), "lr": HEAD_LR},
    ]
)
# reduction="none" so each element can be scaled by sample_weight before averaging.
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev), reduction="none")

best_auc = -1.0
best_epoch = -1

n_train = train_cache.shape[0]
n_val = val_cache.shape[0]
# Fixed denominator (not the chunk size) so the summed chunk losses equal the
# mean over the whole training set.
total_elements = train_labels.numel()

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    epoch_loss_sum = 0.0
    for start in range(0, n_train, TRAIN_BATCH_STUDIES):
        end = min(start + TRAIN_BATCH_STUDIES, n_train)
        logits, _ = forward_batch_timm(model, pool, train_cache[start:end], train_mask[start:end])
        per_element_loss = criterion(logits, train_labels[start:end])
        batch_loss = (per_element_loss * sample_weights[start:end].unsqueeze(1)).sum() / total_elements
        batch_loss.backward()
        epoch_loss_sum += batch_loss.item()
    optimizer.step()

    model.eval()
    pool.eval()
    val_logits_chunks = []
    with torch.no_grad():
        for start in range(0, n_val, VAL_BATCH_STUDIES):
            end = min(start + VAL_BATCH_STUDIES, n_val)
            logits, _ = forward_batch_timm(model, pool, val_cache[start:end], val_mask[start:end])
            val_logits_chunks.append(logits)
    val_logits = torch.cat(val_logits_chunks, dim=0)
    model.train()
    pool.train()
    val_probs = torch.sigmoid(val_logits).cpu().numpy()

    aucs = []
    for i in range(12):
        try:
            auc = roc_auc_score(val_labels[:, i].numpy(), val_probs[:, i])
        except ValueError:
            continue
        if not np.isnan(auc):
            aucs.append(auc)
    epoch_mean_auc = sum(aucs) / len(aucs) if aucs else -1.0
    print(f"epoch {epoch + 1}/{N_EPOCHS}: loss={epoch_loss_sum:.4f} val_auc={epoch_mean_auc:.4f}")

    if epoch_mean_auc > best_auc:
        best_auc = epoch_mean_auc
        best_epoch = epoch + 1

print(f"\nBest epoch: {best_epoch}/{N_EPOCHS}, mean AUC (distilled) = {best_auc:.4f}")
print("Reference (RadImageNet without distillation): 0.7907 on holdout, 0.741 on the leaderboard.")
