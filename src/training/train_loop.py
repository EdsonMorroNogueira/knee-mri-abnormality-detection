"""Training loop on the 58 gold-labelled studies: AdamW with two learning rates, BCEWithLogitsLoss.

Encoder gets a low LR and the attention head a higher one, so the pretrained
weights barely move. Success check: loss should fall visibly from the first to
the last epoch, confirming gradient flows from the pool back to DINOv2's last
block. Introduced in Stage 4.
"""

import pandas as pd
import torch
import torch.nn as nn

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
N_EPOCHS = 10
ENCODER_LR = 1e-5
HEAD_LR = 1e-3

df = pd.read_csv(f"{ROOT}/train.csv")
has_any_target = df[TARGET_COLUMNS].notna().any(axis=1)
gold_df = df[has_any_target].reset_index(drop=True)
study_ids = gold_df["StudyInstanceUID"].tolist()
labels = torch.tensor(gold_df[TARGET_COLUMNS].values, dtype=torch.float32)  # (58, 12)
print(f"Training on {len(study_ids)} gold studies (expected 58)")

cache, mask = build_pixel_cache(study_ids)

model = build_finetune_model()
pool = MultiQueryAttentionPool(dim=model.config.hidden_size, n_slot=6, n_out=12, hidden=64)

# Two parameter groups: trainable encoder params at ENCODER_LR, pool at HEAD_LR.
optimizer = torch.optim.AdamW(
    [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": ENCODER_LR},
        {"params": pool.parameters(), "lr": HEAD_LR},
    ]
)

criterion = nn.BCEWithLogitsLoss()

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    logits, _ = forward_batch(model, pool, cache, mask)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"epoch {epoch + 1}/{N_EPOCHS}: loss={loss.item():.4f}")
