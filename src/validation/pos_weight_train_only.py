"""
Compute per-class BCE pos_weight from the training split only.

Using the full gold set (or the validation split) would leak validation label
statistics into the loss that drives training.
"""

import torch

# pos_weight = (1 - rate) / rate per class, in TARGET_COLUMNS order
# (BCEWithLogitsLoss expects it aligned with the output dimension).
weights = []
for col in TARGET_COLUMNS:
    rate = (train_df[col] > 0).mean()
    if rate == 0:
        weight = 1.0  # no positives in train: avoid division by zero
    else:
        weight = (1 - rate) / rate
    weights.append(weight)

pos_weight = torch.tensor(weights, dtype=torch.float32)

print("pos_weight per class (computed on train only):")
for col, w in zip(TARGET_COLUMNS, pos_weight.tolist()):
    print(f"  {col}: {w:.2f}")
