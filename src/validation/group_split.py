"""
Single 80/20 train/validation split that never separates studies sharing the same report.

Studies with identical report text (same report_hash) are near-duplicates, so
splitting them across train and validation would leak labels into the holdout.
Whole report_hash groups are therefore assigned to one side. The validation size
can overshoot the target slightly because groups are indivisible.
"""

import numpy as np
import pandas as pd

VAL_FRACTION = 0.2
SEED = 42


def group_train_val_split(gold_df: pd.DataFrame, val_fraction=VAL_FRACTION, seed=SEED):
    """Return (train_df, val_df); every study with the same report_hash lands on the same side."""
    groups = gold_df.groupby("report_hash")["StudyInstanceUID"].apply(list).to_dict()
    group_hashes = list(groups.keys())

    rng = np.random.default_rng(seed)
    rng.shuffle(group_hashes)

    n_total = len(gold_df)
    target_val_size = round(n_total * val_fraction)

    val_study_ids = []
    train_study_ids = []
    for h in group_hashes:
        if len(val_study_ids) < target_val_size:
            val_study_ids.extend(groups[h])
        else:
            train_study_ids.extend(groups[h])
    train_df = gold_df[gold_df["StudyInstanceUID"].isin(train_study_ids)].reset_index(drop=True)
    val_df = gold_df[gold_df["StudyInstanceUID"].isin(val_study_ids)].reset_index(drop=True)
    return train_df, val_df


if __name__ == "__main__":
    train_df, val_df = group_train_val_split(gold_df)
    print(
        f"train: {len(train_df)} studies, validation: {len(val_df)} studies "
        f"(target was ~{round(len(gold_df) * VAL_FRACTION)} for validation)"
    )
    overlap = set(train_df["report_hash"]) & set(val_df["report_hash"])
    print(f"report hashes on BOTH sides (expected: empty set): {overlap}")
