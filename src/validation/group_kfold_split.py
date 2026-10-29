"""
Grouped K-fold split over the gold studies, keyed by report hash.

Generalizes the single fixed split to real K folds: no report_hash is shared
between train and validation of the same fold, and every gold study is
validated exactly once (so it trains in N_FOLDS-1 folds, instead of a fixed
handful of studies never being trained on).

Expects `gold_df` (with a report_hash column) to already exist in the session.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

N_FOLDS = 5
SEED = 42


def group_kfold_split(gold_df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED):
    """Return n_folds (train_df, val_df) pairs; each study is in validation in exactly one fold."""
    # GroupKFold has no random_state, so shuffle the rows first; otherwise the
    # original row order decides how groups fall into folds.
    shuffled = gold_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    for train_idx, val_idx in gkf.split(shuffled, groups=shuffled["report_hash"]):
        train_df = shuffled.iloc[train_idx].reset_index(drop=True)
        val_df = shuffled.iloc[val_idx].reset_index(drop=True)
        folds.append((train_df, val_df))
    return folds


if __name__ == "__main__":
    # Synthetic check: 20 studies, 15 unique hashes (5 duplicated in pairs).
    rng = np.random.default_rng(0)
    study_ids = [f"study_{i}" for i in range(20)]
    report_hashes = [f"hash_{i}" for i in range(15)] + [f"hash_{i}" for i in range(5)]  # 5 duplicated
    fake_gold_df = pd.DataFrame({"StudyInstanceUID": study_ids, "report_hash": report_hashes})

    folds = group_kfold_split(fake_gold_df, n_folds=5, seed=0)
    assert len(folds) == 5, f"expected 5 folds, got {len(folds)}"

    all_val_ids = []
    for i, (train_df, val_df) in enumerate(folds):
        overlap = set(train_df["report_hash"]) & set(val_df["report_hash"])
        assert not overlap, f"fold {i}: report_hash {overlap} appears in both train and validation"
        all_val_ids.extend(val_df["StudyInstanceUID"].tolist())

    assert sorted(all_val_ids) == sorted(study_ids), "validation folds do not cover every study exactly once"
    print(f"[OK] group_kfold_split: 5 folds, no report_hash leakage, full coverage of {len(study_ids)} studies")

    # Run on real data if available in the session.
    try:
        real_folds = group_kfold_split(gold_df)
        print(f"\n{len(real_folds)} folds over {len(gold_df)} real gold studies:")
        for i, (train_df, val_df) in enumerate(real_folds):
            print(f"  fold {i}: train={len(train_df)}, validation={len(val_df)}")
    except NameError:
        pass
