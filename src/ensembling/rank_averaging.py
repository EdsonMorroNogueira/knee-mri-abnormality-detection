"""
Combine several models' predictions by rank averaging.

Each model's probabilities are converted to per-class ranks before averaging.
The metric is AUC, which depends only on ordering, so ranks remove calibration
differences between models that would otherwise let one model dominate a raw
probability mean.

Expects all_seed_probs / val_labels / SEEDS from the seed-variant training
script to already exist in the session (optional; the synthetic check below
runs standalone).
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def compute_auc_per_class(true_labels, probs) -> list:
    aucs = []
    for c in range(true_labels.shape[1]):
        try:
            aucs.append(roc_auc_score(true_labels[:, c].numpy(), probs[:, c]))
        except ValueError:
            pass
    return aucs


def mean_auc(aucs: list) -> float:
    return sum(aucs) / len(aucs)


def rank_average(all_probs: np.ndarray) -> np.ndarray:
    """all_probs has shape (n_models, n_val, n_classes); returns (n_val, n_classes) mean ranks."""
    n_models, n_val, n_classes = all_probs.shape
    combined = np.zeros((n_val, n_classes))
    for c in range(n_classes):
        ranks_per_model = np.array([rankdata(all_probs[m, :, c]) for m in range(n_models)])
        combined[:, c] = ranks_per_model.mean(axis=0)
    return combined


# Run on real session data if present; skipped when the file runs on its own.
try:
    print("Individual AUC of each model:")
    for m, seed in enumerate(SEEDS):
        aucs = compute_auc_per_class(val_labels, all_seed_probs[m])
        print(f"  seed {seed}: mean AUC = {mean_auc(aucs):.4f}")

    combined_ranks = rank_average(all_seed_probs)
    rank_aucs = compute_auc_per_class(val_labels, combined_ranks)
    print(f"\nRank averaging: mean AUC = {mean_auc(rank_aucs):.4f}")
except NameError:
    pass  # no all_seed_probs/val_labels/SEEDS in the session


if __name__ == "__main__":
    # 2 models, 4 studies, 1 class: easy to verify by hand.
    fake_probs = np.array([
        [[0.9], [0.1], [0.5], [0.3]],   # model 0: ranks [4, 1, 3, 2]
        [[0.2], [0.8], [0.6], [0.4]],   # model 1: ranks [1, 4, 3, 2]
    ])
    combined = rank_average(fake_probs)
    expected = np.array([[2.5], [2.5], [3.0], [2.0]])  # mean of the ranks above
    assert combined.shape == (4, 1), f"unexpected shape: {combined.shape}"
    assert np.allclose(combined, expected), f"expected {expected.tolist()}, got {combined.tolist()}"
    print("[OK] rank_average matches the manual calculation")
