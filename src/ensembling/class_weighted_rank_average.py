"""
Rank-average ensemble with a separate weight per (model, class).

Each model's weight for a class is its AUC advantage over chance on that class,
so models that are better on a class count more there. The weights are fit and
evaluated on the same holdout, so the resulting AUC is optimistic (leaky) and
should be treated as exploratory until confirmed on a real submission.
Stage 12; the equal-weight baseline is plain rank averaging.

Expects all_seed_probs and val_labels from the seed-variant training script
to already exist in the session (optional; the synthetic check runs standalone).
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def class_weighted_rank_average(all_probs, val_labels):
    """Combine all_probs (n_models, n_val, n_classes) into (n_val, n_classes) using AUC-based per-class weights."""
    n_models, n_val, n_classes = all_probs.shape
    combined = np.zeros((n_val, n_classes))

    for c in range(n_classes):
        ranks_per_model = np.array([rankdata(all_probs[m, :, c]) for m in range(n_models)])
        aucs_per_model = []
        for m in range(n_models):
            try:
                auc = roc_auc_score(val_labels[:, c], all_probs[m, :, c])
            except ValueError:
                auc = 0.5  # degenerate class for this model: treat as chance
            aucs_per_model.append(auc)
        aucs_per_model = np.array(aucs_per_model)

        # Weight = AUC advantage over chance, clipped at 0 (no negative weights).
        weights = np.clip(aucs_per_model - 0.5, 0, None)
        if weights.sum() == 0:
            # No model beats chance on this class: fall back to uniform weights.
            weights = np.ones(n_models)

        combined[:, c] = (ranks_per_model * weights[:, None]).sum(axis=0) / weights.sum()

    return combined


if __name__ == "__main__":
    # Synthetic check: model 0 is perfect on class 0 and random on class 1;
    # model 1 is the opposite. The blend should follow the better model per class.
    rng = np.random.default_rng(0)
    n_val = 20
    val_labels_test = np.zeros((n_val, 2))
    val_labels_test[: n_val // 2, 0] = 1
    val_labels_test[:, 1] = rng.integers(0, 2, n_val)

    probs_test = np.zeros((2, n_val, 2))
    probs_test[0, :, 0] = val_labels_test[:, 0]  # model 0: perfect on class 0
    probs_test[0, :, 1] = rng.random(n_val)  # model 0: random on class 1
    probs_test[1, :, 0] = rng.random(n_val)  # model 1: random on class 0
    probs_test[1, :, 1] = val_labels_test[:, 1]  # model 1: perfect on class 1

    result = class_weighted_rank_average(probs_test, val_labels_test)
    auc_c0 = roc_auc_score(val_labels_test[:, 0], result[:, 0])
    auc_c1 = roc_auc_score(val_labels_test[:, 1], result[:, 1])
    assert auc_c0 > 0.9, f"expected ~1.0 on class 0 (model 0 dominates), got {auc_c0}"
    assert auc_c1 > 0.9, f"expected ~1.0 on class 1 (model 1 dominates), got {auc_c1}"
    print(f"[OK] class_weighted_rank_average routes per class correctly (AUC {auc_c0:.3f}, {auc_c1:.3f})")

    # Run on real session data if available.
    try:
        combined_weighted = class_weighted_rank_average(all_seed_probs, val_labels.numpy())
        aucs = []
        for i in range(12):
            try:
                aucs.append(roc_auc_score(val_labels[:, i].numpy(), combined_weighted[:, i]))
            except ValueError:
                pass
        mean_auc = sum(aucs) / len(aucs)
        print(f"\nClass-weighted ensemble: mean AUC = {mean_auc:.4f}")
        print("Equal-weight rank averaging baseline: 0.6392")
        print("NOTE: this number is optimistic (weights chosen and evaluated on the same holdout); "
              "confirm with a real submission.")
    except NameError:
        pass
