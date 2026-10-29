"""Validate extract_all_classes_llm against the 58 gold-labeled studies.

Compares per-class precision/recall with the earlier regex baselines.

Run llm_report_extractor.py in an earlier notebook cell, and
`model, tokenizer = load_llm()` in the same session, before this cell. This
script reuses that model and does not call load_llm() again: loading twice
exhausts the Kaggle T4 GPU memory (~14.5GB).
"""

import pandas as pd

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"

df = pd.read_csv(f"{ROOT}/train.csv")
has_any_target = df[TARGET_COLUMNS].notna().any(axis=1)
gold_df = df[has_any_target].copy()
print(f"Validating against {len(gold_df)} gold studies")

llm_predictions = gold_df["Report"].apply(lambda text: extract_all_classes_llm(model, tokenizer, text))
for col in TARGET_COLUMNS:
    gold_df[f"{col}_pred_llm"] = llm_predictions.apply(lambda d: d[col])

REGEX_BASELINE = {
    "Fracture": (0.88, 0.39),
    "ACL": (0.85, 0.46),
    "Effusion": (0.65, 0.43),
    "Synovitis": (0.67, 0.22),
}

print(f"\n{'Class':<18}{'Precision':>10}{'Recall':>10}{'#None':>8}   Regex baseline")
print("-" * 70)
for col in TARGET_COLUMNS:
    real = gold_df[col] > 0
    pred = gold_df[f"{col}_pred_llm"]
    num_none = pred.isna().sum()

    is_positive_pred = pred == 1  # None == 1 is False, so None counts as negative

    true_positives = (real & is_positive_pred).sum()
    false_positives = (~real & is_positive_pred).sum()
    false_negatives = (real & ~is_positive_pred).sum()

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        precision_str = f"{precision:.2f}"
    else:
        precision_str = "n/a"

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        recall_str = f"{recall:.2f}"
    else:
        recall_str = "n/a"

    if col in REGEX_BASELINE:
        baseline_precision, baseline_recall = REGEX_BASELINE[col]
        baseline_str = f"regex: {baseline_precision:.2f}/{baseline_recall:.2f}"
    else:
        baseline_str = "(no regex baseline)"

    print(f"{col:<18}{precision_str:>10}{recall_str:>10}{num_none:>8}   {baseline_str}")
