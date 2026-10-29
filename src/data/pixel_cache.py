"""Cache decoded slot images as uint8 for fine-tuning.

Fine-tuning revisits the same images every epoch, so each DICOM is decoded
once and stored as bytes. uint8 takes 4x less memory than float32
(read_slot_image already normalizes to [0, 1]).

Requires `read_slot_image` (not defined here) to be available in
the same session; helpers are inlined because per-file imports do not work
when pasted into a Kaggle notebook cell. (Stage 4.)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"

SLOTS = [
    (f"{plane}_{fluid}{fs}", plane, fluid, fs)
    for plane in ["Sagittal", "Coronal", "Axial"]
    for fluid, fs in [(0, 0), (1, 1)]
]


def count_slices(study_id: str, series_id: str) -> int:
    series_dir = Path(ROOT) / "train_series" / study_id / series_id
    return len(list(series_dir.rglob("*.dcm")))


def pick_slots_for_study(study_df: pd.DataFrame) -> dict:
    """Same function as in slot_selection.py, inlined for notebook use."""
    chosen = {}
    for name, plane, fluid, fs in SLOTS:
        candidates = study_df[
            (study_df["Anatomical_Plane"] == plane)
            & (study_df["Fluid_Sensitive"] == fluid)
            & (study_df["Fat_Suppression"] == fs)
        ]
        if len(candidates) == 0:
            continue
        elif len(candidates) == 1:
            chosen[name] = candidates["SeriesInstanceUID"].iloc[0]
        else:
            best_n = -1
            for _, row in candidates.iterrows():
                n = count_slices(row["StudyInstanceUID"], row["SeriesInstanceUID"])
                if n > best_n:
                    best_n = n
                    chosen[name] = row["SeriesInstanceUID"]
    return chosen


def build_pixel_cache(study_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    series_df = pd.read_csv(f"{ROOT}/train_series.csv")
    n_studies = len(study_ids)
    cache = np.zeros((n_studies, len(SLOTS), 3, 224, 224), dtype=np.uint8)
    mask = np.zeros((n_studies, len(SLOTS)), dtype=np.float32)

    slot_index = {name: idx for idx, (name, *_) in enumerate(SLOTS)}

    for i, study_id in enumerate(study_ids):
        study_df = series_df[series_df["StudyInstanceUID"] == study_id]
        chosen = pick_slots_for_study(study_df)
        for name, series_id in chosen.items():
            img_float = read_slot_image(study_id, series_id)  # (3, 224, 224), [0, 1]
            j = slot_index[name]
            # round before the uint8 cast so 0.999 becomes 255, not 254
            cache[i, j] = np.round(img_float * 255)
            mask[i, j] = 1.0
        print(f"study {i + 1}/{n_studies} ({study_id[-8:]}...): " f"{len(chosen)}/{len(SLOTS)} slots present")

    return cache, mask


if __name__ == "__main__":
    df = pd.read_csv(f"{ROOT}/train.csv")
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
    has_any_target = df[TARGET_COLUMNS].notna().any(axis=1)
    gold_study_ids = df[has_any_target]["StudyInstanceUID"].tolist()
    print(f"Total gold studies: {len(gold_study_ids)} (expected 58)")

    cache, mask = build_pixel_cache(gold_study_ids[:5])  # small sample first
    print(f"cache shape: {cache.shape} (expected (5, 6, 3, 224, 224))")
    print(f"cache dtype: {cache.dtype} (expected uint8)")
    print(f"cache min/max: {cache.min()} / {cache.max()} (expected within 0-255)")
    print(f"mask:\n{mask}")
