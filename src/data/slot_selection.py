"""Pick one series per plane + sequence-type combination ("slot") for each study.

Same idea as pick_slots() in the top-1 notebook, adapted to the two sequence
types that appear in this CSV. When several series match a slot, the one with
the most slices wins (the top-1 notebook's tie-break rule). (Stage 3b, step 1.)
"""
import pandas as pd
from pathlib import Path

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
IMAGES_ROOT = Path(ROOT) / "train_series"

series_df = pd.read_csv(f"{ROOT}/train_series.csv")

PLANES = ["Sagittal", "Coronal", "Axial"]
# (fluid, fat_suppression): only the 2 combinations present in the CSV.
SEQ_TYPES = [(0, 0), (1, 1)]

SLOTS = [
    (f"{plane}_{fluid}{fs}", plane, fluid, fs)
    for plane in PLANES
    for fluid, fs in SEQ_TYPES
]
print(f"Slots defined: {[s[0] for s in SLOTS]}")


def count_slices(study_id: str, series_id: str) -> int:
    series_dir = IMAGES_ROOT / study_id / series_id
    return len(list(series_dir.rglob("*.dcm")))


# Returns {slot_name: SeriesInstanceUID} for one study; empty slots are omitted.
def pick_slots_for_study(study_df: pd.DataFrame) -> dict:
    chosen = {}
    for name, plane, fluid, fs in SLOTS:
        candidates = study_df[
            (study_df["Anatomical_Plane"] == plane) &
            (study_df["Fluid_Sensitive"] == fluid) &
            (study_df["Fat_Suppression"] == fs)
        ]
        if len(candidates) == 0:
            continue
        elif len(candidates) == 1:
            chosen[name] = (candidates["SeriesInstanceUID"].iloc[0])
        else:
            best_n = -1
            for _, row in candidates.iterrows():
                n = count_slices(row["StudyInstanceUID"], row["SeriesInstanceUID"])
                if n > best_n:
                    best_n = n
                    chosen[name] = (row["SeriesInstanceUID"])
    return chosen

# Small sample first: counting files per candidate across all 4407 studies is slow.
sample_studies = series_df["StudyInstanceUID"].unique()[:5]
for study_id in sample_studies:
    study_df = series_df[series_df["StudyInstanceUID"] == study_id]
    chosen = pick_slots_for_study(study_df)
    print(f"\nStudy {study_id[-8:]}...: {len(study_df)} series available")
    for slot_name, series_id in chosen.items():
        print(f"  {slot_name}: series {series_id[-8:]}...")
    missing = [s[0] for s in SLOTS if s[0] not in chosen]
    if missing:
        print(f"  (empty slots: {missing})")
