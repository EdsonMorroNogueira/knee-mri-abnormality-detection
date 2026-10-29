"""Infer knee laterality (L/R) from image geometry instead of the Laterality tag.

The Laterality tag is missing in about half of the studies. In DICOM patient
coordinates +x points to the patient's LEFT, so the sign of the x coordinate of
the image CENTER tells which knee was scanned: negative -> right, positive ->
left. The center is used rather than the corner (ImagePositionPatient) because
the corner is half a field of view away, enough to flip the sign for a knee
near the body midline. (Stage 3, step 2.)
"""
import numpy as np
import pydicom
import pandas as pd
from pathlib import Path

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
IMAGES_ROOT = Path(ROOT) / "train_series"

series_df = pd.read_csv(f"{ROOT}/train_series.csv")
# Small sample on purpose: proof of concept before running all 24371 rows.
sample = series_df[["StudyInstanceUID", "SeriesInstanceUID"]].head(10)


def image_center_x(dcm_path) -> float | None:
    """x coordinate of the image CENTER in patient space."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
        return None
    ipp = np.array([float(v) for v in ds.ImagePositionPatient])
    iop = np.array([float(v) for v in ds.ImageOrientationPatient])
    row_vec, col_vec = iop[:3], iop[3:]
    spacing = [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    n_rows, n_cols = int(ds.Rows), int(ds.Columns)
    # center = corner + half the width along the row + half the height along the column
    center = (ipp
              + row_vec * spacing[1] * n_cols / 2
              + col_vec * spacing[0] * n_rows / 2)
    return center[0]  # x coordinate


for _, row in sample.iterrows():
    series_dir = IMAGES_ROOT / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
    dcm_files = list(series_dir.rglob("*.dcm"))
    if not dcm_files:
        continue
    # One arbitrary slice is enough to estimate the side and is cheaper than reading the series.
    cx = image_center_x(dcm_files[0])
    tag_laterality = pydicom.dcmread(dcm_files[0], stop_before_pixels=True).get("Laterality", None)
    print(f"Study {row['StudyInstanceUID'][-8:]}... series {row['SeriesInstanceUID'][-8:]}...: "
          f"center x={cx}, Laterality tag={tag_laterality}")
