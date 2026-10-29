"""Order the slices of a DICOM series by their real physical position.

The filename (SOP Instance UID) is only a unique identifier, not a spatial
order: the top-1 notebook measured rho~0.01 between filename order and true
position. The correct order comes from two header tags:

- ImageOrientationPatient: 6 numbers (two 3D unit vectors) giving the row and
  column directions of the image in patient space.
- ImagePositionPatient: (x, y, z) of the image corner in patient space.

The slice normal is the cross product of the two orientation vectors.
Projecting each slice position onto it (dot product) gives a scalar k that
increases or decreases monotonically along the stack, which is the real order.
(Stage 3, step 1.)
"""
import numpy as np
import pydicom
from pathlib import Path
from scipy.stats import spearmanr

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
IMAGES_ROOT = Path(ROOT) / "train_series"

# Sample series known to exist.
STUDY_ID = "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260"
SERIES_ID = "1.2.826.0.1.3680043.8.498.12343110195036213483454091715412333772"

series_dir = IMAGES_ROOT / STUDY_ID / SERIES_ID
dcm_files = sorted(series_dir.rglob("*.dcm"))  # filename order (the wrong one)
print(f"Slices in series: {len(dcm_files)}")

rows = []
for path in dcm_files:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    row_vec, col_vec = iop[:3], iop[3:]
    normal = np.cross(row_vec, col_vec)
    k = np.dot(ipp, normal)  # scalar projection onto the normal; this is what orders the slices
    rows.append({"filename": path.name, "k": k})

# Filename order (the "wrong" one)
name_order = [r["filename"] for r in rows]
# Physical-position order (the correct one)
physical_order = [r["filename"] for r in sorted(rows, key=lambda r: r["k"])]

print("\nFilename order (first 5):")
print(name_order[:5])
print("\nPhysical-position order (first 5):")
print(physical_order[:5])
print(f"\nAre the two orders identical? {name_order == physical_order}")

# Spearman rank correlation between filename order and physical order
# (the top-1 notebook measured rho~0.01).
physical_ranks = []
for path in dcm_files:
  idx = physical_order.index(path.name)
  physical_ranks.append(idx)


spearmanr_results = spearmanr(range(len(dcm_files)), physical_ranks)
correlation = spearmanr_results.correlation
pvalue = spearmanr_results.pvalue

print(f"Correlation: {correlation}")
print(f"P-Value: {pvalue}")
