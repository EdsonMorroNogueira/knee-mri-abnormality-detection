"""
Read the central slice of a real DICOM series, apply the physical crop and
resize, and return a 2D image ready for DINOv2 (3 channels, values in [0, 1]).

Why the central slice and not the whole series: DINOv2 expects ONE 2D image
per input, and the goal here is to validate the pipeline end to end with as
few new pieces as possible. Grouping consecutive slices (as the top-1
notebook does) is left for the fine-tuning stage.
"""

import numpy as np
import pydicom
from pathlib import Path
from PIL import Image

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
IMAGES_ROOT = Path(ROOT) / "train_series"


def crop_and_resize_params(pixel_spacing_mm, crop_mm=130, resize_px=224) -> dict:
    """Inlined copy of the physical-scale crop helper from the slot-sampling study.

    Scripts are pasted into Kaggle notebook cells, where `__file__` does not
    exist, so importing from a sibling file by path is not possible.
    """
    s = sum(pixel_spacing_mm) / len(pixel_spacing_mm)
    n_crop_px = round(crop_mm / s)
    mm_per_pixel_eff = crop_mm / resize_px
    return {"n_crop_px": n_crop_px, "mm_per_pixel_eff": mm_per_pixel_eff}


def order_slices_physically(series_dir: Path) -> list[Path]:
    """Order a series' .dcm files by true physical position, not file name."""
    dcm_files = sorted(series_dir.rglob("*.dcm"))
    rows = []
    for path in dcm_files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        iop = np.array([float(x) for x in ds.ImageOrientationPatient])
        ipp = np.array([float(x) for x in ds.ImagePositionPatient])
        row_vec, col_vec = iop[:3], iop[3:]
        normal = np.cross(row_vec, col_vec)
        k = np.dot(ipp, normal)
        rows.append((path, k))
    return [path for path, _ in sorted(rows, key=lambda r: r[1])]


def read_slot_image(study_id: str, series_id: str, resize_px: int = 224) -> np.ndarray:
    series_dir = IMAGES_ROOT / study_id / series_id
    ordered = order_slices_physically(series_dir)
    central_path = ordered[len(ordered) // 2]

    ds = pydicom.dcmread(central_path)
    pixel_spacing = [float(v) for v in ds.PixelSpacing]
    params = crop_and_resize_params(pixel_spacing, crop_mm=130, resize_px=resize_px)
    n_crop_px = params["n_crop_px"]

    pixels = ds.pixel_array.astype(np.float32)

    # Min-max to [0, 1]; epsilon avoids division by zero on a constant slice.
    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-8)

    # Centered crop, clamped to the image size to avoid out-of-range indices.
    h, w = pixels.shape
    crop_h = min(n_crop_px, h)
    crop_w = min(n_crop_px, w)
    offset_h = (h - crop_h) // 2
    offset_w = (w - crop_w) // 2
    crop = pixels[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]

    img = Image.fromarray(crop, mode="F").resize((resize_px, resize_px), Image.BILINEAR)
    img = np.array(img)

    # Replicate grayscale to 3 channels, as DINOv2 expects.
    return np.stack([img, img, img], axis=0).astype(np.float32)


if __name__ == "__main__":
    import pandas as pd

    series_df = pd.read_csv(f"{ROOT}/train_series.csv")
    row = series_df.iloc[0]
    img = read_slot_image(row["StudyInstanceUID"], row["SeriesInstanceUID"])
    print(f"shape: {img.shape} (expected (3, 224, 224))")
    print(f"dtype: {img.dtype} (expected float32)")
    print(f"min/max: {img.min():.4f} / {img.max():.4f} (expected ~0.0 / ~1.0)")
