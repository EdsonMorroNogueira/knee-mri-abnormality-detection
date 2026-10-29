"""Crop/resize in physical scale (mm) rather than fixed pixel sizes.

An N x N slice with spacing s mm/pixel covers N*s mm of anatomy. Resizing to a
fixed pixel size without considering s puts series with different physical
spacing at different scales for the model.

Nyquist limit: for a d mm structure to survive the resize, the effective
spacing (mm/pixel) must be <= d/2. A ~1 mm meniscus tear needs <= 0.5 mm/pixel;
above that the information is destroyed before the first convolution,
whatever the architecture. (Stage 3b, step 2; argument from the top-1
notebook's "Sampling" section.)
"""
import pydicom
from pathlib import Path

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
IMAGES_ROOT = Path(ROOT) / "train_series"

# Same sample series as dicom_slice_ordering.py.
STUDY_ID = "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260"
SERIES_ID = "1.2.826.0.1.3680043.8.498.12343110195036213483454091715412333772"

series_dir = IMAGES_ROOT / STUDY_ID / SERIES_ID
sample_dcm = next(series_dir.rglob("*.dcm"))
ds = pydicom.dcmread(sample_dcm, stop_before_pixels=True)
pixel_spacing = [float(v) for v in ds.PixelSpacing]  # [row, col] mm/pixel
print(f"Actual PixelSpacing of this series: {pixel_spacing}")
print(f"Rows x Columns: {ds.Rows} x {ds.Columns}")
print(f"Actual field of view: {ds.Rows * pixel_spacing[0]:.1f}mm x "
      f"{ds.Columns * pixel_spacing[1]:.1f}mm")


# Reference values from the top-1 notebook for crop_mm=130: 0.580 mm/pixel at 224px, 0.387 mm/pixel at 336px.
def crop_and_resize_params(pixel_spacing_mm, crop_mm=130, resize_px=224) -> dict:
    s = sum(pixel_spacing_mm) / len(pixel_spacing_mm) # arithmetic mean
    n_crop_px = round(crop_mm / s)
    mm_per_pixel_eff = crop_mm / resize_px

    return {"n_crop_px": n_crop_px, "mm_per_pixel_eff": mm_per_pixel_eff}


for resize_px in [224, 336]:
    params = crop_and_resize_params(pixel_spacing, resize_px=resize_px)
    ok = "OK" if params["mm_per_pixel_eff"] <= 0.5 else "BELOW LIMIT"
    print(f"\nresize={resize_px}px: {params} -> Nyquist for a 1mm tear: {ok}")
