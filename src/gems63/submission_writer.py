"""DrivenData Competition GeoTIFF Submission Builder & 12-Point Validator.

Strictly solves and prevents: "Predicted values must be in range [0, 1]".
- Values outside footprint are written as 0.0 with nodata=None in `-zeros.tif`.
- Values inside footprint are strictly in [0.0, 1.0].
- EPSG:32611, 3730x3292, 100m grid, float32.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio

EXPECTED_HEIGHT = 3730
EXPECTED_WIDTH = 3292
EXPECTED_FOOTPRINT_PIXELS = 5167373
EXPECTED_TOTAL_PIXELS = EXPECTED_HEIGHT * EXPECTED_WIDTH
TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_submission(tif_path: Path, foot: np.ndarray, cat: np.ndarray) -> dict:
    """Rigorous 12-point audit verifying DrivenData submission compliance."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        crs_str = str(src.crs)
        transform_tuple = tuple(round(float(x), 4) for x in src.transform)[:6]
        dtypes = src.dtypes
        count = src.count
        height, width = src.height, src.width
        nodata_val = src.nodata

    in_foot = arr[foot]
    out_foot = arr[~foot]

    in_foot_finite = int(np.isfinite(in_foot).sum())
    in_foot_nan = int(np.isnan(in_foot).sum())
    in_foot_inf = int(np.isinf(in_foot).sum())
    in_foot_sentinel = int((in_foot < -1e30).sum())
    in_foot_min = float(np.nanmin(in_foot))
    in_foot_max = float(np.nanmax(in_foot))
    in_foot_in_01 = bool(
        (in_foot_finite == EXPECTED_FOOTPRINT_PIXELS)
        and np.all((in_foot >= 0.0) & (in_foot <= 1.0))
    )
    full_grid_in_01 = bool(
        (int(np.isfinite(arr).sum()) == EXPECTED_TOTAL_PIXELS)
        and np.all((arr >= 0.0) & (arr <= 1.0))
    )
    out_foot_zeros = bool(np.all(out_foot == 0.0))

    emitted_dots = int((in_foot > 0.5).sum())
    on_catalogue_dots = int(((arr > 0.5) & cat).sum())

    checks = {
        "single_band": count == 1,
        "dtype_float32": dtypes[0] == "float32",
        "dimensions_3730x3292": (height == EXPECTED_HEIGHT and width == EXPECTED_WIDTH),
        "crs_epsg_32611": "32611" in crs_str,
        "in_footprint_all_finite": in_foot_finite == EXPECTED_FOOTPRINT_PIXELS,
        "in_footprint_zero_nan": in_foot_nan == 0,
        "in_footprint_zero_inf": in_foot_inf == 0,
        "in_footprint_zero_sentinel": in_foot_sentinel == 0,
        "in_footprint_range_0_1": in_foot_in_01,
        "outside_footprint_zeros": out_foot_zeros,
        "zero_on_catalogue_leakage": on_catalogue_dots == 0,
        "validator_range_0_1_guaranteed": full_grid_in_01,
    }

    all_passed = all(checks.values())
    if not all_passed:
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Submission audit failed for {tif_path.name}: {failed}")

    return {
        "filename": tif_path.name,
        "size_bytes": tif_path.stat().st_size,
        "sha256": sha256_file(tif_path),
        "mode": "zeros",
        "crs": crs_str,
        "transform": list(transform_tuple),
        "shape": [height, width],
        "dtype": dtypes[0],
        "nodata": None if nodata_val is None or np.isnan(nodata_val) else float(nodata_val),
        "emitted_positive_pixels": emitted_dots,
        "footprint_fraction": round(emitted_dots / float(EXPECTED_FOOTPRINT_PIXELS), 6),
        "on_catalogue_positive_pixels": on_catalogue_dots,
        "in_footprint_finite_pixels": in_foot_finite,
        "in_footprint_min": in_foot_min,
        "in_footprint_max": in_foot_max,
        "full_grid_finite_pixels": int(np.isfinite(arr).sum()),
        "checks": checks,
        "all_checks_passed": all_passed,
    }


def write_submission_tif(
    cand_dots: np.ndarray,
    foot: np.ndarray,
    cat: np.ndarray,
    template_tif: Path,
    out_path: Path,
) -> dict:
    """Write compliant GeoTIFF with 0.0 outside footprint and values in [0, 1]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_mask = (cand_dots & foot & ~cat).astype(np.float32)

    with rasterio.open(template_tif) as tpl:
        profile = tpl.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=3,
        zlevel=9,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=None,
    )

    arr = np.where(foot, clean_mask, 0.0).astype(np.float32)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

    return audit_submission(out_path, foot, cat)
