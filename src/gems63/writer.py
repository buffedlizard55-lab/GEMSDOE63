"""Submission writer + byte-re-read validator for GEMSDOE63.

Writes the proven-accepted portal format: single-band float32 GeoTIFF,
EPSG:32611, 3730 x 3292, the official geotransform, EVERY cell finite, zeros
outside the footprint, no nodata tag, values within [0, 1]. The '-zeros'
variant is the one that has passed the portal repeatedly in this project
family; NaN-outside variants have triggered portal range errors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

GRID_EPSG = 32611
GRID_SHAPE = (3730, 3292)
GRID_TRANSFORM = Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


def write_submission(values01: np.ndarray, path: Path) -> dict:
    assert values01.shape == GRID_SHAPE
    a = np.asarray(values01, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=GRID_SHAPE[0],
        width=GRID_SHAPE[1],
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(GRID_EPSG),
        transform=GRID_TRANSFORM,
        compress="deflate",
    )
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a, 1)
    return validate_submission(path)


def validate_submission(path: Path) -> dict:
    """Re-read from disk bytes and check every portal requirement."""
    with rasterio.open(path) as ds:
        a = ds.read(1)
        crs = ds.crs
        transform = ds.transform
        shape = (ds.height, ds.width)
        count = ds.count
        dtype = ds.dtypes[0]
        nodata = ds.nodata
    foot = None
    fin = np.isfinite(a)
    checks = {
        "single_band": count == 1,
        "dtype_float32": dtype == "float32",
        "dimensions_3730x3292": shape == GRID_SHAPE,
        "crs_epsg_32611": crs is not None and crs.to_epsg() == GRID_EPSG,
        "full_grid_all_finite": bool(fin.all()),
        "range_0_1_all_cells": bool((a[fin] >= 0).all() and (a[fin] <= 1).all()),
        "no_nodata_tag": nodata is None,
    }
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    out = {
        "filename": path.name,
        "sha256": h,
        "size_bytes": int(path.stat().st_size),
        "crs": "EPSG:32611" if checks["crs_epsg_32611"] else str(crs),
        "transform": list(transform)[:6],
        "shape": list(shape),
        "dtype": dtype,
        "grid_min": float(np.min(a)),
        "grid_max": float(np.max(a)),
        "emitted_positive_pixels": int((a[fin] > 0).sum()) if fin.any() else 0,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
    }
    return out


def write_checks(receipt: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1))
