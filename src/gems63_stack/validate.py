"""Submission validator: the organiser rules (DrivenData problem page, 'Submission format') plus the
error the user reported ("Predicted values must be in range [0, 1]").

Rules checked (every one is reported; the file is valid only if all pass):
  1. single raster band
  2. dtype float32
  3. CRS = EPSG:32611 and grid shape/transform identical to labels.tif (the template)
  4. outside the competition bounds: null/NaN   (checked against labels.tif nodata == -1)
  5. inside the bounds: finite (no NaN, no inf, no nodata sentinel)
  6. inside the bounds: values in [0, 1]
  7. no nodata sentinel leakage (values <= -1e30 or >= 1e30 anywhere)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio

TEMPLATE_CRS = "EPSG:32611"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(path: Path, template_path: Path) -> dict:
    res = {"file": path.name, "sha256": sha256_file(path), "checks": {}, "errors": []}
    with rasterio.open(template_path) as t:
        t_raw = t.read(1)
        t_shape, t_transform, t_crs = (t.height, t.width), tuple(t.transform)[:6], t.crs.to_string()
    inside = t_raw != -1
    with rasterio.open(path) as s:
        arr = s.read()
        res["count"] = s.count
        res["dtype"] = s.dtypes[0]
        res["crs"] = s.crs.to_string() if s.crs else None
        res["shape"] = [s.height, s.width]
        res["transform"] = list(tuple(s.transform)[:6])
        res["nodata"] = None if s.nodata is None else (str(s.nodata))
    a = arr[0].astype(np.float64)
    checks = res["checks"]
    checks["single_band"] = res["count"] == 1
    checks["dtype_float32"] = res["dtype"] == "float32"
    checks["crs_matches_template"] = res["crs"] == TEMPLATE_CRS == t_crs
    checks["shape_matches_template"] = tuple(res["shape"]) == t_shape
    checks["transform_matches_template"] = np.allclose(res["transform"], t_transform, rtol=0, atol=1e-6)
    out_b = ~inside
    checks["outside_bounds_null_or_nan"] = bool(np.isnan(a[out_b]).all()) if out_b.any() else True
    checks["inside_bounds_all_finite"] = bool(np.isfinite(a[inside]).all())
    ins = a[inside]
    finite_ins = ins[np.isfinite(ins)]
    checks["inside_values_in_0_1"] = bool(finite_ins.size and finite_ins.min() >= 0.0 and finite_ins.max() <= 1.0)
    checks["no_sentinel_values"] = bool(not ((a <= -1e30) | (a >= 1e30)).any())
    res["in_bounds_px"] = int(inside.sum())
    res["dot_px_in_bounds"] = int((ins == 1.0).sum())
    res["zero_px_in_bounds"] = int((ins == 0.0).sum())
    res["min_inside"] = float(finite_ins.min()) if finite_ins.size else None
    res["max_inside"] = float(finite_ins.max()) if finite_ins.size else None
    res["nan_inside_count"] = int(np.isnan(ins).sum())
    res["nan_outside_count"] = int(np.isnan(a[out_b]).sum())
    res["all_checks_passed"] = all(checks.values())
    res["errors"] = [k for k, v in checks.items() if not v]
    return res


def dots_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as s:
        a = s.read(1)
    return (a == 1.0)
