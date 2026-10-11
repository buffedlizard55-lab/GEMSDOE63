"""Local preflight validation for a GEMS submission GeoTIFF.

Checks grid metadata and the project's conservative all-finite, ``[0,1]``
export policy. The portal error's cause remains unproven (IR-57-NAN-02); the
all-finite policy is a local precaution, not a claim about organizer nodata
handling. A local PASS is not organizer acceptance or a uniqueness clearance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio

from .grid import EPSG, HEIGHT, WIDTH, TRANSFORM

SENTINELS = (-9999.0, -999.0, 9999.0, -3.4e38, 1e30)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(path: Path, footprint: np.ndarray | None = None,
             catalogue: np.ndarray | None = None) -> dict:
    """Run local format/range checks on one GeoTIFF; return a receipt dict."""
    path = Path(path)
    with rasterio.open(path) as ds:
        a = ds.read(1)
        meta = dict(crs=str(ds.crs), shape=[ds.height, ds.width], dtype=ds.dtypes[0],
                    count=ds.count, nodata=None if ds.nodata is None else float(ds.nodata),
                    transform=list(tuple(ds.transform)[:6]),
                    profile_compress=ds.profile.get("compress"))
        desc = ds.descriptions
        h, w = ds.height, ds.width

    if a.ndim != 2:
        raise ValueError('prediction TIFF must contain a 2-D first band')
    for mask_name, mask in (("footprint", footprint), ("catalogue", catalogue)):
        if mask is not None and np.asarray(mask).shape != a.shape:
            raise ValueError(f"{mask_name} shape {np.asarray(mask).shape} != raster shape {a.shape}")
    footprint = None if footprint is None else np.asarray(footprint, dtype=bool)
    catalogue = None if catalogue is None else np.asarray(catalogue, dtype=bool)

    finite = np.isfinite(a)
    has_finite = bool(finite.any())
    finite_values = a[finite]
    lo = float(finite_values.min()) if has_finite else None
    hi = float(finite_values.max()) if has_finite else None
    mean = float(finite_values.mean()) if has_finite else None
    in_range = bool(has_finite and lo >= 0.0 and hi <= 1.0)
    checks = {
        "single_band": meta["count"] == 1,
        "dtype_float32": meta["dtype"] == "float32",
        f"dimensions_{HEIGHT}x{WIDTH}": (h, w) == (HEIGHT, WIDTH),
        f"crs_epsg_{EPSG}": meta["crs"] == f"EPSG:{EPSG}",
        "transform_matches_sample_submission": meta["transform"] == list(tuple(TRANSFORM)[:6]),
        "no_nan_anywhere": bool(finite.all()),
        "no_inf_anywhere": bool(not np.isinf(a).any()),
        "no_sentinel_values": bool(not np.isin(finite_values, SENTINELS).any()),
        "range_0_1_guaranteed": in_range,
        "min_ge_0": bool(has_finite and lo >= 0.0),
        "max_le_1": bool(has_finite and hi <= 1.0),
    }
    if footprint is not None:
        n_fp = int(footprint.sum())
        checks["in_footprint_all_finite"] = bool(n_fp > 0 and finite[footprint].all())
        checks["in_footprint_range_0_1"] = bool(
            n_fp > 0 and np.isfinite(a[footprint]).all()
            and (a[footprint] >= 0.0).all() and (a[footprint] <= 1.0).all())
        checks["outside_footprint_zero"] = bool(np.all(a[~footprint] == 0.0))
    if catalogue is not None:
        checks["zero_dots_on_mapped_catalogue"] = int((a[catalogue] > 0).sum()) == 0

    receipt = {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "meta": meta,
        "band_description": list(desc),
        "n_finite": int(finite.sum()),
        "n_nan": int(np.isnan(a).sum()),
        "n_infinite": int(np.isinf(a).sum()),
        "min": lo, "max": hi, "mean": mean,
        "emitted_positive_pixels": int((a > 0).sum()),
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "validation_class": "local format/range preflight; not organizer acceptance or uniqueness clearance",
    }
    if catalogue is not None:
        receipt["on_catalogue_positive_pixels"] = int((a[catalogue] > 0).sum())
    if footprint is not None:
        n_fp = int(footprint.sum())
        receipt["in_footprint_positive_pixels"] = int((a[footprint] > 0).sum())
        receipt["footprint_fraction"] = float(receipt["in_footprint_positive_pixels"] / n_fp) if n_fp else 0.0
    return receipt


def assert_submittable(receipt: dict) -> None:
    """Raise unless every local format/range check passed (not organizer acceptance)."""
    bad = [k for k, v in receipt["checks"].items() if not v]
    if bad:
        raise AssertionError(f"submission fails the local format/range preflight: {bad}")
