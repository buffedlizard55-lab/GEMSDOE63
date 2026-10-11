"""Validate a candidate submission GeoTIFF against the published format rules.

Checks include one float32 band, EPSG:32611, 100 m resolution, exact reference-grid
shape/transform/CRS, values in [0, 1], and NaN-only outside the official footprint.
Checks that require the reference/features rasters are SKIPPED without them; SKIPPED is
never treated as a pass.

CLI: python -m gems.validate_submission FILE.tif [--reference data/labels.tif]
     [--features data/training_features.tif]
Exit code 0 means every required check passed; otherwise 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .constants import EPSG, FEATURE_NODATA_SENTINEL, PIXEL_M, SHAPE_HW


def _check(name, ok, detail, status=None):
    return {"check": name, "status": status or ("PASS" if ok else "FAIL"), "detail": detail}


def _same_transform(a, b, tolerance: float = 1e-6) -> bool:
    return all(abs(float(x) - float(y)) < tolerance for x, y in zip(a[:6], b[:6]))


def footprint_from_features(features_path: str | Path, shape_hw=None) -> np.ndarray:
    """Footprint = cells whose first feature band is finite and above the nodata sentinel."""
    import rasterio

    with rasterio.open(features_path) as src:
        band = src.read(1).astype(np.float64)
    fp = np.isfinite(band) & (band > FEATURE_NODATA_SENTINEL)
    if shape_hw is not None and fp.shape != tuple(shape_hw):
        raise ValueError(f"features shape {fp.shape} != expected {shape_hw}")
    return fp


def validate(path: str | Path, reference_path: str | Path | None = None,
             features_path: str | Path | None = None) -> dict:
    import rasterio

    path = Path(path)
    checks: list[dict] = []
    if not path.exists():
        return {"file": str(path), "passed": False,
                "checks": [_check("file exists", False, "missing")],
                "has_failures": True, "skipped_checks_mean_unverified": False}

    ref_shape = None
    ref_transform = None
    ref_crs = None
    if reference_path is not None:
        with rasterio.open(reference_path) as ref:
            ref_shape = (ref.height, ref.width)
            ref_transform = ref.transform
            ref_crs = ref.crs

    with rasterio.open(path) as src:
        out_shape = (src.height, src.width)
        out_transform = src.transform
        out_crs = src.crs
        checks.append(_check("single band", src.count == 1, f"count={src.count}"))
        checks.append(_check("dtype float32", src.dtypes[0] == "float32", f"dtype={src.dtypes[0]}"))
        checks.append(_check("CRS EPSG:32611", out_crs is not None and out_crs.to_string() == EPSG,
                             f"crs={out_crs.to_string() if out_crs else None}"))
        expected_shape = ref_shape if ref_shape is not None else SHAPE_HW
        checks.append(_check("shape matches", out_shape == tuple(expected_shape),
                             f"file={out_shape} expected={tuple(expected_shape)}"))
        res_ok = abs(abs(src.res[0]) - PIXEL_M) < 1e-9 and abs(abs(src.res[1]) - PIXEL_M) < 1e-9
        checks.append(_check("resolution 100 m", res_ok, f"res={src.res}"))
        if ref_transform is not None:
            checks.append(_check("transform matches reference", _same_transform(out_transform, ref_transform),
                                 f"file={tuple(out_transform)[:6]} ref={tuple(ref_transform)[:6]}"))
            checks.append(_check("CRS matches reference",
                                 out_crs is not None and ref_crs is not None and out_crs == ref_crs,
                                 f"file={out_crs} ref={ref_crs}"))
        else:
            checks.append(_check("transform matches reference", False,
                                 "SKIPPED: no reference raster supplied", status="SKIPPED"))
            checks.append(_check("CRS matches reference", False,
                                 "SKIPPED: no reference raster supplied", status="SKIPPED"))
        nodata = src.nodata
        nodata_ok = nodata is None or (isinstance(nodata, (float, np.floating)) and np.isnan(nodata))
        checks.append(_check("nodata tag is none or NaN", nodata_ok,
                             f"nodata={nodata} (finite nodata tags are rejected conservatively)"))
        arr = src.read(1).astype(np.float64)

    finite = np.isfinite(arr)
    n_inf = int(np.isinf(arr).sum())
    checks.append(_check("no infinities", n_inf == 0, f"inf_count={n_inf}"))
    vals = arr[finite]
    if vals.size:
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin = vmax = float("nan")
    checks.append(_check("values within [0, 1]", vals.size > 0 and vmin >= 0.0 and vmax <= 1.0,
                         f"min={vmin} max={vmax} finite_cells={int(finite.sum())}"))

    if features_path is not None:
        with rasterio.open(features_path) as feats:
            band = feats.read(1).astype(np.float64)
            feat_shape = (feats.height, feats.width)
            feat_transform = feats.transform
            feat_crs = feats.crs
        fp = np.isfinite(band) & (band > FEATURE_NODATA_SENTINEL)
        if ref_shape is not None:
            grid_shape = ref_shape
            grid_transform = ref_transform
            grid_crs = ref_crs
            grid_label = "reference"
        else:
            grid_shape = out_shape
            grid_transform = out_transform
            grid_crs = out_crs
            grid_label = "candidate (reference absent)"
        checks.append(_check("features grid shape matches", feat_shape == grid_shape,
                             f"features={feat_shape} {grid_label}={grid_shape}"))
        checks.append(_check("features transform matches reference grid",
                             _same_transform(feat_transform, grid_transform),
                             f"features={tuple(feat_transform)[:6]} {grid_label}={tuple(grid_transform)[:6]}"))
        checks.append(_check("features CRS matches reference grid",
                             feat_crs is not None and grid_crs is not None and feat_crs == grid_crs,
                             f"features={feat_crs} {grid_label}={grid_crs}"))
        if ref_shape is None:
            # Alignment to the candidate is useful but does not establish official-grid alignment.
            checks.append(_check("features/reference alignment verified", False,
                                 "SKIPPED: no official reference raster supplied", status="SKIPPED"))

        if fp.shape != arr.shape:
            checks.append(_check("footprint shape matches candidate", False,
                                 f"footprint={fp.shape} candidate={arr.shape}"))
        else:
            nan_inside = int((~finite & fp).sum())
            checks.append(_check("no NaN inside footprint", nan_inside == 0,
                                 f"nan_inside={nan_inside} footprint_cells={int(fp.sum())}"))
            outside_finite = int((finite & ~fp).sum())
            checks.append(_check("NaN-only outside footprint", outside_finite == 0,
                                 f"finite_outside={outside_finite}; only NaN is allowed outside"))
    else:
        checks.append(_check("footprint checks", False,
                             "SKIPPED: no features raster supplied", status="SKIPPED"))

    nonzero = int(np.nansum(arr > 0))
    checks.append(_check("info: emitted cells (value > 0)", True, f"emitted={nonzero}", status="INFO"))
    failed = [c for c in checks if c["status"] == "FAIL"]
    skipped = [c for c in checks if c["status"] == "SKIPPED"]
    return {"file": str(path), "passed": not failed and not skipped, "has_failures": bool(failed),
            "checks": checks, "skipped_checks_mean_unverified": bool(skipped)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--reference", help="labels.tif (or any file with the official grid)")
    ap.add_argument("--features", help="training features raster, for the footprint check")
    args = ap.parse_args(argv)
    report = validate(args.file, args.reference, args.features)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
