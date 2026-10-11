"""Raster I/O pinned to the official sample_submission template.

Every submission written by this repository copies its CRS, shape, transform and
dtype from ``data/official/sample_submission.tif`` byte-for-byte, so the
competition's format check cannot fail on geometry.  The template itself was read
and its metadata asserted in ``tests/test_raster_io.py``:

    3292 x 3730, 1 band, float32, EPSG:32611,
    Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0), nodata = nan
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "data" / "official"
CACHE = REPO / "data" / "cache"
TEMPLATE = OFFICIAL / "sample_submission.tif"
LABELS = OFFICIAL / "labels.tif"
FEATURES = CACHE / "gems-geodawn-numerical-features.tif"

EXPECTED_SHAPE = (3730, 3292)
EXPECTED_CRS = "EPSG:32611"
EXPECTED_TRANSFORM = Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


@dataclass
class RasterMeta:
    width: int
    height: int
    count: int
    dtype: str
    crs: str
    nodata: float | None
    transform: tuple

    def to_dict(self) -> dict:
        return asdict(self)


def read_meta(path: Path | str) -> RasterMeta:
    with rasterio.open(path) as src:
        return RasterMeta(
            width=src.width, height=src.height, count=src.count,
            dtype=str(src.dtypes[0]), crs=str(src.crs),
            nodata=(None if src.nodata is None else float(src.nodata)),
            transform=tuple(src.transform)[:6],
        )


def read_band(path: Path | str, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band)


def read_features() -> tuple[np.ndarray, list[dict]]:
    """The 19-band GeoDAWN numerical feature stack plus its per-band tags."""
    with rasterio.open(FEATURES) as src:
        data = src.read()
        tags = [dict(src.tags(i)) for i in range(1, src.count + 1)]
    return data, tags


def footprint() -> np.ndarray:
    """Pixels where the official feature stack has data (survey coverage).

    Outside this footprint the competition asks for null/NaN; inside it, values must
    be finite and in [0, 1].  ``data == nodata`` (the float32 -3.4e38 sentinel) is
    the definition of "no data" used by the reference solution notebook.
    """
    with rasterio.open(FEATURES) as src:
        nd = src.nodata
        b1 = src.read(1)
    return ~(b1 <= nd) if nd is not None else np.isfinite(b1)


def catalogue() -> np.ndarray:
    """USGS/INGENIOUS known-fault pixels (labels.tif, value 1)."""
    return read_band(LABELS) > 0


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_submission(arr: np.ndarray, path: Path | str, *, nodata=np.nan) -> dict:
    """Write a float32 submission whose geometry is copied from the template.

    ``arr`` must be finite and within [0, 1]; that is asserted, not assumed, because
    a previous session's download was rejected by the submission form with
    "Predicted values must be in range [0, 1]".
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.shape != EXPECTED_SHAPE:
        raise ValueError(f"shape {a.shape} != template {EXPECTED_SHAPE}")
    if not np.isfinite(a).all():
        bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"{bad} non-finite values inside the array")
    lo, hi = float(a.min()), float(a.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"values out of [0, 1]: [{lo}, {hi}]")

    with rasterio.open(TEMPLATE) as tpl:
        profile = tpl.profile.copy()
    # The template is *striped* with a block as wide as the raster (3292 px), which
    # libtiff rejects as a tile size ("blocks must be multiples of 16").  Write a
    # tiled layout with an explicit 256x256 tile instead; CRS, shape, transform,
    # dtype and nodata -- the things the format check actually compares -- are all
    # inherited from the template unchanged.
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=nodata,
                   compress="deflate", predictor=2, tiled=True,
                   blockxsize=256, blockysize=256)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a, 1)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
            "min": lo, "max": hi, "positive_pixels": int((a > 0).sum()),
            "meta": read_meta(path).to_dict()}
