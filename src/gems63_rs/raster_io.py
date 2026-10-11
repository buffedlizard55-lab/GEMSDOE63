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


FOOTPRINT_SOURCE: str = ""
"""Which raster the last :func:`footprint` call derived the mask from.

``"features"`` is authoritative.  ``"labels-fallback"`` means the 400 MB feature stack
was absent and the mask came from ``labels.tif`` instead.  The two differ by 4,601 px
out of 5.17 M (0.09%), so a file written on the fallback may place NaN where the
organizer's own coverage says data belongs.  The run card records this and the
validator prints it; it is not something to discover after a submission is rejected.
"""


def footprint() -> np.ndarray:
    """Pixels of survey coverage: where the official feature stack has data.

    Outside this footprint the competition asks for null/NaN; inside it, values must
    be finite and in [0, 1].  ``data == nodata`` (the float32 -3.4e38 sentinel) is
    the definition of "no data" used by the reference solution notebook.

    The feature stack is 400 MB and is deliberately not committed, so on a clean clone
    it is absent.  Rather than raise, this falls back to the valid region of
    ``labels.tif`` and sets :data:`FOOTPRINT_SOURCE` to say so.  Measured difference
    between the two masks on this data: 4,601 px of 5,165,852 (0.09%).
    """
    global FOOTPRINT_SOURCE
    if FEATURES.exists():
        with rasterio.open(FEATURES) as src:
            nd = src.nodata
            b1 = src.read(1)
        FOOTPRINT_SOURCE = "features"
        return ~(b1 <= nd) if nd is not None else np.isfinite(b1)
    with rasterio.open(LABELS) as src:
        nd = src.nodata
        lab = src.read(1)
    FOOTPRINT_SOURCE = "labels-fallback"
    return (lab != nd) if nd is not None else np.ones(lab.shape, dtype=bool)


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
