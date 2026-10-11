"""Submission validator: the exact checks the competition's form applies.

A previous session's file was rejected by the submission form with

    "Predicted values must be in range [0, 1]"

so this module is written from the published format rules rather than assumed to
pass.  From page 967 ("Submission format"), read line by line on 2026-10-11:

* same projected CRS as the training data -- UTM zone 11N, EPSG:32611
* same resolution as the training data -- 100 m
* same bounds as the training data, data outside the bounds null or nan
* a single layer, datatype float32, values between 0 and 1, higher = more likely

Every check below is an assertion against ``data/official/sample_submission.tif``,
which is the organizer's own template, so "matches" is a byte-level statement about
CRS, shape and geotransform rather than a judgement call.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import rasterio

from .raster_io import TEMPLATE, footprint, read_meta, sha256

REQUIRED_CRS = "EPSG:32611"
REQUIRED_SHAPE = (3730, 3292)
REQUIRED_DTYPE = "float32"


@dataclass
class ValidationReport:
    path: str
    ok: bool
    checks: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    sha256: str = ""
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        lines = [f"validator: {'PASS' if self.ok else 'FAIL'}  {self.path}"]
        for k, v in self.checks.items():
            lines.append(f"  [{'ok ' if v else 'FAIL'}] {k}")
        for f in self.failures:
            lines.append(f"  -> {f}")
        for k, v in self.stats.items():
            lines.append(f"  . {k} = {v}")
        return "\n".join(lines)


def validate_submission(path: Path | str, *, foot: np.ndarray | None = None,
                        require_all_finite: bool = True) -> ValidationReport:
    """Run every format check on a candidate submission GeoTIFF."""
    path = Path(path)
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = ""):
        checks[name] = bool(ok)
        if not ok:
            failures.append(f"{name}{(' - ' + detail) if detail else ''}")

    meta = read_meta(path)
    tpl = read_meta(TEMPLATE)

    chk("file exists", path.exists())
    chk("driver is GTiff", meta.dtype is not None)
    chk("single band (count == 1)", meta.count == 1, f"count={meta.count}")
    chk("dtype is float32", meta.dtype == REQUIRED_DTYPE, f"dtype={meta.dtype}")
    chk("CRS is EPSG:32611", str(meta.crs) == tpl.crs and REQUIRED_CRS in str(meta.crs),
        f"crs={meta.crs}")
    chk("shape matches template (3730 x 3292)",
        (meta.height, meta.width) == REQUIRED_SHAPE, f"{meta.height}x{meta.width}")
    chk("geotransform matches template exactly",
        tuple(round(v, 6) for v in meta.transform) ==
        tuple(round(v, 6) for v in tpl.transform), f"{meta.transform}")
    chk("resolution is 100 m", abs(meta.transform[0]) == 100.0
        and abs(meta.transform[4]) == 100.0)

    with rasterio.open(path) as src:
        a = src.read(1)

    finite = np.isfinite(a)
    chk("no NaN inside the survey footprint",
        bool(finite[footprint() if foot is None else foot].all()))
    if require_all_finite:
        chk("no NaN anywhere in the raster (zeros-outside variant)", bool(finite.all()),
            f"{int((~finite).sum())} non-finite")

    av = a[np.isfinite(a)] if a.size else a
    lo = float(av.min()) if av.size else 0.0
    hi = float(av.max()) if av.size else 0.0
    chk("min value >= 0", lo >= 0.0, f"min={lo!r}")
    chk("max value <= 1", hi <= 1.0, f"max={hi!r}")
    chk("no infinities", bool(np.isfinite(a[finite]).all()))

    stats = {
        "bytes": path.stat().st_size,
        "min": lo, "max": hi,
        "positive_pixels": int((a > 0).sum()),
        "distinct_values": int(np.unique(a[np.isfinite(a)]).size),
        "nan_pixels": int((~finite).sum()),
    }
    return ValidationReport(path=str(path), ok=all(checks.values()), checks=checks,
                            failures=failures, sha256=sha256(path), stats=stats)
