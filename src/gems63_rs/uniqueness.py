"""Lane-drift / uniqueness gate against every earlier raster in the registry.

The parallel-run brief sets two hard gates, checked both on the surface *before*
placement and on the final dots:

* Spearman rank-correlation with any registry raster **> 0.90**  -> duplicate
* more than **70 %** of the candidate's dots within **3 px** of one registry
  raster's dots                                      -> duplicate

Both numbers are computed here rather than asserted.  Rank correlation is taken over
the union of the two emissions' support (with a fixed random subsample when that set
is large, so the estimate is reproducible), and the overlap uses an exact Euclidean
distance transform at the competition's own 300 m = 3 px radius.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

RANK_CORR_GATE = 0.90
OVERLAP_GATE = 0.70
OVERLAP_RADIUS_PX = 3.0


def spearman_rank_corr(a: np.ndarray, b: np.ndarray, *,
                       support: str = "full",
                       max_points: int = 1_000_000,
                       seed: int = 20261011) -> tuple[float, int]:
    """Spearman rho between two emissions, on a fixed reproducible subsample.

    ``support="full"`` (the default) ranks over the whole grid.  That matters: every
    raster in the registry and every candidate here is a **binary** dot field, and
    over the union of two binary supports both vectors are constant 1, so rho is
    undefined and a naive implementation silently returns 0 -- which would pass an
    identical raster as unique.  Over the full grid, Spearman on binary masks is the
    phi coefficient, which is well defined and is 1.0 only for identical rasters.

    ``support="union"`` is kept for graded (non-binary) surfaces, where it is the
    more sensitive comparison.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if support == "union":
        sel = (a > 0) | (b > 0)
    elif support == "full":
        sel = np.ones(a.shape, dtype=bool)
    else:
        raise ValueError(f"unknown support {support!r}")
    idx = np.flatnonzero(sel.ravel())
    if idx.size == 0:
        return 0.0, 0
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    av = a.ravel()[idx]
    bv = b.ravel()[idx]
    if av.std() == 0 and bv.std() == 0:
        return 1.0, int(idx.size)          # both constant and identical
    ra = rankdata(av)
    rb = rankdata(bv)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0, int(idx.size)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return rho, int(idx.size)


def overlap_fraction(candidate: np.ndarray, other: np.ndarray,
                     radius_px: float = OVERLAP_RADIUS_PX) -> float:
    """Fraction of candidate emission pixels within `radius_px` of `other`."""
    c = np.asarray(candidate) > 0
    o = np.asarray(other) > 0
    n = int(c.sum())
    if n == 0:
        return 0.0
    if not o.any():
        return 0.0
    d = ndi.distance_transform_edt(~o)
    return float((d[c] <= radius_px).mean())


@dataclass
class RegistryComparison:
    alias: str
    sha256: str
    rank_corr: float
    overlap_fraction: float
    owner_reported_score: float | None
    rank_gate_exceeded: bool
    overlap_gate_exceeded: bool

    @property
    def duplicate(self) -> bool:
        return self.rank_gate_exceeded or self.overlap_gate_exceeded

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duplicate"] = self.duplicate
        return d


def compare_to_registry(candidate: np.ndarray, registry: dict[str, dict]) -> list[RegistryComparison]:
    """Compare a candidate emission against every registry raster.

    ``registry`` maps an alias to ``{"array": ndarray, "sha256": str,
    "owner_reported_score": float | None}``.
    """
    out = []
    for alias, entry in registry.items():
        other = entry["array"]
        rho, _ = spearman_rank_corr(candidate, other)
        ov = overlap_fraction(candidate, other)
        out.append(RegistryComparison(
            alias=alias, sha256=entry.get("sha256", ""),
            rank_corr=rho, overlap_fraction=ov,
            owner_reported_score=entry.get("owner_reported_score"),
            rank_gate_exceeded=rho > RANK_CORR_GATE,
            overlap_gate_exceeded=ov > OVERLAP_GATE,
        ))
    return out


def uniqueness_gate(comparisons: list[RegistryComparison]) -> dict:
    """Summarise the gate. `clear` True means the candidate is not a lane duplicate."""
    dups = [c for c in comparisons if c.duplicate]
    worst_rho = max((c.rank_corr for c in comparisons), default=0.0)
    worst_ov = max((c.overlap_fraction for c in comparisons), default=0.0)
    return {
        "n_registry_rasters": len(comparisons),
        "clear": len(dups) == 0,
        "duplicates": [c.alias for c in dups],
        "max_rank_corr": float(worst_rho),
        "max_overlap_fraction": float(worst_ov),
        "rank_corr_gate": RANK_CORR_GATE,
        "overlap_gate": OVERLAP_GATE,
        "overlap_radius_px": OVERLAP_RADIUS_PX,
    }
