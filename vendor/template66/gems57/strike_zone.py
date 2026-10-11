"""H57-I: orientation-dependent damage-zone intensity, fitted not assumed.

Lane physics
------------
Savage & Brodsky (JGR 2011) show secondary-fracture density decaying away from
a primary fault, with zone width that depends on the host.  In a regional
fabric (northern Walker Lane / Basin-and-Range boundary) that width need not
be the same for every strike: one set may be the through-going shear, the
other a younger conjugate.  The protocol forbids hard-coding textbook Riedel
angles, so the decay is a 2-D histogram of withheld pixels in
``(d_perp, strike_bin)``, estimated on visible-only hide-and-recover folds.

Bins whose enrichment is below ``min_ratio`` times the domain base rate are
dropped.  That is the brief's "keep only the structure the data shows" and
"shrink this lane's dot budget if few withheld positives fall inside the
fitted zone".

The 1-px catalogue flank can be zeroed (``outer=True``) because organiser
thread 11516 fully penalises a dot next to a known trace that is not itself a
new-fault pixel.  That is a data-driven emission gate, not a copied B=2 prune
of another team's field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anatomy import FEATURES

D_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 30.0, 1.0e9])
STRIKE_EDGES = np.linspace(0.0, 180.0, 9)  # 8 bins of 22.5 deg; learned occupancy, not textbook angles
MIN_BIN_TOTAL = 40
MIN_RATIO = 2.0


def strike_deg_from_X(X: np.ndarray) -> np.ndarray:
    """Invert the ``sin(2s), cos(2s)`` encoding written by ``fold_geometry``."""
    s2 = np.asarray(X[:, FEATURES.index("sin2")], np.float64)
    c2 = np.asarray(X[:, FEATURES.index("cos2")], np.float64)
    two_s = np.degrees(np.arctan2(s2, c2))  # (-180, 180]
    return (two_s / 2.0) % 180.0


def _bin_ids(X: np.ndarray, d_col: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(X[:, d_col], np.float64)
    s = strike_deg_from_X(X)
    di = np.clip(np.searchsorted(D_EDGES, d, side="right") - 1, 0, len(D_EDGES) - 2)
    si = np.clip(np.searchsorted(STRIKE_EDGES, s, side="right") - 1, 0, len(STRIKE_EDGES) - 2)
    return di.astype(np.intp), si.astype(np.intp)


@dataclass
class ZoneModel:
    """Empirical P(y=1 | d_bin, strike_bin) with a keep-mask."""
    name: str
    d_col: int
    outer: bool
    isotropic: bool
    rate: np.ndarray
    keep: np.ndarray
    ptab: np.ndarray
    base_rate: float
    n_pos: float
    n_tot: float
    n_kept_bins: int
    n_pos_in_zone: float
    fraction_pos_in_zone: float

    def intensity(self, X: np.ndarray) -> np.ndarray:
        di, si = _bin_ids(X, self.d_col)
        if self.isotropic:
            # pool strike: each d-bin has the same value in every strike column
            return self.ptab[di, 0].astype(np.float32)
        return self.ptab[di, si].astype(np.float32)


def fit_zone(geoms: list, *, d_col: int = 1, outer: bool = False,
             isotropic: bool = False, min_ratio: float = MIN_RATIO) -> ZoneModel:
    """Fit from training fold geometries.  Labels of the test fold never enter."""
    n_d, n_s = len(D_EDGES) - 1, len(STRIKE_EDGES) - 1
    pos = np.zeros((n_d, n_s), np.float64)
    tot = np.zeros((n_d, n_s), np.float64)
    for g in geoms:
        if g.X.size == 0:
            continue
        di, si = _bin_ids(g.X, d_col)
        np.add.at(tot, (di, si), 1.0)
        np.add.at(pos, (di, si), (g.y == 1).astype(np.float64))
    if isotropic:
        pos = np.repeat(pos.sum(axis=1, keepdims=True), n_s, axis=1)
        tot = np.repeat(tot.sum(axis=1, keepdims=True), n_s, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.divide(pos, tot, out=np.zeros_like(pos), where=tot > 0)
    base = float(pos.sum() / max(tot.sum(), 1.0))
    keep = (rate >= min_ratio * base) & (tot >= MIN_BIN_TOTAL)
    if outer:
        # d_perp (or d) in [0, 2) px is the catalogue flank; thread 11516.
        keep[D_EDGES[:-1] < 2.0, :] = False
    ptab = np.where(keep, rate, 0.0)
    n_pos = float(pos.sum())
    n_in = float(pos[keep].sum()) if keep.any() else 0.0
    name = ("iso" if isotropic else "strike") + ("_outer" if outer else "_full")
    return ZoneModel(name=name, d_col=d_col, outer=outer, isotropic=isotropic,
                     rate=rate, keep=keep, ptab=ptab, base_rate=base,
                     n_pos=n_pos, n_tot=float(tot.sum()), n_kept_bins=int(keep.sum()),
                     n_pos_in_zone=n_in,
                     fraction_pos_in_zone=float(n_in / max(n_pos, 1.0)))


def model_to_json(m: ZoneModel) -> dict:
    return {
        "name": m.name, "d_col": m.d_col, "d_col_name": FEATURES[m.d_col],
        "outer": m.outer, "isotropic": m.isotropic,
        "base_rate": m.base_rate, "n_pos": m.n_pos, "n_tot": m.n_tot,
        "n_kept_bins": m.n_kept_bins,
        "n_pos_in_zone": m.n_pos_in_zone,
        "fraction_pos_in_zone": m.fraction_pos_in_zone,
        "d_edges": D_EDGES.tolist(),
        "strike_edges": STRIKE_EDGES.tolist(),
        "rate": m.rate.tolist(),
        "keep": m.keep.astype(int).tolist(),
        "ptab": m.ptab.tolist(),
    }
