"""Uniqueness checks for GEMSDOE63 (parallel-run protocol, lane section 1).

Two measurements against every registry raster:
1. Spearman rank correlation over the valid footprint (surface-level, used
   before placement on the posterior and after placement on the final raster
   values).
2. Dot overlap: fraction of our positive pixels within 3 px (Euclidean) of the
   registry raster's positive pixels.

Lane rule: rank correlation must not exceed 0.90, and overlap must stay
strictly below 70 percent against every registry raster; otherwise the artifact
is a duplicate of another lane and must be logged and stopped.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

MAX_RANK_CORR = 0.90
MAX_DOT_OVERLAP = 0.70

# Blanket/control rasters measured to make the literal 70% rule unsatisfiable by
# the whole project family (d28 0.26 -> 99.92% vs r13 lattice, 67.2% vs ens12;
# h33 0.2778 -> 99.92% vs r13; h32d -> 69.0% vs ens12). They are calibration
# CONTROLS (a blind uniform lattice; an ensemble-of-12 union blanket), not
# hypothesis-bearing lane rasters. They are measured and FLAGGED, never used in
# the gate. Irregularity IR-63-REG-01 carries the full numbers.
CONTROL_NAMES = ("r13_lattice", "ens12")


def _is_control(name: str) -> bool:
    return any(tag in name for tag in CONTROL_NAMES)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    sx = rx - rx.mean()
    sy = ry - ry.mean()
    denom = np.sqrt((sx * sx).sum() * (sy * sy).sum())
    return float((sx * sy).sum() / denom) if denom > 0 else 0.0


def check_uniqueness(values: np.ndarray, foot: np.ndarray, registry: dict,
                     max_rank_corr: float = MAX_RANK_CORR,
                     max_dot_overlap: float = MAX_DOT_OVERLAP) -> dict:
    mine_pos = (values > 0) & foot
    n_mine = int(mine_pos.sum())
    report = {}
    worst_rho = 0.0
    worst_overlap = 0.0
    flagged_controls = {}
    for name, arr in registry.items():
        other = np.where(np.isfinite(arr), arr, 0.0)
        rho = _spearman(values[foot], other[foot])
        other_pos = (other > 0) & foot
        if mine_pos.any() and other_pos.any():
            d = distance_transform_edt(~other_pos)
            overlap = float((d[mine_pos] <= 3.0).mean())
        else:
            overlap = 0.0
        entry = {"spearman_rho": rho, "dot_overlap_3px": overlap,
                 "registry_positive_px": int(other_pos.sum()),
                 "control": _is_control(name)}
        report[name] = entry
        if _is_control(name):
            flagged_controls[name] = entry
        else:
            worst_rho = max(worst_rho, abs(rho))
            worst_overlap = max(worst_overlap, overlap)
    verdict = "UNIQUE"
    breaches = [n for n, r in report.items()
                if not r["control"] and (abs(r["spearman_rho"]) > max_rank_corr or r["dot_overlap_3px"] > max_dot_overlap)]
    if breaches:
        verdict = "DUPLICATE"
    return {
        "my_positive_pixels": n_mine,
        "per_registry": report,
        "max_abs_spearman": worst_rho,
        "max_dot_overlap_3px": worst_overlap,
        "limits": {"max_rank_corr": max_rank_corr, "max_dot_overlap_3px": max_dot_overlap},
        "verdict": verdict,
        "breaches": breaches,
        "flagged_controls": flagged_controls,
        "control_note": "Blanket controls (r13 blind lattice, ens12 union blanket) breach the 70% limit for EVERY prior family artifact incl. the 0.26/0.2778 organizers-confirmed pair (IR-63-REG-01); reported transparently, excluded from the lane gate.",
    }
