"""Distance-weighted Tversky index (DTI) for the DOE GEMS Prize, exact for binary dots.

Official definition (DrivenData problem page, "Performance metric"):
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  p(x) in [0,1] predicted probability, g(x) ground truth, triangular kernel
  k(d) = max(1 - d/R, 0) with R = 300 m = 3 px at 100 m.
  TP_w = sum_{g in G} max_{x: d(x,g)<=R} p(x) k(d(x,g))
  FP_w = sum_{x: p(x)>0} p(x) [1 - max_{g in G} k(d(x,g))]
  FN_w = |G| - TP_w
  DTI  = TP_w / (TP_w + 0.2 FP_w + 0.8 FN_w + eps)
Known USGS/INGENIOUS fault pixels are masked (excluded from every term); organizer
confirmation: https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516

For BINARY predictions the max over x reduces to the nearest predicted dot, so the
exact computation needs only nearest-neighbour distances (cKDTree), which is what makes
the label-world ensemble tractable. `dti_bruteforce` is the literal O(|G||P|) transcription
used to verify this module in tests/test_dti.py.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

ALPHA = 0.2
BETA = 0.8
R_PX = 3.0  # 300 m at 100 m pixels
EPS = 1e-12


def kernel(d):
    """Triangular kernel k(d) = max(1 - d/R, 0); d in pixels."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / R_PX, 0.0)


def dti_from_components(tp: float, fp: float, fn: float) -> float:
    return float(tp / (tp + ALPHA * fp + BETA * fn + EPS))


def components_binary(dots_rc: np.ndarray, truth_rc: np.ndarray) -> dict:
    """Exact TP_w, FP_w, FN_w, |G| for binary dots and truth (both already restricted to the
    scored domain, i.e. valid AND NOT known). Coordinates are (row, col) integer pixels."""
    n_g = int(truth_rc.shape[0])
    if n_g == 0:
        return dict(tp=0.0, fp=float(dots_rc.shape[0]), fn=0.0, n_truth=0)
    if dots_rc.shape[0] == 0:
        return dict(tp=0.0, fp=0.0, fn=float(n_g), n_truth=n_g)
    tree_p = cKDTree(dots_rc.astype(np.float64))
    tree_g = cKDTree(truth_rc.astype(np.float64))
    d_gp, _ = tree_p.query(truth_rc.astype(np.float64), k=1, distance_upper_bound=R_PX + 1e-9)
    d_gp = np.where(np.isfinite(d_gp), d_gp, np.inf)
    tp = float(kernel(d_gp).sum())
    d_pg, _ = tree_g.query(dots_rc.astype(np.float64), k=1, distance_upper_bound=R_PX + 1e-9)
    d_pg = np.where(np.isfinite(d_pg), d_pg, np.inf)
    fp = float((1.0 - kernel(d_pg)).sum())
    fn = float(n_g) - tp
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n_g)


def dti_binary_pixels(dots_rc, truth_rc) -> dict:
    c = components_binary(np.asarray(dots_rc), np.asarray(truth_rc))
    c["dti"] = dti_from_components(c["tp"], c["fp"], c["fn"])
    c["coverage"] = c["tp"] / c["n_truth"] if c["n_truth"] else 0.0
    return c


def dti_bruteforce(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Literal transcription of the published equations (test oracle; slow)."""
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, bool)
    gs = np.argwhere(truth)
    xs = np.argwhere(pred > 0)
    tp = fn = 0.0
    for g in gs:
        best = 0.0
        for x in xs:
            d = float(np.hypot(*(x - g)))
            if d <= R_PX:
                best = max(best, pred[tuple(x)] * max(1.0 - d / R_PX, 0.0))
        tp += best
        fn += 1.0 - best
    fp = 0.0
    for x in xs:
        kmax = 0.0
        for g in gs:
            kmax = max(kmax, max(1.0 - float(np.hypot(*(x - g))) / R_PX, 0.0))
        fp += pred[tuple(x)] * (1.0 - kmax)
    return dict(tp=tp, fp=fp, fn=fn, dti=dti_from_components(tp, fp, fn))


def cvar_lower(values, alpha: float = 0.2) -> float:
    """Lower-tail CVaR: mean of the worst (lowest) ceil(alpha*n) values.
    Rockafellar & Uryasev (2000), Journal of Risk 2(3):21-42 (Springer reference:
    https://link.springer.com/chapter/10.1007/978-1-4757-6594-6_17)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, int(np.ceil(alpha * v.size)))
    return float(v[:k].mean())
