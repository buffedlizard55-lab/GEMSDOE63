"""Official Distance-Weighted Tversky Index (DTI) for the DOE GEMS Prize Challenge.

PROVENANCE: adapted verbatim (bit-exact) from the shared template
buffedlizard55-lab/GEMSDOE32 src/gems32/metric.py (session 32, 2026-10-04), which
states: "Verified from official competition problem description (DrivenData Page
967): https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
and official organizer forum clarification on known-fault masking (Thread 11516):
https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516"

Mathematical formulation:
- Triangular kernel: k(d) = max(1 - d / R, 0), with R = 300 m = 3 pixels at 100 m.
- TP_w = sum_{g in G} max_{x : d(x,g) <= R} p(x) * k(d(x,g))
- FP_w = sum_{x : p(x) > 0} p(x) * [1 - max_{g in G} k(d(x,g))]
- FN_w = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))] = |G| - TP_w
- DTI(alpha=0.2, beta=0.8) = TP_w / (TP_w + 0.2 * FP_w + 0.8 * FN_w + eps)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

ALPHA: float = 0.2
BETA: float = 0.8
RADIUS_PX: float = 3.0
EPS: float = 1e-12


def kernel(d, radius: float = RADIUS_PX):
    """k(d) = max(1 - d / R, 0). d and radius in pixels (1 px = 100 m)."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


def _offsets(radius: float = RADIUS_PX):
    r = int(np.ceil(radius))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = float(kernel(np.hypot(dy, dx), radius))
            if k > 0.0:
                out.append((dy, dx, k))
    return out


_OFFS = _offsets()


def dti_from_counts(tp: float, fp: float, n_truth: int, alpha: float = ALPHA, beta: float = BETA) -> float:
    """Pooled DTI from pooled weighted counts (exactly as the shared template)."""
    fn = float(n_truth) - tp
    return float(tp / (tp + alpha * fp + beta * fn + EPS))


def dti_binary(pred_bool: np.ndarray, truth_bool: np.ndarray, active: np.ndarray | None = None) -> dict:
    """Exact DTI for binary predictions using Euclidean distance transforms."""
    p = np.asarray(pred_bool, bool)
    g = np.asarray(truth_bool, bool)
    if active is not None:
        p = p & active
        g = g & active
    n_truth = int(g.sum())
    n_pred = int(p.sum())
    if n_truth == 0:
        return {"tp": 0.0, "fp": float(n_pred), "fn": 0.0, "dti": 0.0, "n_truth": 0, "n_pred": n_pred}
    if n_pred == 0:
        return {"tp": 0.0, "fp": 0.0, "fn": float(n_truth), "dti": 0.0, "n_truth": n_truth, "n_pred": 0}

    dp = distance_transform_edt(~p)
    k_dp = kernel(dp[g])
    tp = float(k_dp.sum())
    fn = float(n_truth) - tp

    dg = distance_transform_edt(~g)
    k_dg = kernel(dg[p])
    fp = float((1.0 - k_dg).sum())

    dti = tp / (tp + ALPHA * fp + BETA * fn + EPS)
    return {"tp": tp, "fp": fp, "fn": fn, "dti": float(dti), "n_truth": n_truth, "n_pred": n_pred}


def cvar_penalized_score(scores: list[float] | np.ndarray, worst_quantile: float = 0.25, penalty_weight: float = 0.5) -> dict:
    """Compute mean DTI, worst-tail CVaR, and tail-penalized CVaR score.
    Reference: Rockafellar and Uryasev (Journal of Risk, 2000).
    """
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(arr)
    k = max(1, int(round(n * worst_quantile)))
    worst_tail = float(arr[:k].mean())
    mean_score = float(arr.mean())
    penalized = mean_score - penalty_weight * (mean_score - worst_tail)
    return {
        "mean_dti": mean_score,
        "worst_tail_cvar": worst_tail,
        "penalized_cvar": penalized,
        "min": float(arr[0]),
        "max": float(arr[-1]),
        "scores": [float(x) for x in arr],
    }
