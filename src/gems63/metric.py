"""Distance-weighted Tversky index (DTI) exactly as defined by the DOE GEMS competition.

Source: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
Formula:
  DTI = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)
  with alpha = 0.2, beta = 0.8, triangular kernel radius = 300 m (3 px at 100 m).
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt

ALPHA = 0.2
BETA = 0.8
RADIUS_PX = 3.0
EPS = 1e-12


def kernel(d: np.ndarray, radius: float = RADIUS_PX) -> np.ndarray:
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


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
