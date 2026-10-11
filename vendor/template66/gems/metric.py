"""Distance-weighted Tversky index (DTI), as published by the competition.

Official definition (problem page, "Performance metric"):

    TP_w = sum_{g in G} max_{x: d(x,g) <= R} p(x) k(d(x,g))
    FP_w = sum_{x: p(x) > 0} p(x) [1 - max_{g in G} k(d(x,g))]
    FN_w = sum_{g in G} [1 - max_{x: d(x,g) <= R} p(x) k(d(x,g))]
    DTI  = TP_w / (TP_w + alpha FP_w + beta FN_w + eps)
    k(d) = max(1 - d / R, 0),  R = 300 m,  alpha = 0.2,  beta = 0.8

Interpretation choices, NOT stated on the page (see docs/research/metric-derivation.md):
  * d is the Euclidean distance between pixel centres, in metres (square 100 m cells);
  * NaN predictions are treated as 0 (no emission); values outside [0, 1] are rejected;
  * ground-truth positives are label > 0 (reference notebook convention), everything else is negative;
  * eps = 1e-9 (the page does not publish eps).

Identity (exact under the definitions above, tested):
    TP_w + FN_w = |G|   and   DTI = TP_w / (0.2 (TP_w + S - M) + 0.8 |G|)
    with S = sum_x p(x) and M = sum_x p(x) max_g k(d(x,g)).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .constants import ALPHA, BETA, KERNEL_RANGE_M, PIXEL_M


def kernel(distance_m, range_m: float = KERNEL_RANGE_M):
    """Triangular kernel k(d) = max(1 - d/R, 0)."""
    return np.clip(1.0 - np.asarray(distance_m, dtype=float) / range_m, 0.0, None)


def _clean_prediction(pred) -> np.ndarray:
    p = np.array(pred, dtype=float, copy=True)
    p[np.isnan(p)] = 0.0
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("predictions must lie in [0, 1] (NaN is treated as 0)")
    return p


def tversky_terms(pred, gt, pixel_m: float = PIXEL_M, range_m: float = KERNEL_RANGE_M):
    """Return (TP_w, FP_w, FN_w) for a probability raster and a binary truth raster."""
    p = _clean_prediction(pred)
    g = np.asarray(gt) > 0  # fault cell: label > 0, as in the reference notebook (y > 0); nodata -1 is excluded
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    n_g = int(g.sum())

    # M(y) = max over offsets o with |o| < R of p(y + o) * k(|o|).  Then TP_w = sum_{g} M(g).
    r_px = int(np.floor(range_m / pixel_m))
    h, w = p.shape
    padded = np.pad(p, r_px, mode="constant", constant_values=0.0)
    m_field = np.zeros_like(p)
    for dy in range(-r_px, r_px + 1):
        for dx in range(-r_px, r_px + 1):
            k = float(kernel(pixel_m * np.hypot(dy, dx), range_m))
            if k <= 0.0:
                continue
            shifted = padded[r_px + dy : r_px + dy + h, r_px + dx : r_px + dx + w]
            np.maximum(m_field, shifted * k, out=m_field)
    tp = float(m_field[g].sum())
    fn = float(n_g - tp)  # equals sum_g (1 - M(g)), since M(g) <= 1

    # FP_w = sum_x p(x) (1 - max_g k(d(x, g))).  Nearest-truth distance via EDT.
    if n_g == 0:
        k_max = np.zeros_like(p)
    else:
        dist_m = ndimage.distance_transform_edt(~g) * pixel_m
        k_max = kernel(dist_m, range_m)
    fp = float(np.sum(p * (1.0 - k_max)))
    return tp, fp, fn


def dti_from_terms(tp: float, fp: float, fn: float, alpha: float = ALPHA,
                   beta: float = BETA, eps: float = 1e-9) -> float:
    return tp / (tp + alpha * fp + beta * fn + eps)


def dti(pred, gt, alpha: float = ALPHA, beta: float = BETA, eps: float = 1e-9) -> float:
    """Distance-weighted Tversky index of a probability raster against a binary truth raster."""
    tp, fp, fn = tversky_terms(pred, gt)
    return dti_from_terms(tp, fp, fn, alpha, beta, eps)


def marginal_credit_bar(dti_value: float, alpha: float = ALPHA) -> float:
    """Break-even kernel credit for one extra dot (derived in docs/research/metric-derivation.md).

    Adding a dot that is the nearest emission to a previously uncovered truth pixel, with
    kernel credit k, raises DTI iff k > alpha * DTI. Tested numerically in tests/test_metric.py.
    """
    return alpha * dti_value


def tversky_terms_bruteforce(pred, gt, pixel_m: float = PIXEL_M, range_m: float = KERNEL_RANGE_M):
    """Literal O(N*M) transcription of the published formulas. For tests only."""
    p = _clean_prediction(pred)
    g = np.asarray(gt) > 0  # fault cell: label > 0, as in the reference notebook (y > 0); nodata -1 is excluded
    gs = [tuple(x) for x in np.argwhere(g)]
    h, w = p.shape
    cells = [(i, j) for i in range(h) for j in range(w)]

    def dist(a, b):
        return pixel_m * float(np.hypot(a[0] - b[0], a[1] - b[1]))

    tp = 0.0
    for gg in gs:
        best = 0.0
        for x in cells:
            d = dist(x, gg)
            if d <= range_m:
                best = max(best, p[x] * float(kernel(d, range_m)))
        tp += best
    fn = 0.0
    for gg in gs:
        best = 0.0
        for x in cells:
            d = dist(x, gg)
            if d <= range_m:
                best = max(best, p[x] * float(kernel(d, range_m)))
        fn += 1.0 - best
    fp = 0.0
    for x in cells:
        if p[x] > 0:
            kmax = max((float(kernel(dist(x, gg), range_m)) for gg in gs), default=0.0)
            fp += p[x] * (1.0 - kmax)
    return tp, fp, fn
