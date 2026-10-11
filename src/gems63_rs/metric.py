"""Distance-weighted Tversky index (DTI) — exact transcription of the published metric.

Source of truth, read line by line on 2026-10-11 from
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ :

    k(d)  = max(1 - d/R, 0),  R = 300 m  (= 3 px at the competition's 100 m grid)

    TP_w = sum_{g in G}   max_{x : d(x,g) <= R}  p(x) * k(d(x,g))
    FP_w = sum_{x : p(x)>0}  p(x) * [ 1 - max_{g in G} k(d(x,g)) ]
    FN_w = sum_{g in G}   [ 1 - max_{x : d(x,g) <= R}  p(x) * k(d(x,g)) ]

    DTI(alpha, beta) = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps)
    alpha = 0.2, beta = 0.8

Published worked example (same page): a single vertical ground-truth line gives
TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 and TI_w(0.2, 0.8) = 3.00 / (3.00 + 0.2*1.89
+ 0.8*2.00) = 0.60.  ``tests/test_metric.py`` asserts that example.

Two identities that follow from the formulas and drive the whole emission design;
both are re-derived independently in ``docs/metric-notes.md``:

1.  FN_w = |G| - TP_w, because every ground-truth pixel contributes
    max + (1 - max) = 1 across the TP and FN terms.  Hence with N = |G|

        DTI = TP_w / ( 0.2*(TP_w + FP_w) + 0.8*N )

2.  Adding emission mass p at a pixel whose realised kernel weight is k moves
    TP_w -> TP_w + k*p and FP_w -> FP_w + (1-k)*p, i.e. the denominator by
    exactly 0.2*p, so the score rises iff  k > alpha * DTI.  (``marginal_bar``.)

Masking: organizer chrisk-dd (DrivenData Staff), forum thread 11516, 2026-09-16 —
"Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from
evaluation, so they do not count towards penalty terms" and "Re-evaluation will
also mask/exclude the existing USGS/INGENIOUS faults."  ``valid`` implements that
mask: masked pixels are removed from *both* the prediction and the truth sums.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

ALPHA = 0.2
BETA = 0.8
RADIUS_M = 300.0
PIXEL_M = 100.0
RADIUS_PX = RADIUS_M / PIXEL_M  # 3.0
EPS = 1e-12


def kernel(d, radius: float = RADIUS_PX) -> np.ndarray:
    """Triangular kernel k(d) = max(1 - d/R, 0)."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


def offsets_within(radius: float = RADIUS_PX) -> list[tuple[int, int, float]]:
    """All integer pixel offsets with Euclidean norm <= radius, plus their k value.

    For R = 3 px this is the 29-pixel disc used by the competition metric.
    """
    r = int(np.floor(radius))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = float(np.hypot(dy, dx))
            if d <= radius + 1e-9:
                out.append((dy, dx, d))
    return out


def _max_kernel_filter(p: np.ndarray, radius: float = RADIUS_PX) -> np.ndarray:
    """For every pixel g: max over x with d(x,g) <= R of p(x) * k(d(x,g)).

    Exact for arbitrary float p (not just binary).  O(|offsets| * N).
    """
    best = np.zeros(p.shape, dtype=np.float64)
    h, w = p.shape
    for dy, dx, d in offsets_within(radius):
        k = float(kernel(d, radius))
        if k <= 0.0:
            continue
        shifted = np.zeros_like(best)
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        shifted[ys0:ys1, xs0:xs1] = p[max(0, -dy):min(h, h - dy),
                                      max(0, -dx):min(w, w - dx)]
        np.maximum(best, shifted * k, out=best)
    return best


@dataclass
class DTIComponents:
    """The three published terms plus the resulting index."""

    tp_w: float
    fp_w: float
    fn_w: float
    n_truth: int
    n_emitted: int
    emitted_mass: float

    @property
    def dti(self) -> float:
        return dti_from_components(self.tp_w, self.fp_w, self.fn_w)


def dti_from_components(tp: float, fp: float, fn: float,
                        alpha: float = ALPHA, beta: float = BETA) -> float:
    """The published closed form. Checked against the worked example in the tests."""
    return float(tp) / (float(tp) + alpha * float(fp) + beta * float(fn) + EPS)


def dti_components(prediction: np.ndarray,
                   truth: np.ndarray,
                   valid: np.ndarray | None = None,
                   alpha: float = ALPHA,
                   beta: float = BETA,
                   radius: float = RADIUS_PX) -> DTIComponents:
    """Exact distance-weighted Tversky components.

    Parameters
    ----------
    prediction : float array in [0, 1]
        Predicted fault probability p(x). Anything > 0 counts as emission.
    truth : bool array
        Ground-truth fault pixels G.
    valid : bool array or None
        The scored domain. Pixels outside it are removed from *every* sum, which
        is the organizer's masking rule for known USGS/INGENIOUS fault pixels.
    """
    p = np.asarray(prediction, dtype=np.float64)
    g = np.asarray(truth, dtype=bool)
    if valid is None:
        valid = np.ones(p.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)

    p = np.where(valid, p, 0.0)
    g = g & valid
    n_truth = int(g.sum())
    emit = p > 0.0
    n_emitted = int(emit.sum())
    emitted_mass = float(p[emit].sum()) if n_emitted else 0.0

    if n_truth == 0:
        # No truth: TP_w = FN_w = 0, so DTI = 0/(0 + 0.2*FP_w) = 0.
        fp = float(p[emit].sum()) if n_emitted else 0.0
        return DTIComponents(0.0, fp, 0.0, 0, n_emitted, emitted_mass)

    # TP_w and FN_w: max over predicted pixels within R, per ground-truth pixel.
    tpw_per_truth = _max_kernel_filter(p, radius)[g]
    tp = float(tpw_per_truth.sum())
    fn = float(n_truth) - tp

    # FP_w: kmax(x) = max over truth of k(d(x,g)); truth is binary so an exact
    # Euclidean distance transform gives d(x, G) directly.
    dg = distance_transform_edt(~g)
    fp = float((p[emit] * (1.0 - kernel(dg[emit], radius))).sum())

    return DTIComponents(tp, fp, fn, n_truth, n_emitted, emitted_mass)


def dti(prediction, truth, valid=None, **kw) -> float:
    """Convenience: the scalar index only."""
    return dti_components(prediction, truth, valid, **kw).dti


def binary_dti_components(emit: np.ndarray,
                          truth: np.ndarray,
                          valid: np.ndarray | None = None,
                          *,
                          edt_emit: np.ndarray | None = None,
                          alpha: float = ALPHA,
                          beta: float = BETA,
                          radius: float = RADIUS_PX) -> DTIComponents:
    """Exact DTI for **binary** emission, in two distance transforms.

    For p in {0, 1} the per-truth maximum collapses:

        max_{x : d(x,g) <= R} p(x) k(d(x,g)) = k( d(g, nearest emitted pixel) )

    because the kernel is monotone decreasing in d, so the largest term is the one
    from the closest emitted pixel.  That turns the 29-shift max-filter into a single
    Euclidean distance transform, which is what makes a several-hundred-world
    ensemble tractable.  ``edt_emit`` may be supplied precomputed so a candidate is
    transformed once and scored against every world.

    ``tests/test_metric.py`` asserts this equals the general float path bit-for-bit
    on random binary inputs.
    """
    e = np.asarray(emit, dtype=bool)
    g = np.asarray(truth, dtype=bool)
    if valid is None:
        valid = np.ones(e.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    e = e & valid
    g = g & valid
    n_truth = int(g.sum())
    n_emitted = int(e.sum())
    if n_truth == 0:
        return DTIComponents(0.0, float(n_emitted), 0.0, 0, n_emitted, float(n_emitted))
    if edt_emit is None:
        edt_emit = distance_transform_edt(~e)
    tp = float(kernel(edt_emit[g], radius).sum())
    fn = float(n_truth) - tp
    dg = distance_transform_edt(~g)
    fp = float((1.0 - kernel(dg[e], radius)).sum())
    return DTIComponents(tp, fp, fn, n_truth, n_emitted, float(n_emitted))


def dti_bruteforce(prediction, truth, valid=None, alpha=ALPHA, beta=BETA,
                   radius=RADIUS_PX) -> DTIComponents:
    """Literal O(|G|*|P|) transcription of the published equations. Tests only."""
    p = np.asarray(prediction, dtype=np.float64)
    g = np.asarray(truth, dtype=bool)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        p = np.where(valid, p, 0.0)
        g = g & valid
    gs = np.argwhere(g)
    xs = np.argwhere(p > 0)
    tp = fn = 0.0
    for gy, gx in gs:
        best = 0.0
        for xy, xx in xs:
            d = float(np.hypot(xy - gy, xx - gx))
            if d <= radius:
                best = max(best, float(p[xy, xx]) * max(1.0 - d / radius, 0.0))
        tp += best
        fn += 1.0 - best
    fp = 0.0
    for xy, xx in xs:
        kmax = 0.0
        for gy, gx in gs:
            kmax = max(kmax, max(1.0 - float(np.hypot(xy - gy, xx - gx)) / radius, 0.0))
        fp += float(p[xy, xx]) * (1.0 - kmax)
    n_truth = int(g.sum())
    return DTIComponents(tp, fp, fn, n_truth, int(len(xs)), float(p[xs[:, 0], xs[:, 1]].sum()))


def marginal_bar(current_dti: float, alpha: float = ALPHA) -> float:
    """Minimum realised kernel weight k for added emission to raise the score.

    Derivation: with T = TP_w, S = FP_w, D = 0.2*(T + S) + 0.8*N, adding mass p
    with realised weight k gives T' = T + k*p and D' = D + 0.2*p, so

        DTI' > DTI  <=>  (T + k p)(D) > T(D + 0.2 p)  <=>  k > 0.2*T/D = alpha*DTI.
    """
    return alpha * current_dti


def expected_kernel_weight(pi: np.ndarray, radius: float = RADIUS_PX) -> np.ndarray:
    """E[ max_{g in truth, d<=R} k(d(x,g)) ] for a per-pixel truth probability field.

    The truth pixels within R of x are modelled as independent Bernoulli(pi).  Let
    the distinct kernel values be v_1 > v_2 > ... and S_j = prod_{i <= j}(1 - pi_i),
    the probability that no offset with value >= v_j is a truth pixel (S_0 = 1).
    Then max = v_j exactly when S_{j-1} holds but S_j does not, so

        E[max] = sum_j v_j * (S_{j-1} - S_j).

    Verified against Monte-Carlo in ``tests/test_metric.py``.  This is an
    *independence* approximation: real fault traces are spatially correlated, so it
    is calibrated against simulated worlds in ``scripts/run_lane.py`` rather than
    trusted analytically.
    """
    pi = np.clip(np.asarray(pi, dtype=np.float64), 0.0, 1.0)
    offs = [(dy, dx, d) for dy, dx, d in offsets_within(radius)]
    # Group offsets by distinct kernel value.
    groups: dict[float, list[tuple[int, int]]] = {}
    for dy, dx, d in offs:
        kv = round(float(kernel(d, radius)), 10)
        groups.setdefault(kv, []).append((dy, dx))
    h, w = pi.shape
    vals = sorted(groups.keys(), reverse=True)
    surv = np.ones(pi.shape, dtype=np.float64)   # S_{j-1}: no truth at value >= v_j
    out = np.zeros(pi.shape, dtype=np.float64)
    for t in vals:
        if t <= 0.0:
            break                                 # k = 0 contributes nothing
        nxt = surv.copy()
        for dy, dx in groups[t]:
            sh = np.zeros_like(pi)
            ys0, ys1 = max(0, dy), min(h, h + dy)
            xs0, xs1 = max(0, dx), min(w, w + dx)
            sh[ys0:ys1, xs0:xs1] = pi[max(0, -dy):min(h, h - dy),
                                      max(0, -dx):min(w, w - dx)]
            nxt *= (1.0 - sh)
        out += t * (surv - nxt)                   # v_j * (S_{j-1} - S_j)
        surv = nxt
    return out
