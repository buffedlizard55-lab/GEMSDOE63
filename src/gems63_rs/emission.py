"""Candidate emission policies: how probability mass is laid on the grid.

The metric's own arithmetic fixes most of the design (derived in
``docs/metric-notes.md`` and asserted in ``tests/test_metric.py``):

    DTI = TP_w / ( 0.2*(TP_w + FP_w) + 0.8*N )

* a dot that lands *on* a truth pixel pays **no** false-positive penalty, because
  FP_w carries the factor (1 - kmax) and kmax = 1 there;
* the score rises when emission mass with realised kernel weight k is added iff
  k > 0.2 * DTI (``metric.marginal_bar``);
* that condition does not depend on the emitted value p, and the interior optimum
  in p exceeds 1 for every k above the bar, so **binary emission at 1.0 is optimal**.

What is left to choose is therefore purely geometric: *which* pixels, how wide a
ribbon, how far apart, and how many.  Those are the policies below, and they are the
objects the CVaR selector in :mod:`gems63.cvar` chooses between.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .metric import RADIUS_PX, expected_kernel_weight, marginal_bar

STRUCT8 = ndi.generate_binary_structure(2, 2)


def ribbon(source: np.ndarray, half_width_px: float,
           *, exclude_source: bool = True,
           allowed: np.ndarray | None = None) -> np.ndarray:
    """The annulus {0 < d(x, source) <= half_width_px} around a trace network.

    ``exclude_source`` removes the source pixels themselves.  For the catalogue that
    is the right thing to do: known USGS/INGENIOUS fault pixels are masked out of
    evaluation (thread 11516), so emission there can earn nothing.
    """
    src = np.asarray(source, dtype=bool)
    d = ndi.distance_transform_edt(~src)
    out = d <= half_width_px
    if exclude_source:
        out &= ~src
    if allowed is not None:
        out &= np.asarray(allowed, dtype=bool)
    return out


def distance_band(source: np.ndarray, lo_px: float, hi_px: float,
                  *, allowed: np.ndarray | None = None) -> np.ndarray:
    """{lo < d(x, source) <= hi} -- lets the selector reject the innermost ring."""
    d = ndi.distance_transform_edt(~np.asarray(source, dtype=bool))
    out = (d > lo_px) & (d <= hi_px)
    if allowed is not None:
        out &= np.asarray(allowed, dtype=bool)
    return out


def _disc_footprint(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return np.hypot(y, x) <= radius_px + 1e-9


def nms_thin(mask: np.ndarray, priority: np.ndarray, min_sep_px: float) -> np.ndarray:
    """Greedy spacing control by non-maximum suppression on `priority`.

    Keeps the pixels that are the local maximum of `priority` within a disc of
    radius ``min_sep_px``, so retained dots are at least ~``min_sep_px`` apart.
    Vectorised: one maximum filter, no Python loop over candidates, and cropped to
    the mask's bounding box because a maximum filter over the full 3730x3292 grid
    costs seconds per call and this is called once per candidate.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m.copy()
    rr, cc = np.nonzero(m)
    r = int(np.ceil(min_sep_px))
    sl = (slice(max(0, rr.min() - r), min(m.shape[0], rr.max() + r + 1)),
          slice(max(0, cc.min() - r), min(m.shape[1], cc.max() + r + 1)))
    sub = m[sl]
    p = np.where(sub, np.asarray(priority, dtype=np.float64)[sl], -np.inf)
    fp = _disc_footprint(min_sep_px)
    localmax = ndi.maximum_filter(p, footprint=fp, mode="constant", cval=-np.inf)
    out = np.zeros_like(m)
    out[sl] = sub & (p >= localmax - 1e-12)
    return out


def budget_cap(mask: np.ndarray, priority: np.ndarray, k: int) -> np.ndarray:
    """Keep the top-`k` pixels of `mask` by `priority` (a hard emission budget)."""
    m = np.asarray(mask, dtype=bool)
    if k >= int(m.sum()):
        return m.copy()
    idx = np.flatnonzero(m.ravel())
    pr = np.asarray(priority, dtype=np.float64).ravel()[idx]
    if k <= 0:
        return np.zeros_like(m)
    take = idx[np.argpartition(-pr, k - 1)[:k]]
    out = np.zeros_like(m)
    out.ravel()[take] = True
    return out


def dti_optimal(pi: np.ndarray, dti_estimate: float,
                *, allowed: np.ndarray | None = None,
                radius_px: float = RADIUS_PX) -> np.ndarray:
    """Emit exactly where the expected kernel weight clears the marginal bar.

    ``metric.marginal_bar`` proves the score rises iff k > alpha * DTI, so the
    Bayes-optimal hard emission for this metric is {x : E[k(x)] > alpha * DTI}.
    ``pi`` is the per-pixel probability that the pixel is a *scoreable* truth pixel.
    """
    bar = marginal_bar(dti_estimate)
    ek = expected_kernel_weight(pi, radius_px)
    out = ek > bar
    if allowed is not None:
        out &= np.asarray(allowed, dtype=bool)
    return out


def endpoints_of_segments(cat: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each 8-connected segment: its tip pixels and its principal-axis direction.

    A tip is a catalogue pixel with exactly one 8-neighbour in the same segment --
    the natural place for a trace to continue past its mapped end.

    Implemented with one global neighbour count plus per-segment bounding boxes.  The
    earlier version ran ``binary_dilation`` and a 3x3 convolution over the *whole*
    3730x3292 grid once per segment, i.e. ~6400 full-array passes for the 3198 real
    catalogue segments, which made a single candidate build take minutes.
    """
    cat = np.asarray(cat, dtype=bool)
    lab, n = ndi.label(cat, structure=STRUCT8)
    if n == 0:
        return []
    # Count, for every pixel, its 8-neighbours that belong to the SAME segment.
    same = np.zeros(cat.shape, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.zeros_like(lab)
            h, w = lab.shape
            ys0, ys1 = max(0, dy), min(h, h + dy)
            xs0, xs1 = max(0, dx), min(w, w + dx)
            sh[ys0:ys1, xs0:xs1] = lab[max(0, -dy):min(h, h - dy),
                                       max(0, -dx):min(w, w - dx)]
            same += (sh == lab) & (lab > 0)
    tips_all = cat & (same == 1)

    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        if int(sub.sum()) < 3:
            continue
        tips = sub & tips_all[sl]
        if not tips.any():
            continue
        rr, cc = np.nonzero(sub)
        rc = np.column_stack([rr, cc]).astype(float)
        rc -= rc.mean(axis=0)
        cov = rc.T @ rc / max(1, len(rc))
        w, v = np.linalg.eigh(cov)
        axis = v[:, -1]                     # principal direction (row, col)
        tr, tc = np.nonzero(tips)
        out.append((np.column_stack([tr + sl[0].start, tc + sl[1].start]), axis))
    return out


def continuations(cat: np.ndarray, length_px: int,
                  *, allowed: np.ndarray | None = None) -> np.ndarray:
    """Extend every mapped trace past its tips along its own strike.

    Directly targets the population the organizers named: a "new fault" is "any fault
    pixel not already captured by USGS/INGENIOUS" and "can include newly mapped
    geometry of an existing fault system" (chrisk-dd, thread 11536, 2026-09-23).
    """
    cat = np.asarray(cat, dtype=bool)
    out = np.zeros(cat.shape, dtype=bool)
    for tips, axis in endpoints_of_segments(cat):
        for ty, tx in tips:
            rc = np.array([ty, tx], dtype=float)
            for sign in (1.0, -1.0):
                t = np.arange(1, length_px + 1)
                pts = rc + sign * t[:, None] * axis[None, :]
                r = np.round(pts[:, 0]).astype(int)
                c = np.round(pts[:, 1]).astype(int)
                ok = (r >= 0) & (r < cat.shape[0]) & (c >= 0) & (c < cat.shape[1])
                if ok.any():
                    out[r[ok], c[ok]] = True
    out &= ~cat
    if allowed is not None:
        out &= np.asarray(allowed, dtype=bool)
    return out


def policy_grid(half_widths=(1.0, 2.0, 3.0, 4.0, 5.0),
                separations=(1.0, 2.0, 3.0)) -> list[dict]:
    """The (ribbon width, spacing, budget) grid the CVaR selector searches."""
    out = []
    for w in half_widths:
        for s in separations:
            out.append({"kind": "ribbon", "half_width_px": float(w),
                        "min_sep_px": float(s), "budget": None})
    return out
