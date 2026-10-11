"""Emission policies for GEMSDOE63: posterior-ordered thinning and speckle filtering.

The d28 forensic autopsy found that the best-scoring emission family
(sparse unit-probability dots) was thinned in raw row-major order "without
sorting by geological posterior score, allowing lower-confidence endpoints to
suppress adjacent ridge crests". Every policy here therefore emits dots in
strict posterior order with an exact minimum-separation constraint (a greedy
unit-disk graph independent set, highest posterior first), so no low-posterior
dot can ever suppress a high-posterior crest.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label


def thin_ordered(P: np.ndarray, allowed: np.ndarray, min_sep: float,
                 budget: int | None = None) -> np.ndarray:
    """Greedy posterior-ordered thinning with exact euclidean min separation.

    Returns a boolean grid of emitted dots (unit weight candidate).
    """
    H, W = P.shape
    vals = P[allowed]
    order = np.argsort(-vals, kind="stable")
    rows, cols = np.nonzero(allowed)
    rows = rows[order]
    cols = cols[order]
    r = int(np.ceil(min_sep))
    stamp = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= min_sep * min_sep:
                stamp.append((dy, dx))
    taken = np.zeros((H, W), dtype=bool)
    out = np.zeros((H, W), dtype=bool)
    count = 0
    vy = vals[order]
    for i in range(rows.size):
        if vy[i] <= 0:
            break
        y = rows[i]
        x = cols[i]
        if taken[y, x]:
            continue
        out[y, x] = True
        count += 1
        if budget is not None and count >= budget:
            break
        for dy, dx in stamp:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < H and 0 <= xx < W:
                taken[yy, xx] = True
    return out


def anti_speckle_filter(dots: np.ndarray, P: np.ndarray, tau_support: float,
                        min_support_px: int = 3) -> np.ndarray:
    """Remove dots that sit on tiny isolated components of the posterior support
    (components of {P >= tau_support} smaller than min_support_px pixels)."""
    support = P >= tau_support
    comp, ncomp = cc_label(support, structure=np.ones((3, 3), int))
    sizes = np.bincount(comp.ravel(), minlength=ncomp + 1)
    yy, xx = np.nonzero(dots)
    cid = comp[yy, xx]
    keep = sizes[cid] >= min_support_px
    out = np.zeros_like(dots)
    out[yy[keep], xx[keep]] = True
    return out


def build_candidates(P: np.ndarray, P_aug: np.ndarray, allowed: np.ndarray,
                     support_tau: float) -> dict:
    """The lane's candidate set: spacing x posterior variants + budget and
    anti-speckle variants. All binary unit-dot emissions (the proven family)."""
    cands = {}
    # distinct euclidean-lattice suppression levels (defect fix: s2.4==s2.8, s3.2==s3.6
    # produced byte-identical stamp sets in session 1; these levels are all distinct)
    levels = (2.0, 2.24, 2.83, 3.0, 3.17, 3.61)
    for s in levels:
        cands[f"thin-s{s:.2f}"] = thin_ordered(P, allowed, min_sep=s)
    for s in (2.24, 2.83, 3.17):
        cands[f"sgmc-thin-s{s:.2f}"] = thin_ordered(P_aug, allowed, min_sep=s)
    cands["thin-s2.83-b30k"] = thin_ordered(P, allowed, min_sep=2.83, budget=30_000)
    cands["thin-s2.83-b38k"] = thin_ordered(P, allowed, min_sep=2.83, budget=38_000)
    for s in (2.24, 2.83):
        base = thin_ordered(P, allowed, min_sep=s)
        cands[f"as-thin-s{s:.2f}"] = anti_speckle_filter(base, P, tau_support=support_tau)
    return cands
