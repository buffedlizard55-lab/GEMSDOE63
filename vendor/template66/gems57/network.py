"""Fault-network geometry: segments, local strike, length, and offset frame.

All geometry is derived from the mapped-fault raster *as supplied for the fold*,
never from the full catalogue, so that a hide-and-recover fold cannot leak a
withheld segment into a feature (parallel-run protocol rule 2).

Coordinate conventions
----------------------
Arrays are indexed ``[row, col]``; row increases southwards, col eastwards, at
100 m/px.  A direction vector ``(dr, dc)`` therefore has east component ``dc``
and north component ``-dr``, and its geological azimuth (clockwise from north)
is ``atan2(dc, -dr)``, folded to the half-open interval ``[0, 180)`` for strike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

STRUCT3 = np.ones((3, 3), bool)


def junctions(mask: np.ndarray) -> np.ndarray:
    """Fault pixels with >= 3 eight-connected fault neighbours (trace intersections)."""
    m = np.asarray(mask, bool)
    n = ndi.convolve(m.astype(np.int16), np.ones((3, 3), np.int16), mode="constant") - m.astype(np.int16)
    return m & (n >= 3)


def segments(mask: np.ndarray, max_len_px: int = 12) -> tuple[np.ndarray, int, np.ndarray]:
    """Split a binary trace mask into segments.

    Segments are the branches between intersection pixels, further cut into
    chunks of at most ``max_len_px`` pixels (1.2 km at 100 m) so that strike is
    locally meaningful and a single long trace is not withheld all at once.

    Returns ``(seg_id, n_seg, seg_chunk)`` where ``seg_chunk`` groups chunks that
    belong to the same underlying branch (used only for diagnostics).
    """
    m = np.asarray(mask, bool)
    junc = junctions(m)
    branches, nb = ndi.label(m & ~junc, structure=STRUCT3)
    if nb == 0:
        return np.zeros(m.shape, np.int32), 0, np.zeros(m.shape, np.int32)
    H, W = m.shape
    flat = branches.ravel()
    order = np.argsort(flat, kind="stable").astype(np.int64)
    sorted_ids = flat[order]
    bounds = np.searchsorted(sorted_ids, np.arange(nb + 2))
    seg = np.zeros(m.shape, np.int32)
    nxt = 1
    for b in range(1, nb + 1):
        a, c = int(bounds[b]), int(bounds[b + 1])
        if c <= a:
            continue
        idx = order[a:c]
        n = c - a
        if n <= max_len_px:
            seg.ravel()[idx] = nxt
            nxt += 1
            continue
        # Cut the branch into <= max_len_px chunks along its principal axis so
        # each chunk stays spatially contiguous.
        fy = (idx // W).astype(np.float64)
        fx = (idx % W).astype(np.float64)
        cy, cx = fy.mean(), fx.mean()
        pts = np.stack([fy - cy, fx - cx], 1)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        t = pts @ vt[0]
        o = np.argsort(t)
        for start in range(0, n, max_len_px):
            sel = idx[o[start:start + max_len_px]]
            seg.ravel()[sel] = nxt
            nxt += 1
    return seg, nxt - 1, branches


@dataclass
class SegmentTable:
    seg_id: np.ndarray
    n_seg: int
    length: np.ndarray          # (n_seg+1,) segment length in pixels
    strike: np.ndarray          # (n_seg+1,) segment strike, azimuth degrees in [0,180)
    cy: np.ndarray
    cx: np.ndarray


def segment_table(seg: np.ndarray, n_seg: int) -> SegmentTable:
    """Per-segment length, principal-axis strike and centroid."""
    length = np.zeros(n_seg + 1, np.float64)
    strike = np.full(n_seg + 1, np.nan, np.float64)
    cy = np.zeros(n_seg + 1, np.float64)
    cx = np.zeros(n_seg + 1, np.float64)
    if n_seg == 0:
        return SegmentTable(seg, 0, length, strike, cy, cx)
    H, W = seg.shape
    flat = seg.ravel()
    order = np.argsort(flat, kind="stable").astype(np.int64)
    sorted_ids = flat[order]
    bounds = np.searchsorted(sorted_ids, np.arange(n_seg + 2))
    for s in range(1, n_seg + 1):
        a, b = int(bounds[s]), int(bounds[s + 1])
        if b <= a:
            continue
        idx = order[a:b]
        py = (idx // W).astype(np.float64)
        px = (idx % W).astype(np.float64)
        length[s] = b - a
        cy[s], cx[s] = py.mean(), px.mean()
        if b - a >= 3:
            pts = np.stack([py - py.mean(), px - px.mean()], 1)
            _, _, vt = np.linalg.svd(pts, full_matrices=False)
            strike[s] = _azimuth(vt[0][0], vt[0][1])
        elif b - a == 2:
            strike[s] = _azimuth(py[1] - py[0], px[1] - px[0])
    return SegmentTable(seg, n_seg, length, strike, cy, cx)


def _azimuth(dr: float, dc: float) -> float:
    """Geological strike (azimuth clockwise from north, folded to [0,180))."""
    a = np.degrees(np.arctan2(dc, -dr)) % 180.0
    return float(a)


def local_strike(mask: np.ndarray, smooth_px: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel trace orientation and coherence from the structure tensor.

    ``smooth_px`` is the tensor-smoothing half-width in pixels (3 px = 300 m).
    Returns ``(strike_deg, coherence)`` with ``strike_deg`` in ``[0, 180)`` and
    ``coherence`` in ``[0, 1]`` (0 = isotropic / unreliable, 1 = perfectly linear).
    Pixels off the (dilated) trace carry ``nan`` strike and 0 coherence.
    """
    m = np.asarray(mask, np.float32)
    g_y, g_x = np.gradient(ndi.gaussian_filter(m, 1.0))
    jxx = ndi.gaussian_filter(g_x * g_x, smooth_px)
    jyy = ndi.gaussian_filter(g_y * g_y, smooth_px)
    jxy = ndi.gaussian_filter(g_x * g_y, smooth_px)
    tr = jxx + jyy
    diff = jxx - jyy
    coh = np.sqrt(diff * diff + 4.0 * jxy * jxy) / np.maximum(tr, 1e-12)
    # theta_grad points along the strongest intensity change, i.e. perpendicular
    # to the trace; the trace direction is theta_grad + 90 deg.
    theta = 0.5 * np.arctan2(2.0 * jxy, diff) + np.pi / 2.0
    dr, dc = np.sin(theta), np.cos(theta)
    strike = np.degrees(np.arctan2(dc, -dr)) % 180.0
    support = ndi.binary_dilation(np.asarray(mask, bool), STRUCT3, iterations=1)
    strike = np.where(support, strike, np.nan).astype(np.float32)
    coh = np.where(support, coh, 0.0).astype(np.float32)
    del g_y, g_x, jxx, jyy, jxy, tr, diff, theta, dr, dc, support
    return strike, coh


def nearest_frame(visible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance to, and index of, the nearest visible fault pixel for every cell.

    Returns ``(dist_px, idx_row, idx_col)``.  Cells with no visible fault within
    the array carry the far-field value from the EDT.
    """
    mask = np.asarray(visible, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('nearest visible fault is undefined for an empty/non-2D catalogue')
    d, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
    return d, iy, ix


def offset_components(dy: np.ndarray, dx: np.ndarray, strike_deg: np.ndarray):
    """Decompose an offset vector into along-strike and cross-strike parts.

    ``strike_deg`` is the azimuth of the *reference* trace at the anchor point.
    Returns ``(d_par, d_perp, side)`` with ``d_par`` the signed along-strike
    offset (px), ``d_perp`` the absolute cross-strike stepover (px) and ``side``
    in ``{-1, +1, 0}`` giving which side of the trace the cell lies on.
    """
    th = np.radians(strike_deg)
    # unit along-strike vector in (row, col) space
    ur, uc = -np.cos(th), np.sin(th)
    d_par = dy * ur + dx * uc
    # perpendicular (rotate along-strike by +90 deg in (row,col))
    pr, pc = -uc, ur
    d_perp_s = dy * pr + dx * pc
    return d_par, np.abs(d_perp_s), np.sign(d_perp_s)


def rel_strike(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Acute angle between two strikes, folded to [0, 90] degrees."""
    d = np.abs(np.asarray(a_deg, float) - np.asarray(b_deg, float)) % 180.0
    return np.where(d > 90.0, 180.0 - d, d)


def component_detached(mask: np.ndarray, detach_px: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Per-component test: is this component at least ``detach_px`` from all others?

    Returns ``(comp_label, is_detached)`` where ``is_detached`` is indexed by
    component label (index 0 unused).  Computed with a windowed Euclidean
    distance transform per component, which is exact for the threshold test: if
    the ``(2*detach_px+1)`` window around a component contains no pixel of any
    other component, the true separation exceeds ``detach_px``.
    """
    m = np.asarray(mask, bool)
    comp, ncomp = ndi.label(m, structure=STRUCT3)
    is_det = np.zeros(ncomp + 1, bool)
    if ncomp == 0:
        return comp, is_det
    H, W = m.shape
    r = int(detach_px)
    objs = ndi.find_objects(comp)
    for cid, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y0 = max(0, sl[0].start - r); y1 = min(H, sl[0].stop + r)
        x0 = max(0, sl[1].start - r); x1 = min(W, sl[1].stop + r)
        win = comp[y0:y1, x0:x1]
        own = (win == cid)
        other = (win != 0) & ~own
        if not other.any():
            is_det[cid] = True
            continue
        d = ndi.distance_transform_edt(~other)
        is_det[cid] = bool(d[own].min() > detach_px)
    return comp, is_det
