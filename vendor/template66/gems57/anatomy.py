"""Fault-zone anatomy: a *fitted* secondary-strand intensity around known faults.

Lane hypothesis
---------------
Secondary strands (Riedel shears, splays, parallel strands, newly mapped
geometry of an existing system) are not distributed isotropically around a
mapped trace.  Distributed-shear analogue experiments (Schreurs 2003) and the
classical Riedel framework (Tchalenko 1970) produce en echelon arrays at an
acute angle to the bulk shear; damage-zone work (Savage & Brodsky, JGR 2011)
shows secondary-fracture density decaying away from the primary with a zone
width that grows with displacement.  In the northern Walker Lane the expression
is left-stepping dextral strands (Faulds, Henry & Hinz).

Nothing here is hard-coded.  Every quantity -- the distance decay, the
cross-strike stepover scale, the along-strike extent, the left/right asymmetry,
the orientation selectivity and the length (displacement-proxy) scaling -- is
**estimated from the hide-and-recover holdout**: the relative-strike and
distance distributions of withheld segments measured against their nearest
visible fault.  Features that the data does not support are dropped.

Sense of slip
-------------
The protocol asks to condition on recorded sense of slip "where the database
has it".  The raster catalogue (``existing_faults.tif``) carries no sense field:
its values are ``{-1 (nodata, outside the study area), 0, 1}``.  The INGENIOUS
**vector** attribute table does carry it: ``data/external/trace_segments_utm11.csv``
has a ``sense`` column (N 66,861 / RL 8,448 / LL 7,628 / blank 1,394 segments)
and ``qfault_attributes.csv`` has ``SLIPSENSE``.  The first version of this module
said no such field existed; that was wrong (``IR-57-SLIP-02``).  The recorded
sense enters as the opt-in ``SENSE_FEATURES`` (``sense_sgn``, ``sense_side``),
which is measured against the shipped 8-feature set in
``scripts/run_sense_experiment.py``.  The untethered ``side`` feature (fitted
left/right asymmetry, no recorded sense) is kept as it was (``IR-57-SLIP-01``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .grid import Grid
from .network import (STRUCT3, SegmentTable, local_strike, nearest_frame,
                      offset_components, segment_table, segments)

# Opt-in recorded-sense features (lane brief: "conditioned on recorded sense of
# slip where the database has it").  Appended to X only when ``sense_src`` is
# passed to ``fold_geometry``; the default feature matrix is unchanged, so the
# shipped pipeline and every earlier measurement are unaffected.
SENSE_FEATURES = (
    "sense_sgn",    # +1 nearest visible strike-slip record RL, -1 LL, 0 otherwise (N / blank)
    "sense_side",   # sense_sgn * side: handedness-relative side of the strand (fitted, not assumed)
)

FEATURES = (
    "d",            # Euclidean distance to nearest visible fault (px)
    "d_perp",       # cross-strike stepover (px)
    "d_par_abs",    # along-strike offset magnitude (px)
    "side",         # signed side of the trace (-1 / +1)
    "log_len",      # log length of the anchor *component* -- displacement proxy
    "sin2",         # cyclic strike encoding, sin(2*strike)
    "cos2",         # cyclic strike encoding, cos(2*strike)
    "coherence",    # structure-tensor linearity at the anchor
    "density",      # visible fault pixels within a 5 px (500 m) radius
    # Explicit orientation x distance interactions (H57-D, session 2,
    # 2026-10-09).  sin2/cos2 alone are weak marginals, so the orientation
    # selectivity of the halo is only usable in interaction.  These two
    # products let the model express "the distance decay depends on the
    # parent trace's strike" without any hard-coded angle: the fitted
    # coefficients define the preferred orientations.  Both are computable
    # from the visible catalogue alone, so the holdout protocol is unchanged.
    # Measured in scripts/run_cv_r2.py: +0.0018 HOLDOUT-DTI vs the 8-feature
    # shipped set (inside the noise) -- a negative result, kept addressable
    # so the ablation stays reproducible.
    "sin2d",        # sin(2*strike) * d  -- orientation-modulated distance decay
    "cos2d",        # cos(2*strike) * d  -- orientation-modulated distance decay
)

# NOTE (bug fixed 2026-10-09): ``log_len`` was originally the length of the
# <= 12 px segment chunk, which made it a constant (log1p(12) = 2.56) and
# destroyed the displacement proxy the lane requires.  Segments are chunked only
# so that strike is locally meaningful and so that withholding removes a
# contiguous piece of trace; the displacement proxy is the size of the whole
# connected mapped component (the fault system) containing the anchor pixel,
# which is what scales with damage-zone width in Savage & Brodsky (2011).


@dataclass
class FoldGeometry:
    """Per-pixel geometry of the active domain of one fold cell, from visible faults only.

    ``X`` has one column per entry of ``feature_names``: the catalogue-geometry
    block (:data:`FEATURES`), optionally followed by the geophysical
    corroboration block (:data:`gems57.geo.GEO_FEATURES`) when ``fold_geometry``
    was called with ``geo`` planes.  Geophysical columns are static raster
    planes, so they cannot leak the withholding mask, but every column is still
    passed through the leakage canary.
    """
    key: str
    rows: np.ndarray
    cols: np.ndarray
    X: np.ndarray                     # (n, len(feature_names)) float32
    y: np.ndarray                     # 1 = withheld (hidden) truth pixel
    visible: np.ndarray
    n_hidden: int
    seg: SegmentTable = field(repr=False)
    feature_names: tuple = FEATURES


def _density(visible: np.ndarray, radius_px: int = 5) -> np.ndarray:
    """Count of visible fault pixels in a disc of ``radius_px`` around each cell."""
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    k = (yy * yy + xx * xx) <= radius_px * radius_px
    return ndi.convolve(visible.astype(np.float32), k.astype(np.float32), mode="constant")


def fold_geometry(grid: Grid, visible: np.ndarray, hidden: np.ndarray,
                  domain: np.ndarray, key: str,
                  geo: dict | None = None,
                  sense_src: np.ndarray | None = None) -> FoldGeometry:
    """Build the feature matrix for one fold cell using **visible faults only**.

    Two opt-in blocks may be appended after :data:`FEATURES`, in this order:

    ``sense_src`` is the unmasked int8 sense raster (see
    ``faultzone.trace_sense_raster``).  It is restricted to *visible* pixels before
    any use, so a withheld segment's sense cannot reach a feature (leakage rule);
    the appended columns are :data:`SENSE_FEATURES`.

    ``geo`` optionally maps names in :data:`gems57.geo.GEO_FEATURES` to full-grid
    float32 planes (see :func:`gems57.geo.load_planes`); those columns come last and
    ``FoldGeometry.feature_names`` records the combined order.
    """
    strike, coh = local_strike(visible, smooth_px=3.0)
    seg, n_seg = segments(visible, max_len_px=12)[:2]
    tab = segment_table(seg, n_seg)
    comp, ncomp = ndi.label(visible, structure=STRUCT3)
    comp_len = np.bincount(comp.ravel(), minlength=ncomp + 1).astype(np.float64)
    dens = _density(visible, 5)

    d, iy, ix = nearest_frame(visible)
    active = domain & ~visible
    names = (tuple(FEATURES)
             + (SENSE_FEATURES if sense_src is not None else ())
             + (tuple(geo) if geo else ()))
    ys, xs = np.nonzero(active)
    if ys.size == 0:
        return FoldGeometry(key, ys, xs, np.zeros((0, len(names)), np.float32),
                            np.zeros(0, np.int8), visible, 0, tab,
                            feature_names=names)

    ay, ax = iy[ys, xs], ix[ys, xs]
    dy = (ys - ay).astype(np.float32)
    dx = (xs - ax).astype(np.float32)

    # strike at the anchor: fall back to the anchor segment's principal strike
    anc_seg = seg[ay, ax]
    s_anchor = strike[ay, ax]
    s_seg = tab.strike[np.clip(anc_seg, 0, len(tab.strike) - 1)]
    s = np.where(np.isfinite(s_anchor), s_anchor, s_seg)
    # IR-57-STRIKE-01: the old argument order replaced EVERY finite strike
    # with zero, making sin2 constant 0/cos2 constant 1 and rotating all offsets
    # onto a global north/south frame. Invalid strike, not valid strike, falls back.
    s = np.where(np.isfinite(s), s, 0.0)

    d_par, d_perp, side = offset_components(dy, dx, s)
    # displacement proxy: size of the whole mapped component, not the 12 px chunk
    ln = comp_len[np.clip(comp[ay, ax], 0, len(comp_len) - 1)]

    th2 = np.radians(2.0 * s)
    cols = [
        d[ys, xs],
        d_perp,
        np.abs(d_par),
        side,
        np.log1p(ln),
        np.sin(th2),
        np.cos(th2),
        coh[ay, ax],
        dens[ys, xs],
        np.sin(th2) * d[ys, xs],
        np.cos(th2) * d[ys, xs],
    ]
    if sense_src is not None:
        # nearest VISIBLE pixel carrying a recorded sense (visible-only, leakage-safe)
        src = visible & (sense_src > 0)
        if src.any():
            distance, (sy, sx) = ndi.distance_transform_edt(~src, return_indices=True)
            code = sense_src[sy[ay, ax], sx[ay, ax]].astype(np.float64)
            # Do not assign a distant record to an unrelated visible fault.
            code[distance[ay, ax] > 1.0] = 0
            del distance, sy, sx
        else:
            code = np.zeros(ay.shape, np.float64)
        sgn = np.where(code == 2, 1.0, np.where(code == 3, -1.0, 0.0))
        cols += [sgn, sgn * side]
    if geo:
        # order of insertion in ``geo`` defines the appended column order;
        # callers pass a dict built from GEO_FEATURES to keep it canonical
        for gname in geo:
            cols.append(geo[gname][ys, xs])
    X = np.stack(cols, axis=1).astype(np.float32)

    y = hidden[ys, xs].astype(np.int8)
    return FoldGeometry(key=key, rows=ys, cols=xs, X=X, y=y, visible=visible,
                        n_hidden=int(y.sum()), seg=tab, feature_names=names)


# --------------------------------------------------------------------------- #
# The measurement the lane protocol asks for: distributions of withheld pixels
# --------------------------------------------------------------------------- #

def measure_withheld(g: FoldGeometry, d_edges=None) -> dict:
    """Distance / stepover / along-strike / side distributions of withheld pixels."""
    if d_edges is None:
        d_edges = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 33, 41, 51, 1e9])
    pos = g.y == 1
    Xi = g.X
    n_pos = int(pos.sum())
    n_all = int(g.y.size)

    def hist(col: int, edges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = Xi[:, col]
        hp, _ = np.histogram(v[pos], bins=edges)
        ha, _ = np.histogram(v, bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            enrich = np.where(ha > 0, hp / np.maximum(ha, 1), 0.0)
        return hp.astype(float), ha.astype(float), enrich

    h_d, a_d, e_d = hist(0, d_edges)
    perp_edges = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 1e9])
    h_p, a_p, e_p = hist(1, perp_edges)
    par_edges = np.array([0, 1, 2, 4, 6, 9, 13, 18, 25, 34, 46, 1e9])
    h_par, a_par, e_par = hist(2, par_edges)

    base = n_pos / max(n_all, 1)
    side_pos = float(Xi[pos, 3].sum()) if n_pos else 0.0
    n_left = int(((Xi[:, 3] < 0) & pos).sum())
    n_right = int(((Xi[:, 3] > 0) & pos).sum())
    a_left = int((Xi[:, 3] < 0).sum())
    a_right = int((Xi[:, 3] > 0).sum())

    len_edges = np.array([0, 5, 10, 20, 40, 80, 1e9])
    h_l, a_l, e_l = hist(4, len_edges)

    # joint (cross-strike stepover, along-strike offset): this is the direct test
    # of the en echelon prediction.  A purely isotropic halo would put its mass on
    # the diagonal; en echelon stepping puts it off-diagonal at small stepover and
    # finite along-strike offset.
    pe = np.array([0, 1, 2, 3, 4, 6, 9, 14, 1e9])
    ae = np.array([0, 1, 2, 3, 4, 6, 9, 14, 1e9])
    Hp, _, _ = np.histogram2d(Xi[pos, 1], Xi[pos, 2], bins=[pe, ae])
    Ha, _, _ = np.histogram2d(Xi[:, 1], Xi[:, 2], bins=[pe, ae])
    with np.errstate(divide="ignore", invalid="ignore"):
        Ej = np.where(Ha > 0, Hp / np.maximum(Ha, 1), 0.0)

    return {
        "n_withheld": n_pos,
        "n_domain": n_all,
        "base_rate": float(base),
        "distance": {"edges": d_edges.tolist(), "n_withheld": h_d.tolist(),
                     "n_domain": a_d.tolist(), "enrichment": e_d.tolist()},
        "stepover": {"edges": perp_edges.tolist(), "n_withheld": h_p.tolist(),
                     "n_domain": a_p.tolist(), "enrichment": e_p.tolist()},
        "along_strike": {"edges": par_edges.tolist(), "n_withheld": h_par.tolist(),
                         "n_domain": a_par.tolist(), "enrichment": e_par.tolist()},
        "segment_length": {"edges": len_edges.tolist(), "n_withheld": h_l.tolist(),
                           "n_domain": a_l.tolist(), "enrichment": e_l.tolist()},
        "side": {"n_withheld_left": n_left, "n_withheld_right": n_right,
                 "n_domain_left": a_left, "n_domain_right": a_right,
                 "rate_left": n_left / max(a_left, 1), "rate_right": n_right / max(a_right, 1),
                 "log_ratio_R_over_L": float(np.log((n_right / max(a_right, 1))
                                                    / max(n_left / max(a_left, 1), 1e-12)))},
        "joint_stepover_alongstrike": {
            "stepover_edges": pe.tolist(), "along_edges": ae.tolist(),
            "n_withheld": Hp.tolist(), "n_domain": Ha.tolist(),
            "enrichment": Ej.tolist()},
    }


def relative_strike_distribution(grid: Grid, visible: np.ndarray, hidden: np.ndarray,
                                 bins: int = 12) -> dict:
    """Relative strike of withheld trace pixels vs their nearest visible trace.

    This is the direct test of the en echelon / Riedel prediction: if withheld
    strands are systematically oblique to the primary, the relative-strike
    histogram of *withheld* pixels must differ from that of visible pixels
    measured the same way.
    """
    sv, cv = local_strike(visible, 3.0)
    sh, ch = local_strike(hidden, 3.0)
    _, iy, ix = nearest_frame(visible)
    edges = np.linspace(0, 90, bins + 1)

    def fold(a, b):
        d = np.abs(a - b) % 180.0
        return np.where(d > 90.0, 180.0 - d, d)

    # withheld pixels vs their nearest visible trace
    ys, xs = np.nonzero(hidden)
    if ys.size:
        ay, ax = iy[ys, xs], ix[ys, xs]
        ok = (np.isfinite(sh[ys, xs]) & np.isfinite(sv[ay, ax])
              & (cv[ay, ax] > 0.2) & (ch[ys, xs] > 0.2))
        hid = fold(sh[ys, xs][ok], sv[ay, ax][ok])
    else:
        hid = np.zeros(0)

    # NULL: visible pixels vs their nearest visible pixel that belongs to a
    # DIFFERENT connected component.  Comparing a visible pixel to its own
    # nearest visible pixel is degenerate (it returns itself, angle 0), which
    # was the bug in the first implementation.
    vcomp, _ = ndi.label(visible, structure=STRUCT3)
    vy, vx = np.nonzero(visible)
    vis = np.zeros(0)
    if vy.size > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.stack([vy, vx], 1))
        _d, idx = tree.query(np.stack([vy, vx], 1), k=min(13, len(vy)))
        cid = vcomp[vy, vx]
        nbr_cid = vcomp[vy[idx], vx[idx]]
        other = nbr_cid != cid[:, None]
        first = np.argmax(other, axis=1)
        has = other.any(axis=1)
        sel = np.arange(vy.size)[has]
        j = idx[sel, first[has]]
        a = sv[vy[sel], vx[sel]]
        b = sv[vy[j], vx[j]]
        okn = np.isfinite(a) & np.isfinite(b) & (cv[vy[j], vx[j]] > 0.2) & (cv[vy[sel], vx[sel]] > 0.2)
        vis = fold(a[okn], b[okn])
    h_h, _ = np.histogram(hid, bins=edges) if hid.size else (np.zeros(bins), edges)
    h_v, _ = np.histogram(vis, bins=edges) if vis.size else (np.zeros(bins), edges)
    return {"edges": edges.tolist(),
            "angle_convention": "unsigned axial difference in degrees, folded to [0,90]",
            "sample_unit": "fault raster pixel; not segment-weighted",
            "minimum_local_coherence": 0.2,
            "n_withheld": h_h.astype(float).tolist(),
            "n_visible_reference": h_v.astype(float).tolist(),
            "n_withheld_total": int(hid.size),
            "n_visible_total": int(vis.size),
            "median_withheld": float(np.median(hid)) if hid.size else None,
            "median_visible": float(np.median(vis)) if vis.size else None,
            "p25_withheld": float(np.percentile(hid, 25)) if hid.size else None,
            "p75_withheld": float(np.percentile(hid, 75)) if hid.size else None,
            "null_caveat": "Visible-reference comparison selects the nearest different-component trace among the 13 nearest queried visible pixels (including self); locations without one and low-coherence pixels are omitted. Pixel-weighted and censored; descriptive only, no significance inferred."}


def package_holdout_structure(relative_strike: dict, *, withheld_positive_pixels: int,
                              distance_positive_quantiles_px: list[float],
                              distance_domain_quantiles_px: list[float],
                              pixel_size_m: int = 100) -> dict:
    """Keep angular histograms nested so counts cannot overwrite bin arrays.

    ``relative_strike_distribution`` uses ``n_withheld`` for the per-bin
    histogram. The scalar count of all withheld positive pixels has a distinct
    key here; separating them prevents a reporting collision from erasing the
    very distribution the experiment is meant to measure.
    """
    edges = np.asarray(relative_strike.get("edges"), dtype=np.float64)
    withheld_bins = np.asarray(relative_strike.get("n_withheld"), dtype=np.float64)
    visible_bins = np.asarray(relative_strike.get("n_visible_reference"), dtype=np.float64)
    if edges.ndim != 1 or withheld_bins.shape != (len(edges) - 1,):
        raise ValueError("relative-strike histogram does not match its bin edges")
    if visible_bins.shape != withheld_bins.shape:
        raise ValueError("withheld and visible-reference histograms differ in length")
    if (not np.isfinite(edges).all() or not np.isfinite(withheld_bins).all()
            or not np.isfinite(visible_bins).all() or (withheld_bins < 0).any()
            or (visible_bins < 0).any()):
        raise ValueError("relative-strike histogram contains invalid values")
    if int(withheld_bins.sum()) != int(relative_strike.get("n_withheld_total", -1)):
        raise ValueError("relative-strike withheld histogram total is inconsistent")
    if int(visible_bins.sum()) != int(relative_strike.get("n_visible_total", -1)):
        raise ValueError("relative-strike reference histogram total is inconsistent")
    q_pos = np.asarray(distance_positive_quantiles_px, dtype=np.float64)
    q_domain = np.asarray(distance_domain_quantiles_px, dtype=np.float64)
    if q_pos.shape != (5,) or q_domain.shape != (5,):
        raise ValueError("distance summary must contain the five declared quantiles")
    if (not np.isfinite(q_pos).all() or not np.isfinite(q_domain).all()
            or (np.diff(q_pos) < 0).any() or (np.diff(q_domain) < 0).any()):
        raise ValueError("distance quantiles must be finite and sorted")
    if int(withheld_positive_pixels) < 0 or int(pixel_size_m) <= 0:
        raise ValueError("invalid positive count or pixel size")
    return {
        "evidence_class": "HOLDOUT-STRUCTURE (descriptive, not a score)",
        "withheld_positive_pixels": int(withheld_positive_pixels),
        "relative_strike": relative_strike,
        "distance_quantile_probabilities": [0.1, 0.5, 0.9, 0.95, 0.99],
        "distance_positive_quantiles_px": q_pos.tolist(),
        "distance_domain_quantiles_px": q_domain.tolist(),
        "null_caveat": relative_strike.get("null_caveat",
            "Visible-reference comparison is descriptive; no significance is inferred."),
        "pixel_size_m": int(pixel_size_m),
    }
