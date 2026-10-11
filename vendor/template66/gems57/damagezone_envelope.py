"""H58: displacement-scaled damage-zone envelope with sense-conditioned obliquity.

Lane
----
Fault-zone anatomy -- where do the *secondary* strands sit around a mapped fault?
This module supplies the two quantities the earlier H57 builds never measured on
the repaired instrument.

``W(L)`` -- fitted damage-zone half-width
    Savage & Brodsky (JGR 2011) find secondary-fracture density decaying away
    from the primary with a zone width that grows with displacement and then
    more slowly, so the scaling is expected to be sub-linear.  Connected
    component length of the mapped trace is the repository's displacement
    proxy.  :func:`fit_width_law` estimates the exponent ``gamma`` in
    ``W(L) = w0 * (L / L0) ** gamma`` by bin-wise quantile regression on the
    hide-and-recover folds, with a bin-level bootstrap interval.  Nothing is
    hard-coded: :func:`evidence/h58_structure.json` records which bins supported
    the fit.  ``evidence/exp4_width.json`` measured a ``sqrt(length)`` variant
    only on the retired ``gems52-pooled-hide-v1`` frame, so the law is retested
    here on the repaired instrument.

``obliq_signed`` -- sense-conditioned radial obliquity
    The signed angle between the local strike of the nearest *visible* trace and
    the azimuth of the anchor -> pixel ray.  A Riedel pattern (Tchalenko 1970;
    Schreurs 2003) predicts a handed asymmetry of the halo: for a given slip
    sense the en echelon shears nucleate on one side of the master.  The shipped
    model only ever had the unsigned ``side`` of the trace and an axial ``cos2``
    strike encoding, and ``REMAINING_WORK.md`` section 7 records that signed
    angular handedness was never tested.  Multiplying by the recorded sense sign
    of the nearest visible strike-slip host record (RL +1, LL -1, else 0) gives
    a handedness-relative obliquity that the folds can either support or reject.

Every column is a deterministic function of the visible mask (plus the
visible-restricted INGENIOUS sense raster), so the hide-and-recover protocol
holds and each column still passes the shared single-feature AUC canary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import local_strike, nearest_frame

#: Pixels of mapped-component length used for the width-law bins (1 px = 100 m).
LEN_EDGES = np.array([0, 5, 10, 20, 40, 80, 160, 320, 1e12], dtype=float)
#: Signed-obliquity bins in degrees, host-relative, before/after the trace.
OBLIQUITY_EDGES = np.array([-90, -60, -30, -12, 12, 30, 60, 90], dtype=float)
MIN_BIN_POSITIVES = 200
#: Organiser thread 11516 penalises a dot next to a known trace that is not a
#: new-fault pixel; the top owner-reported live raster pruned exactly its
#: ``<= 2 px`` flank.  Emitted here as an explicit, tested policy, not a default.
FLANK_PX_DEFAULT = 2.0
#: Clipping range for the fitted half-width so a degenerate exponent cannot
#: produce a zero-width or whole-footprint envelope.
W_MIN_PX, W_MAX_PX = 2.0, 60.0
EXTRA_FEATURES = ("obliq_signed", "obliq_signed_sense", "d_over_w")


@dataclass
class WidthLaw:
    """``W(L) = exp(log_w0) * (L / l0) ** gamma``, clipped to ``[W_MIN_PX, W_MAX_PX]``."""

    gamma: float
    log_w0: float
    l0_px: float
    quantile: float
    bins: list
    gamma_ci95: list
    n_positives: int
    sublinear: bool
    n_bins: int

    def half_width(self, length_px: np.ndarray) -> np.ndarray:
        l = np.maximum(np.asarray(length_px, np.float64), 1.0)
        return np.clip(np.exp(self.log_w0) * (l / self.l0_px) ** self.gamma, W_MIN_PX, W_MAX_PX)

    def to_json(self) -> dict:
        return dict(
            model="W(L) = exp(log_w0) * (L/l0)**gamma, clipped to [W_MIN_PX, W_MAX_PX]",
            evidence_class="HOLDOUT-STRUCTURE (fitted on visible-only folds, not a score)",
            gamma=self.gamma, gamma_ci95=self.gamma_ci95, log_w0=self.log_w0,
            w0_px=float(np.exp(self.log_w0)), l0_px=self.l0_px, quantile=self.quantile,
            n_positives=self.n_positives, n_bins=self.n_bins, sublinear=self.sublinear,
            bins=self.bins, w_min_px=W_MIN_PX, w_max_px=W_MAX_PX,
            min_bin_positives=MIN_BIN_POSITIVES,
            interpretation=("gamma<1 with an interval excluding 1 supports sub-linear "
                            "(Savage-Brodsky-like) widening; gamma>=1 means the mapped "
                            "component length does not narrow the halo in this instrument"),
        )


def fit_width_law(length_px: np.ndarray, distance_px: np.ndarray, populated: np.ndarray,
                  *, quantile: float = 0.90, l0_px: float = 40.0) -> WidthLaw:
    """Fit ``log W = a + gamma log L`` to per-length-bin quantiles of withheld distances.

    ``length_px`` is the size of the mapped connected component that owns the
    nearest visible trace pixel, ``distance_px`` the distance to that visible
    trace, and ``populated`` a boolean mask selecting the withheld positives in
    the *training* folds.  Bins with fewer than ``MIN_BIN_POSITIVES`` positives
    are dropped so a sparse bin cannot drive the exponent.  The bootstrap
    resamples bins, because the fit is one point per bin and pixel resampling
    would understate model uncertainty.
    """
    length_px = np.asarray(length_px, np.float64).ravel()
    distance_px = np.asarray(distance_px, np.float64).ravel()
    populated = np.asarray(populated, bool).ravel()
    if not (length_px.shape == distance_px.shape == populated.shape):
        raise ValueError("length, distance and populated masks must be aligned 1-D arrays")
    if distance_px.size and not np.isfinite(distance_px).all():
        raise ValueError("withheld distances must be finite")
    if populated.any() and populated.size != length_px.size:
        raise ValueError("populated mask length must match the sample arrays")
    if int(populated.sum()) < 2 * MIN_BIN_POSITIVES:
        raise ValueError("too few withheld positives to fit a width law")
    length_px, distance_px = length_px[populated], distance_px[populated]
    bins = []
    for lo, hi in zip(LEN_EDGES[:-1], LEN_EDGES[1:]):
        sel = (length_px >= lo) & (length_px < hi)
        n_pos = int(sel.sum())
        if n_pos < MIN_BIN_POSITIVES:
            continue
        bins.append(dict(len_lo=float(lo), len_hi=float(hi), n_positives=n_pos,
                         length_median_px=float(np.median(length_px[sel])),
                         width_px=float(np.quantile(distance_px[sel], quantile))))
    if len(bins) < 3:
        raise ValueError("need at least three populated length bins to fit an exponent")
    x = np.log([b["length_median_px"] for b in bins])
    z = np.log([max(b["width_px"], 1e-6) for b in bins])
    wt = np.sqrt([b["n_positives"] for b in bins])

    def wls(xs, zs, ws):
        xm = np.average(xs, weights=ws)
        zm = np.average(zs, weights=ws)
        sxx = float(np.sum(ws * (xs - xm) ** 2))
        g = float(np.sum(ws * (xs - xm) * (zs - zm)) / max(sxx, 1e-12))
        return g, float(zm - g * xm)

    gamma, a = wls(x, z, wt)
    rng = np.random.default_rng(20261010)
    boots = []
    for _ in range(400):
        idx = rng.integers(0, len(bins), size=len(bins))
        g, _ = wls(x[idx], z[idx], wt[idx])
        if np.isfinite(g):
            boots.append(g)
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return WidthLaw(gamma=float(gamma), log_w0=float(a), l0_px=float(l0_px), bins=bins,
                    gamma_ci95=ci, quantile=float(quantile), n_positives=int(populated.sum()),
                    sublinear=bool(ci[1] < 1.0), n_bins=len(bins))


def signed_obliquity(visible: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Signed host-relative angle (degrees, ``[-90, 90]``) at the given pixels.

    The reference direction is the local strike of the nearest *visible* trace
    pixel; the measured direction is the ray from that anchor to the pixel.  A
    positive value is clockwise from the host strike in map view.  The angle is
    axial, so it is folded to ``[-90, 90]`` and a strand at +/-6 deg from the
    host shares the near-parallel central bin.  Pixels whose anchor has no
    finite strike (isolated single-pixel traces) are set to 0, and that
    censoring count is returned and reported rather than hidden.  A pixel lying
    exactly on its anchor has an undefined ray; it takes the ``az = 0``
    convention here and is excluded by the pipeline anyway, because the analysis
    domain and both flank policies require positive distance.
    """
    visible = np.asarray(visible, bool)
    rows = np.asarray(rows, np.intp)
    cols = np.asarray(cols, np.intp)
    if rows.shape != cols.shape or rows.ndim != 1:
        raise ValueError("rows and cols must be matching 1-D index arrays")
    if not visible.any():
        raise ValueError("signed obliquity needs a non-empty visible mask")
    _d, iy, ix = nearest_frame(visible)
    strike, _coh = local_strike(visible, smooth_px=3.0)
    ay, ax = iy[rows, cols], ix[rows, cols]
    dr = (rows - ay).astype(np.float64)
    dc = (cols - ax).astype(np.float64)
    az = np.degrees(np.arctan2(dc, -dr)) % 180.0
    s = strike[ay, ax].astype(np.float64)
    ok = np.isfinite(s)
    rel = np.where(ok, (az - np.where(ok, s, 0.0) + 90.0) % 180.0 - 90.0, 0.0)
    rel = np.clip(rel, -90.0, 90.0).astype(np.float32)
    return rel, int((~ok).sum())


def obliquity_at_anchor(visible: np.ndarray, rows, cols):
    """Compatibility wrapper returning only the angle array."""
    return signed_obliquity(visible, rows, cols)[0]


def flank_suppress(d_cat: np.ndarray, flank_px: float = FLANK_PX_DEFAULT) -> np.ndarray:
    """Boolean 'keep' mask against the catalogue-adjacency penalty (thread 11516)."""
    return np.asarray(d_cat, np.float64) > float(flank_px)
