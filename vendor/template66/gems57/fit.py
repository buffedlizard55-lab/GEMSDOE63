"""Fitted damage-zone factors for the fault-zone anatomy lane.

Every factor is an empirical distribution measured on the hide-and-recover
holdout (evidence/exp1b_pooled.npz): no textbook Riedel/Tchalenko angle is
hard-coded anywhere.  The one band the holdout cannot measure (d < 3 px,
removed by the protocol's 300 m truth buffer) is extended from the SGMC
off-catalogue population and flagged as such.
"""
from __future__ import annotations

import numpy as np

D_MAX_PX = 60.0          # fitted zone reach (holdout: 97% of withheld px within 50 px)
NEAR_BAND_PX = 3.0       # below this the holdout buffer removed the truth


def fit_distance_density(d_px: np.ndarray, d_max: float = D_MAX_PX) -> dict:
    """Empirical radial density f(d) on 1-px bins, normalised to a density."""
    hist, edges = np.histogram(d_px, bins=np.arange(0.0, d_max + 1.0, 1.0))
    hist = hist.astype(np.float64)
    density = hist / max(hist.sum(), 1.0)          # per-bin probability mass
    return dict(hist=hist, edges=edges, density=density,
                median=float(np.median(d_px)), n=int(d_px.size))


def extend_near_band_sgmc(d_px_holdout, d_px_sgmc, near_band: float = NEAR_BAND_PX) -> np.ndarray:
    """Density for d < near_band from SGMC, scaled for continuity at the band edge.

    The holdout's 300 m truth buffer removes d < ~3 px, so the holdout cannot
    measure the near-trace band.  SGMC off-catalogue faults (real faults
    captured by neither USGS nor INGENIOUS) measure it: 23.8% of their pixels
    sit within 3 px of a known fault.  The SGMC shape is scaled so its density
    matches the holdout density at the band edge (continuity), and the result
    is flagged SGMC-informed.
    """
    hold = fit_distance_density(d_px_holdout)
    nb = int(near_band)                             # near band covers d < nb
    edges = np.arange(0.0, nb + 1.0, 1.0)           # nb 1-px bins: [0,1),...,[nb-1,nb)
    sgmc_hist, _ = np.histogram(d_px_sgmc, bins=edges)
    sgmc_hist = sgmc_hist.astype(np.float64)
    sgmc_dens = sgmc_hist / max(sgmc_hist.sum(), 1.0)
    # continuity at the band edge: scale SGMC so its last near bin matches the
    # holdout's first measured bin (d in [nb, nb+1))
    hold_edge = hold["density"][nb]
    sgmc_edge = sgmc_dens[-1]
    scale = hold_edge / sgmc_edge if sgmc_edge > 0 else 1.0
    density = hold["density"].copy()
    density[:nb] = sgmc_dens * scale
    return density


def fit_azimuth_density(phi_deg: np.ndarray, bin_deg: float = 15.0) -> dict:
    """Empirical offset-azimuth density g(phi), phi in [0,90] (0=along strike)."""
    v = phi_deg[~np.isnan(phi_deg)]
    hist, edges = np.histogram(v, bins=np.arange(0.0, 90.0 + bin_deg, bin_deg))
    hist = hist.astype(np.float64)
    density = hist / max(hist.sum(), 1.0)
    return dict(hist=hist, edges=edges, density=density,
                median=float(np.median(v)), n=int(v.size))


def fit_tip_weight(u: np.ndarray) -> dict:
    """Along-strike position weight: beyond-tip (extension) vs inside (strand)."""
    inside = (u >= 0) & (u <= 1)
    n_in = int(inside.sum())
    n_out = int((~inside & ~np.isnan(u)).sum())
    # weight per pixel: inside pixels share mass n_in, beyond-tip share n_out
    return dict(frac_inside=float(n_in / max(n_in + n_out, 1)),
                frac_beyond_tip=float(n_out / max(n_in + n_out, 1)),
                w_inside=1.0, w_beyond_tip=1.0,
                note="per-pixel weights; both classes are kept, the split is reported")


def fit_length_scaling(d_px: np.ndarray, L_px: np.ndarray,
                       bins=((0, 5), (5, 15), (15, 40), (40, 100), (100, np.inf))
                       ) -> dict:
    """Displacement-proxy scaling: median zone distance per nearest-segment
    length bin.  Longer faults carry wider damage zones (Savage & Brodsky
    2011) -- the SCALING is fitted, not assumed."""
    out = {}
    medians = []
    for lo, hi in bins:
        sel = (L_px >= lo) & (L_px < hi)
        if sel.sum() < 30:
            continue
        med = float(np.median(d_px[sel]))
        out[f"{lo}-{hi if np.isfinite(hi) else 'inf'}"] = dict(
            n=int(sel.sum()), median_d_px=med)
        medians.append(med)
    base = float(np.median(d_px))
    for k in out:
        out[k]["scale_vs_median"] = out[k]["median_d_px"] / base if base > 0 else 1.0
    return dict(bins=out, base_median_d_px=base)


def length_scale_lookup(L_px: np.ndarray, scaling: dict) -> np.ndarray:
    """Per-pixel distance-axis scale s(L) from the fitted length scaling."""
    edges = [(0, 5), (5, 15), (15, 40), (40, 100), (100, np.inf)]
    s = np.ones_like(L_px, dtype=np.float64)
    for lo, hi in edges:
        key = f"{lo}-{hi if np.isfinite(hi) else 'inf'}"
        if key not in scaling["bins"]:
            continue
        sel = (L_px >= lo) & (L_px < hi)
        sc = scaling["bins"][key]["scale_vs_median"]
        s[sel] = max(sc, 0.2)
    return s


def density_at(d_px: np.ndarray, fit: dict, near_density: np.ndarray | None = None) -> np.ndarray:
    """Evaluate the fitted radial density at arbitrary distances (0 outside).

    ``near_density`` (same 1-px binning) replaces the density inside the
    d < 3 px band that the holdout's truth buffer removed."""
    edges = fit["edges"]
    idx = np.clip(np.digitize(d_px, edges) - 1, 0, len(fit["density"]) - 1)
    out = fit["density"][idx]
    out = np.where((d_px >= 0) & (d_px <= edges[-1]), out, 0.0)
    if near_density is not None:
        sel = d_px < NEAR_BAND_PX
        out = np.where(sel, near_density[idx], out)
    return out


def azimuth_at(phi_deg: np.ndarray, fit: dict) -> np.ndarray:
    """Evaluate the fitted azimuth density (NaN -> 0 weight)."""
    edges = fit["edges"]
    v = np.nan_to_num(phi_deg, nan=-1.0)
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(fit["density"]) - 1)
    out = fit["density"][idx]
    return np.where(v >= 0, out, 0.0)
