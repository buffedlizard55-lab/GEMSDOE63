"""Corroboration-weighted posterior surface for GEMSDOE63.

Builds a continuous prior field from four independently-scored prior surfaces,
each a soft [0,1] raster produced by earlier sessions from different feature
families (multiline/corroborated geophysics in h19-*, topo-geophys ridges in
h16-1, an ensemble-of-12 in ens12). The d28 forensic autopsy
(buff-edlizard55-lab/GEMSDOE32 evidence/d28_forensic_autopsy.json) identified
"1,732 uncorroborated off-scarp speckles" as a measurable defect of the 0.26
emission, so the posterior here multiplies a weighted blend by a corroboration
factor: a pixel only scores well when independent physics families agree.

Weights are derived INSIDE this run from each surface's solo baseline
holdout DTI (the protocol-sanctioned instrument), never from leaderboard scores.
"""

from __future__ import annotations

import numpy as np


def load_soft(path, foot: np.ndarray) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float32)
    a = np.where(np.isfinite(a), a, 0.0)
    a = np.clip(a, 0.0, 1.0)
    a[~foot] = 0.0
    return a


def build_posterior(surfaces: dict, weights: dict, foot: np.ndarray,
                    collar_mask: np.ndarray) -> dict:
    """surfaces: name -> [0,1] array. weights: name -> float (normalized inside).

    P(x) = (sum_i w_i s_i(x)) * (0.25 + 0.75 * C(x)/n)
    with C(x) = number of surfaces at or above their own positive median at x.
    """
    names = list(surfaces.keys())
    n = len(names)
    w = np.array([max(float(weights.get(k, 1.0)), 0.0) for k in names], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(n)
    w = w / w.sum()
    stack = [surfaces[k] for k in names]
    blend = np.zeros_like(stack[0], dtype=np.float32)
    C = np.zeros_like(stack[0], dtype=np.int8)
    thresholds = {}
    for i, k in enumerate(names):
        s = stack[i]
        pos = s[foot & (s > 0)]
        med = float(np.median(pos)) if pos.size else 0.5
        thresholds[k] = med
        blend += np.float32(w[i]) * s
        C += (s >= med).astype(np.int8)
    corr = 0.25 + 0.75 * (C.astype(np.float32) / float(n))
    P = (blend * corr).astype(np.float32)
    P[collar_mask] = 0.0
    P[~foot] = 0.0
    return {"P": P, "weights": {k: float(wi) for k, wi in zip(names, w)},
            "thresholds": thresholds, "corroboration": C}


def sgmc_augment(P: np.ndarray, sgmc_off: np.ndarray, allowed: np.ndarray,
                 sigma: float = 1.5, gamma: float = 0.5) -> np.ndarray:
    """Augment posterior with the SGMC off-catalogue surrogate-fault field,
    restricted to pixels where at least one input surface is nonzero
    (corroborated augmentation - the anti-speckle rule)."""
    from scipy.ndimage import gaussian_filter

    f = gaussian_filter(sgmc_off.astype(np.float32), sigma=sigma)
    if f.max() > 0:
        f = f / f.max()
    evid = (P > 0)
    aug = P + np.float32(gamma) * f.astype(np.float32) * evid.astype(np.float32)
    aug = np.clip(aug, 0.0, 1.0)
    aug[~allowed] = 0.0
    return aug.astype(np.float32)
