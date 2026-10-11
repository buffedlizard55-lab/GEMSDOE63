"""Emission policies: score map -> sparse binary dots (the emission family that scored best on the
public board in the owner's repos; see docs/research in GEMSDOE32), plus the catalogue-flank rule.

Policy theta = (family, budget q, spacing s, flank radius r):
  family  logit : logistic regression on the catalogue-free feature stack, trained on the
                  VISIBLE catalogue of the fold (buffered), out-of-fold for the holdout.
          ridge : catalogue-free lineness ensemble (no training, no labels).
  q       : budget = fraction of scored-domain pixels whose score is at or above the threshold
  s       : non-maximum suppression radius (px): dots are local maxima in a (2s+1) window
  r       : flank rule: drop dots within r px of VISIBLE catalogue (0 = off)
Dots are placed only in the scored domain (footprint and not known); value 1.0 at dots, 0.0 elsewhere
in the footprint, NaN outside (organiser format, see registry/grid_facts.json).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

from .grid import Grid, buffer_zone, clearance_zone

RIDGE_COLS = ["lineness_tmi_hg", "lineness_tilt_tc", "lineness_elev_slope"]


def ridge_score_vec(X: np.ndarray, names: list[str]) -> np.ndarray:
    cols = [names.index(c) for c in RIDGE_COLS]
    Z = X[:, cols].astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return Z.mean(1)


def logit_score_vec(X, names, g: Grid, visible: np.ndarray, withheld: np.ndarray, rng) -> np.ndarray:
    """Train on the visible catalogue minus a buffer around withheld truth; predict everywhere."""
    fp = g.footprint
    buf = buffer_zone(withheld) if withheld.any() else np.zeros_like(fp)
    train_pos = (visible & ~buf)[fp]
    clear = clearance_zone(g)
    train_neg_pool = (fp & ~clear & ~buf)[fp]
    pos_idx = np.flatnonzero(train_pos)
    neg_pool = np.flatnonzero(train_neg_pool)
    n_neg = min(200_000, neg_pool.size)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(pos_idx.size), np.zeros(n_neg)])
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    Z = (X - mu) / sd
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf.fit(Z[idx], y)
    return clf.predict_proba(Z)[:, 1]


def dots_from_score(score_grid: np.ndarray, active: np.ndarray, q: float, s: int) -> np.ndarray:
    vals = score_grid[active]
    thr = np.quantile(vals, 1.0 - q)
    cand = active & (score_grid >= thr)
    mx = ndimage.maximum_filter(np.where(active, score_grid, -np.inf), size=2 * s + 1, mode="constant", cval=-np.inf)
    return cand & (score_grid >= mx)


def flank_prune(dots: np.ndarray, visible: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not visible.any():
        return dots
    d = ndimage.distance_transform_edt(~visible)
    return dots & (d > r)


def policy_dots(score_grid, g: Grid, visible, q, s, r) -> np.ndarray:
    active = g.footprint & ~visible
    dots = dots_from_score(score_grid, active, q, s)
    return flank_prune(dots, visible, r)


def score_to_grid(vec: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    out = np.zeros(footprint.shape, dtype=np.float32)
    out[footprint] = vec
    return out
