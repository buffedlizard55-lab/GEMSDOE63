"""Hide-and-recover spatially-blocked holdout cells for GEMSDOE63.

PROVENANCE: cell construction adapted from the shared template
buffedlizard55-lab/GEMSDOE32 src/gems32/holdout.py (draws 20/21 x 4 quadrants,
hide_frac 0.20, 1.5 km collar, 1.2 km domain erosion), itself implementing the
team's shared hide-and-recover protocol: withhold whole fault segments with a
buffer, derive catalogue-based features only from visible faults, mask visible
faults pixel-exactly, score pooled DTI (alpha 0.2, beta 0.8, 300 m triangular
kernel). Kept identical so instruments are comparable across sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion, label

FOLD_NAMES = ("NW", "NE", "SW", "SE")
FOLDS = (0, 1, 2, 3)
DRAWS = (20, 21)
COLLAR_PX = 15
DOMAIN_ERODE = 12
NEAR_LABEL_PX = 3


def read_binary(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1)
    return np.isfinite(a) & (a > 0)


def load_grid(ddir: Path):
    """Load footprint, labels (catalogue), and SGMC off-catalogue surrogate faults."""
    with rasterio.open(ddir / "sample_submission.tif") as ds:
        foot = np.isfinite(ds.read(1))
        transform = ds.transform
        crs = ds.crs
    labels = read_binary(ddir / "labels.tif") & foot
    with rasterio.open(ddir / "external" / "derived_sgmc_faults_100m_u8.tif") as ds:
        sgm = ds.read(1) > 0
    near = binary_dilation(labels, iterations=NEAR_LABEL_PX)
    sgmc_off = sgm & ~labels & ~near & foot
    return foot, labels, sgmc_off, near, transform, crs


def quadrant_ids(footprint: np.ndarray) -> np.ndarray:
    footprint = np.asarray(footprint, bool)
    yy, xx = np.nonzero(footprint)
    ym, xm = int(np.median(yy)), int(np.median(xx))
    H, W = footprint.shape
    gy, gx = np.ogrid[:H, :W]
    q = np.full((H, W), -1, np.int8)
    q[(gy < ym) & (gx < xm) & footprint] = 0
    q[(gy < ym) & (gx >= xm) & footprint] = 1
    q[(gy >= ym) & (gx < xm) & footprint] = 2
    q[(gy >= ym) & (gx >= xm) & footprint] = 3
    return q


@dataclass
class Cell:
    key: str
    fold: int
    seed: int
    bbox: tuple
    domain_sub: np.ndarray          # eroded fold domain within bbox
    active_sub: np.ndarray          # domain & ~visible within bbox
    visible_sub: np.ndarray         # visible catalogue within bbox
    hidden_comp_coords: list        # list of (rows, cols) per hidden component, bbox coords
    fold_comp_shapes: list          # catalogue component shapes in this fold (for decoy sampling)
    n_hidden_truth: int             # unperturbed hidden truth pixels inside active
    n_fold_components: int = 0      # observed catalogue components in this fold (decoy density)
    n_domain_px: int = 0            # domain area in pixels (decoy density)


def _pick_components(rng, comp_size: np.ndarray, ids: np.ndarray, target_px: float) -> np.ndarray:
    if ids.size == 0:
        return ids
    perm = rng.permutation(ids)
    cum = np.cumsum(comp_size[perm])
    k = int(np.searchsorted(cum, target_px)) + 1
    return perm[: min(k, perm.size)]


def build_cells(foot: np.ndarray, labels: np.ndarray, hide_frac: float = 0.20) -> list:
    """Construct the 8 hide-and-recover cells; identical draws to the template."""
    quad = quadrant_ids(foot)
    comp, n_comp = label(labels, structure=np.ones((3, 3), int))
    comp_size = np.bincount(comp.ravel(), minlength=n_comp + 1)
    all_ids = np.arange(1, n_comp + 1)

    quad_bboxes: dict = {}
    collars: dict = {}
    domains: dict = {}
    for fold in FOLDS:
        q = quad == fold
        rows = np.flatnonzero(q.any(axis=1))
        cols = np.flatnonzero(q.any(axis=0))
        r0 = max(0, int(rows[0]) - 6)
        r1 = min(foot.shape[0], int(rows[-1]) + 7)
        c0 = max(0, int(cols[0]) - 6)
        c1 = min(foot.shape[1], int(cols[-1]) + 7)
        quad_bboxes[fold] = (slice(r0, r1), slice(c0, c1))
        collars[fold] = binary_dilation(q, structure=np.ones((3, 3), bool), iterations=COLLAR_PX) & foot
        domains[fold] = binary_erosion(q, iterations=DOMAIN_ERODE)

    # per-fold catalogue component shapes (for decoy sampling in the world generator)
    comp_coords: dict = {}

    cells: list = []
    for seed in DRAWS:
        for fold in FOLDS:
            rng = np.random.default_rng(10_000 * (seed + 1) + fold)
            q = quad == fold
            collar = collars[fold]
            domain = domains[fold]
            touch_collar = np.isin(all_ids, np.unique(comp[collar & labels]))
            in_test = np.isin(all_ids, np.unique(comp[q & labels]))
            test_ids = all_ids[in_test]
            train_ids = all_ids[~touch_collar]
            hid_test = _pick_components(rng, comp_size, test_ids, hide_frac * float((labels & q).sum()))
            hid_train = _pick_components(rng, comp_size, train_ids, hide_frac * float(comp_size[train_ids].sum()))
            hidden_full = np.isin(comp, hid_test)
            hidden_train = np.isin(comp, hid_train)
            visible = labels & ~hidden_full & ~hidden_train
            hidden = labels & ~visible

            sl = quad_bboxes[fold]
            r0, c0 = sl[0].start, sl[1].start
            domain_sub = domain[sl]
            active_sub = domain_sub & ~visible[sl]
            visible_sub = visible[sl]

            comp_sub = comp[sl]
            hidden_ids_cell = np.unique(comp_sub[hidden[sl]])
            hidden_ids_cell = hidden_ids_cell[hidden_ids_cell > 0]
            hidden_comp_coords = []
            for cid in hidden_ids_cell:
                rr, cc = np.nonzero(comp_sub == cid)
                hidden_comp_coords.append((rr.astype(np.int32), cc.astype(np.int32)))

            if fold not in comp_coords:
                shapes = []
                fold_ids = np.unique(comp[q & labels])
                fold_ids = fold_ids[fold_ids > 0]
                n_fold_components = int(fold_ids.size)
                for cid in fold_ids[:400]:
                    rr, cc = np.nonzero(comp == cid)
                    rr = rr - rr.min()
                    cc = cc - cc.min()
                    shapes.append((rr.astype(np.int16), cc.astype(np.int16), int(rr.size)))
                comp_coords[fold] = shapes

            truth_sub = hidden[sl] & active_sub
            cells.append(
                Cell(
                    key=f"draw{seed}_fold{fold}",
                    fold=fold,
                    seed=seed,
                    bbox=sl,
                    domain_sub=domain_sub,
                    active_sub=active_sub,
                    visible_sub=visible_sub,
                    hidden_comp_coords=hidden_comp_coords,
                    fold_comp_shapes=comp_coords[fold],
                    n_hidden_truth=int(truth_sub.sum()),
                    n_fold_components=n_fold_components,
                    n_domain_px=int(domain_sub.sum()),
                )
            )
    return cells
