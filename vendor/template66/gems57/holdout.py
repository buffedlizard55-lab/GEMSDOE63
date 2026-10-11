"""Hide-and-recover spatially-blocked holdout.

Design (parallel-run protocol rule 2)
-------------------------------------
* The mapped catalogue is split into **whole segments** (branches between
  intersections, cut into <= 12 px chunks -- see :func:`gems57.network.segments`).
* The footprint is split into four spatial quadrants (NW, NE, SW, SE) at the
  median footprint row/col.  Fold *k* withholds segments inside quadrant *k*,
  and ``DOMAIN_ERODE`` (12 px = 1.2 km) keeps the scored domain away from the
  fold boundary, which is what provides the spatial separation between the
  withheld strands and the catalogue context used to build the features.
* Only segments lying wholly inside the scored domain are eligible for
  withholding, so truth is never clipped by the domain edge.
* Every catalogue-derived feature is computed from the **visible** mask only.
* Visible fault pixels are masked out of scoring pixel-exactly (``known``),
  matching the organiser's statement that a dot near a known trace but far from
  any new-fault pixel is fully penalised (forum thread 11516).
* Score: pooled DTI, alpha = 0.2, beta = 0.8, 300 m (3 px) triangular kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .grid import Grid
from .metric import ALPHA, BETA, EPS, kernel
from .network import STRUCT3, component_detached, segments

FOLD_NAMES = ("NW", "NE", "SW", "SE")
DOMAIN_ERODE = 12     # 1.2 km boundary erosion; withheld segments must lie wholly inside
DETACH_PX = 4         # 400 m separation defining the conservative "detached" subset
HIDE_FRAC = 0.20      # fraction of the fold's catalogue pixels withheld
MAX_SEG_PX = 12


@dataclass
class Cell:
    """One fold cell.  ``active`` and ``k_dg`` are cropped to ``bbox`` to keep the
    whole context resident in memory; the bbox carries 6 px of padding so the
    3 px kernel sees the same neighbourhood it would on the full grid."""
    key: str
    fold: int
    seed: int
    mode: str
    bbox: tuple[slice, slice]
    active: np.ndarray                  # bool, scored domain minus visible faults (cropped)
    truth_yx: tuple[np.ndarray, np.ndarray]   # indices *within the crop*
    k_dg: np.ndarray                    # float32 kernel(distance to truth) (cropped)
    n_truth: int


@dataclass
class HoldoutContext:
    grid: Grid
    quad: np.ndarray
    cells: list[Cell]
    hidden_by_cell: dict                # key -> full-grid bool mask of withheld pixels

    def visible(self, key: str) -> np.ndarray:
        """Visible catalogue for a cell: everything mapped except what is withheld."""
        return self.grid.catalogue & ~self.hidden_by_cell[key]

    def cell(self, key: str) -> Cell:
        for c in self.cells:
            if c.key == key:
                return c
        raise KeyError(key)

    def cells_of(self, mode: str) -> list[Cell]:
        return [c for c in self.cells if c.mode == mode]


def quadrant_ids(footprint: np.ndarray) -> np.ndarray:
    fp = np.asarray(footprint, bool)
    yy, xx = np.nonzero(fp)
    ym, xm = int(np.median(yy)), int(np.median(xx))
    gy, gx = np.ogrid[: fp.shape[0], : fp.shape[1]]
    q = np.full(fp.shape, -1, np.int8)
    q[(gy < ym) & (gx < xm) & fp] = 0
    q[(gy < ym) & (gx >= xm) & fp] = 1
    q[(gy >= ym) & (gx < xm) & fp] = 2
    q[(gy >= ym) & (gx >= xm) & fp] = 3
    return q


def _pick_segments(rng, sizes: np.ndarray, ids: np.ndarray, target_px: float) -> np.ndarray:
    """Random whole segments until their cumulative size reaches ``target_px``."""
    if ids.size == 0:
        return ids
    perm = rng.permutation(ids)
    cum = np.cumsum(sizes[perm])
    k = int(np.searchsorted(cum, target_px)) + 1
    return perm[: min(k, perm.size)]


def build_holdout(grid: Grid, hide_frac: float = HIDE_FRAC,
                  seeds: tuple[int, ...] = (20, 21),
                  modes: tuple[str, ...] = ("all", "detached"),
                  detach_px: int = DETACH_PX) -> HoldoutContext:
    """Build the fold cells once; re-usable for any number of candidates.

    ``mode="all"``      -- every segment lying wholly inside the fold domain is
                           eligible for withholding.  Visible context is the
                           whole remaining catalogue, including traces that are
                           continuous with a withheld one.  This is the primary
                           instrument.
    ``mode="detached"`` -- only segments that lie at least ``detach_px`` from
                           every other segment are eligible.  These are the
                           withheld strands whose recovery does *not* benefit
                           from a visible trace running into them, so this is the
                           conservative reading and the closer analogue to a
                           genuinely unmapped splay.
    """
    cat = grid.catalogue
    quad = quadrant_ids(grid.footprint)
    seg, n_seg = segments(cat, max_len_px=MAX_SEG_PX)[:2]
    sizes = np.bincount(seg.ravel(), minlength=n_seg + 1)
    all_ids = np.arange(1, n_seg + 1)

    hidden_by_cell: dict[str, np.ndarray] = {}
    cells: list[Cell] = []

    for seed in seeds:
        for fold, qname in enumerate(FOLD_NAMES):
            q = quad == fold
            rows = np.flatnonzero(q.any(axis=1))
            cols = np.flatnonzero(q.any(axis=0))
            # pad the bbox by 6 px so the 3-px kernel sees the same neighbourhood
            # it would on the full grid
            sl = (slice(max(0, rows[0] - 6), min(grid.height, rows[-1] + 7)),
                  slice(max(0, cols[0] - 6), min(grid.width, cols[-1] + 7)))
            domain = ndi.binary_erosion(q, iterations=DOMAIN_ERODE) & grid.footprint
            # only segments wholly inside the domain can be withheld, so that
            # truth is never clipped by the domain boundary
            clipped = np.isin(all_ids, np.unique(seg[~domain & cat]))
            eligible_base = all_ids[~clipped]

            if "detached" in modes:
                # "detached" = the segment belongs to a connected fault component
                # lying at least ``detach_px`` away from every other component.
                # Such strands are not simply the continuation of a visible
                # trace, so recovering them is the harder, live-analogue problem.
                comp, is_det = component_detached(cat, detach_px)
                detached_ids = np.isin(all_ids, np.unique(seg[comp[is_det[comp]]]))
                del comp, is_det
            else:
                detached_ids = None

            for mode in modes:
                pool = eligible_base if mode == "all" else eligible_base[detached_ids[eligible_base]]
                rng = np.random.default_rng(10_000 * (seed + 1) + 7 * fold + (0 if mode == "all" else 1))
                target = hide_frac * float((cat & q & domain).sum())
                hidden_ids = _pick_segments(rng, sizes, pool, target)
                hidden = np.isin(seg, hidden_ids) & grid.footprint
                visible = cat & ~hidden

                key = f"draw{seed}_fold{qname}_{mode}"
                hidden_by_cell[key] = hidden

                active = domain & ~visible
                truth = hidden & domain & active
                tyx_full = np.nonzero(truth)
                k_full = kernel(ndi.distance_transform_edt(~truth))
                # crop to the padded quadrant bbox
                a_sub = active[sl].copy()
                k_sub = k_full[sl].astype(np.float32)
                t_sub = truth[sl]
                tyx = np.nonzero(t_sub)
                cells.append(Cell(key=key, fold=fold, seed=seed, mode=mode, bbox=sl,
                                  active=a_sub, truth_yx=tyx, k_dg=k_sub,
                                  n_truth=int(tyx[0].size)))
                del hidden, visible, active, truth, tyx_full, k_full, a_sub, k_sub, t_sub

    return HoldoutContext(grid=grid, quad=quad, cells=cells,
                          hidden_by_cell=hidden_by_cell)


def score_cell(cell: Cell, emitted: np.ndarray) -> dict:
    """Exact DTI of a binary emission inside one fold cell.

    ``emitted`` may be either the full grid or an array already cropped to
    ``cell.bbox``.
    """
    if emitted.shape == cell.active.shape:
        p = emitted & cell.active
    else:
        p = emitted[cell.bbox] & cell.active
    n_emit = int(p.sum())
    n_t = cell.n_truth
    if n_t == 0 or n_emit == 0:
        return dict(dti=0.0, coverage=0.0, tp=0.0, fp=float(n_emit), fn=float(n_t),
                    n_truth=n_t, n_emitted=n_emit)
    dp = ndi.distance_transform_edt(~p)
    tp = float(kernel(dp[cell.truth_yx]).sum())
    fn = float(n_t) - tp
    fp = float((1.0 - cell.k_dg[p]).sum())
    return dict(dti=float(tp / (tp + ALPHA * fp + BETA * fn + EPS)),
                coverage=float(tp / n_t), tp=tp, fp=fp, fn=fn,
                n_truth=n_t, n_emitted=n_emit)


def evaluate(emitted: np.ndarray, ctx: HoldoutContext, cell_mode: str = "") -> dict:
    """Per-fold and pooled DTI for a binary emission mask."""
    emitted = np.asarray(emitted, bool) & ctx.grid.footprint
    per_fold = {}
    tp = fp = fn = n_truth = n_emit = 0.0
    for cell in ctx.cells:
        if cell_mode and cell.mode != cell_mode:
            continue
        r = score_cell(cell, emitted)
        per_fold[cell.key] = r
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        n_truth += r["n_truth"]; n_emit += r["n_emitted"]
    pooled = float(tp / (tp + ALPHA * fp + BETA * fn + EPS)) if (tp + fp + fn) > 0 else 0.0
    by_draw: dict[str, list[float]] = {}
    by_quad: dict[str, list[float]] = {}
    for k, r in per_fold.items():
        d, q, _m = k.split("_")
        by_draw.setdefault(d, []).append(r["dti"])
        by_quad.setdefault(q, []).append(r["dti"])
    return {
        "emitted_pixels": int(emitted.sum()),
        "on_visible_catalogue": int((emitted & ctx.grid.catalogue).sum()),
        "pooled_dti": pooled,
        "pooled_coverage": float(tp / n_truth) if n_truth else 0.0,
        "pooled_tp": tp, "pooled_fp": fp, "pooled_fn": fn, "pooled_n_truth": int(n_truth),
        "mean_fold_dti": float(np.mean([r["dti"] for r in per_fold.values()])),
        "std_fold_dti": float(np.std([r["dti"] for r in per_fold.values()])),
        "per_fold_dti": {k: r["dti"] for k, r in per_fold.items()},
        "per_draw_dti": {k: float(np.mean(v)) for k, v in sorted(by_draw.items())},
        "per_quadrant_dti": {k: float(np.mean(v)) for k, v in sorted(by_quad.items())},
        "folds": per_fold,
    }


def wilson_ci(k_success_mass: float, n_mass: float, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (used for coverage CIs)."""
    if n_mass <= 0:
        return (0.0, 0.0)
    p = k_success_mass / n_mass
    denom = 1.0 + z * z / n_mass
    centre = (p + z * z / (2 * n_mass)) / denom
    half = z * np.sqrt(p * (1 - p) / n_mass + z * z / (4 * n_mass * n_mass)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def score(prediction, fold, valid, restrict_to_region=True, extra=False):
    """Compatibility adapter for the shared template's dictionary-fold callers.

    v2 fixes the missing adapter; official arithmetic remains in metric.py.
    """
    from .metric import dti_exact
    region = fold['region'] if restrict_to_region else np.ones_like(valid, bool)
    known = fold.get('masked_known', fold['visible'])
    r = dti_exact(prediction, fold['truth'], valid=np.asarray(valid, bool) & region, known=known)
    return {**r, 'tpw': r['tp'], 'fpw': r['fp'], 'fnw': r['fn']}


def buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3,
                            boundary_px=DOMAIN_ERODE):
    """A repaired label-blind quadrant draw with REAL catalogue context buffering.

    Whole original 8-connected components are withheld (no <=12-pixel chunks).
    Only components wholly inside a label-blind eroded quadrant are eligible.
    The same globally hidden draw supplies every outer LOQO model, so a tested
    component cannot reappear in a training quadrant's catalogue features.
    Buffer-removal is for FEATURE context, not a truth-shaped evaluation halo.
    Other catalogue pixels in that context collar remain pixel-exactly masked
    from scoring; they are not invented negatives or silently clipped truth.
    """
    if not 0 < hide_frac < 1 or buffer_px < 0 or boundary_px < buffer_px:
        raise ValueError('invalid holdout fraction/buffer')
    cat = np.asarray(grid.catalogue, bool)
    quad = quadrant_ids(grid.footprint)
    comp, count = ndi.label(cat, STRUCT3)
    sizes = np.bincount(comp.ravel(), minlength=count+1)
    hidden = np.zeros(cat.shape, bool)
    regions, receipts = [], []
    for q, name in enumerate(FOLD_NAMES):
        region = ndi.binary_erosion(quad == q, iterations=boundary_px) if boundary_px else quad == q
        region &= grid.footprint
        candidate = np.unique(comp[cat & region])
        clipped = np.unique(comp[cat & ~region])
        eligible = np.setdiff1d(candidate[candidate > 0], clipped, assume_unique=True)
        rng = np.random.default_rng(seed * 10_000 + q)
        chosen = _pick_segments(rng, sizes, eligible, hide_frac * float((cat & region).sum()))
        withheld = np.isin(comp, chosen) & cat
        if (withheld & ~region).any():
            raise AssertionError('a withheld component was clipped')
        hidden |= withheld
        regions.append(region)
        receipts.append(dict(fold=name, seed=seed, held_components=int(len(chosen)),
            withheld_positive_pixels=int(withheld.sum()), eligible_components=int(len(eligible)),
            boundary_buffer_px=boundary_px, feature_buffer_px=buffer_px))
    if not hidden.any():
        raise ValueError('no eligible withheld components')
    collar = ndi.distance_transform_edt(~hidden) <= buffer_px
    visible = cat & ~collar
    masked_known = cat & ~hidden
    if not visible.any():
        raise ValueError('no visible catalogue context')
    if (visible & hidden).any() or np.any(visible & collar):
        raise AssertionError('withheld truth/context buffer reached a feature')
    minimum = float(ndi.distance_transform_edt(~visible)[hidden].min())
    if minimum <= buffer_px:
        raise AssertionError('context buffer was not honored')
    folds = [dict(fold=i, name=FOLD_NAMES[i], region=regions[i], truth=hidden & regions[i],
                  visible=visible, masked_known=masked_known, hidden_all=hidden,
                  receipt={**receipts[i], 'nearest_visible_to_truth_px': minimum,
                           'split_version': 'buffered-whole-components-loqo-v2',
                           'evaluation_region_label_blind': True}) for i in range(4)]
    return folds, quad
