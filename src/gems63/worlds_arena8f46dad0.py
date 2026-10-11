"""Label-world generator for lane 63 (CVaR robust selection).

The organiser can add, remove or change Phase-2 faults relative to anything
we validated on, and catalogue geometry carries offsets of up to about 400 m
(4 px at the 100 m grid) relative to the geophysical sources used to draw it
(prompt-cited Hermant et al. measurement; see evidence/calibration.json for
the anchors used here).  A *label world* is therefore one plausible scored
truth obtained from the withheld holdout segments by:

* rigid shifts of whole withheld components by up to +/- 4 px, with a
  calibrated magnitude distribution (sigma about 1.7 px, capped at 4 px);
* omission of a random 10-30 % of the withheld component pixel mass;
* addition of decoy segments replacing each omitted component, drawn at the
  observed component-size distribution and placed inside the scored domain.

Worlds are generated per holdout cell; the world INDEX is joint across cells
(world ``w`` uses draw ``w`` everywhere), so pooled per-world DTI is a score
of one coherent "Phase-2 truth".  Worlds are candidate-independent: they are
generated once and re-used for every candidate raster or policy.

Evidence class: these are WORLD-DTI perturbations of the hide-and-recover
holdout.  They are NOT organiser scores and NOT projections of leaderboard
values.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from gems57.metric import kernel

STRUCT3 = np.ones((3, 3), bool)

# Calibrated shift prior (see evidence/calibration.json):
#   P(no shift) = 0.30 ; magnitudes 1..4 px with decreasing weights;
#   direction uniform over the 8 lattice directions.
P_NO_SHIFT = 0.30
MAG_WEIGHTS = np.array([0.30, 0.27, 0.24, 0.19], dtype=np.float64)  # |d| = 1..4 px
DIRS8 = np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dy, dx) != (0, 0)], dtype=np.int64)
DROP_RANGE = (0.10, 0.30)
MASTER_SEED = 63


@dataclass(frozen=True)
class WorldCell:
    """One world for one holdout cell, stored as crop-local truth indices."""
    key: str
    world: int
    truth_yx: tuple[np.ndarray, np.ndarray]   # indices within the cell crop
    n_truth: int
    n_kept_components: int
    n_dropped_components: int
    n_decoys: int
    dropped_mass: int
    shifted_mass: int

    @property
    def has_truth(self) -> bool:
        return self.n_truth > 0


def _component_shift_prior(rng: np.random.Generator, n: int):
    """Per-component shift vectors drawn from the calibrated prior."""
    dyx = np.zeros((n, 2), dtype=np.int64)
    moved = rng.random(n) >= P_NO_SHIFT
    n_mov = int(moved.sum())
    if n_mov:
        mag = 1 + rng.choice(len(MAG_WEIGHTS), size=n_mov, p=MAG_WEIGHTS)
        d8 = DIRS8[rng.integers(0, len(DIRS8), size=n_mov)]
        # scale the unit direction to the requested magnitude (round)
        scale = mag / np.maximum(
            np.abs(d8).max(axis=1), 1)
        dyx[moved] = np.round(d8 * scale[:, None]).astype(np.int64)
        # enforce the 4 px cap componentwise
        dyx[moved] = np.clip(dyx[moved], -4, 4)
    return dyx


def build_worlds_for_cell(cell, hidden_full: np.ndarray, catalogue_full: np.ndarray,
                          n_worlds: int, seed: int = MASTER_SEED) -> list[WorldCell]:
    """Generate ``n_worlds`` plausible truths for one holdout cell.

    ``hidden_full`` is the full-grid mask of withheld pixels for this cell and
    ``catalogue_full`` the full mapped catalogue.  Truth is finally intersected
    with ``cell.active`` (the crop-local scored domain), exactly as the shared
    evaluator does.
    """
    bbox = cell.bbox
    active_crop = cell.active
    hidden_crop = hidden_full[bbox]
    comp_crop, n_comp = ndi.label(hidden_crop, STRUCT3)
    if n_comp == 0:
        return [WorldCell(cell.key, w, (np.zeros(0, np.int64), np.zeros(0, np.int64)),
                          0, 0, 0, 0, 0) for w in range(n_worlds)]
    sizes = np.bincount(comp_crop.ravel(), minlength=n_comp + 1)[1:]
    total_mass = int(sizes.sum())
    comp_ids = np.arange(1, n_comp + 1)

    worlds: list[WorldCell] = []
    key_digest = zlib.crc32(cell.key.encode()) % (10 ** 6)  # deterministic across runs
    for w in range(n_worlds):
        rng = np.random.default_rng(
            1_000_003 * seed + 9176 * key_digest + 17 * w)
        # --- omissions: drop whole components totalling 10-30 % of the mass
        drop_frac = rng.uniform(*DROP_RANGE)
        target = drop_frac * total_mass
        perm = rng.permutation(n_comp)
        cum = np.cumsum(sizes[perm])
        k_drop = int(np.searchsorted(cum, target)) + 1
        dropped = set(int(comp_ids[perm[i]]) for i in range(min(k_drop, n_comp)))
        dropped_mass = int(sizes[perm[: min(k_drop, n_comp)]].sum())

        # --- shifts of kept components
        shifts = _component_shift_prior(rng, n_comp)

        kept_mask = np.isin(comp_crop, [c for c in comp_ids if c not in dropped])
        shifted = np.zeros_like(kept_mask)
        ys, xs = np.nonzero(kept_mask)
        ids = comp_crop[ys, xs] - 1
        dy, dx = shifts[ids].T
        ny = ys + dy
        nx = xs + dx
        hgt, wdt = shifted.shape
        ok = (ny >= 0) & (ny < hgt) & (nx >= 0) & (nx < wdt)
        shifted[ny[ok], nx[ok]] = True

        # --- decoys: one per dropped component, same size, random placement,
        # orientation from the kept components' principal axes
        n_decoys = 0
        decoy_mask = np.zeros_like(shifted)
        for cid in dropped:
            size_c = int(sizes[cid - 1])
            length = max(2, size_c)   # catalogues are ~1 px wide
            ori = rng.random()
            # orientation sampled uniformly over [0, 180)
            ang = ori * np.pi
            uy, ux = np.sin(ang), np.cos(ang)   # (row, col) direction
            placed = False
            for _try in range(6):
                cy = rng.integers(2, max(3, hgt - 2))
                cx = rng.integers(2, max(3, wdt - 2))
                half = length // 2
                yy = np.round(cy + (np.arange(length) - half) * uy).astype(np.int64)
                xx = np.round(cx + (np.arange(length) - half) * ux).astype(np.int64)
                okk = (yy >= 0) & (yy < hgt) & (xx >= 0) & (xx < wdt)
                if okk.sum() < max(2, length // 2):
                    continue
                seg = np.zeros_like(shifted)
                seg[yy[okk], xx[okk]] = True
                # decoys must land in the scored domain, off visible faults
                if (seg & active_crop).sum() >= max(2, length // 2):
                    decoy_mask |= seg
                    n_decoys += 1
                    placed = True
                    break
            del placed

        truth_crop = (shifted | decoy_mask) & active_crop
        # never score truth on visible catalogue: active already excludes it
        ty, tx = np.nonzero(truth_crop)
        worlds.append(WorldCell(
            key=cell.key, world=w,
            truth_yx=(ty.astype(np.int64), tx.astype(np.int64)),
            n_truth=int(ty.size),
            n_kept_components=int(n_comp - len(dropped)),
            n_dropped_components=int(len(dropped)),
            n_decoys=n_decoys,
            dropped_mass=dropped_mass,
            shifted_mass=int(sizes[np.any(shifts != 0, axis=1)].sum()),
        ))
    return worlds


def world_summary(worlds_by_cell: dict) -> dict:
    """Bookkeeping for the evidence record."""
    rows = []
    for key, worlds in worlds_by_cell.items():
        rows.append(dict(
            cell=key, n_worlds=len(worlds),
            mean_truth_px=float(np.mean([w.n_truth for w in worlds])) if worlds else 0.0,
            mean_dropped_components=float(np.mean([w.n_dropped_components for w in worlds])) if worlds else 0.0,
            mean_decoys=float(np.mean([w.n_decoys for w in worlds])) if worlds else 0.0,
            mean_dropped_mass=float(np.mean([w.dropped_mass for w in worlds])) if worlds else 0.0,
            mean_shifted_mass=float(np.mean([w.shifted_mass for w in worlds])) if worlds else 0.0,
        ))
    return dict(
        n_cells=len(rows), cells=rows,
        calibration=dict(p_no_shift=P_NO_SHIFT,
                         magnitude_weights=MAG_WEIGHTS.tolist(),
                         max_shift_px=4, max_shift_m=400,
                         drop_range=list(DROP_RANGE),
                         decoy_policy="one decoy per dropped component, same pixel size, uniform placement in active domain",
                         anchors=[
                             "prompt-cited Hermant et al. catalogue offsets up to 400 m (= 4 px at the 100 m grid)",
                             "organizer statement that Phase-2 faults may be added, removed or changed (competition problem page)",
                             "60GEMSDOE evidence/h33_holdout.json drift-corrected vs raw holdout gap (instrument-specific)",
                         ]))
