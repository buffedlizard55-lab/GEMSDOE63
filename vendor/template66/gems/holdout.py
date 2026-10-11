"""Hide-and-recover holdout splits for fault labels.

Whole fault segments are assigned to spatially held blocks. Every held segment is excluded
from catalogue-derived features, including its portions outside the scored blocks; a buffer
then removes nearby visible faults. The score mask is restricted to lidar-covered positives
inside the selected blocks. If authoritative segment IDs are unavailable, 8-connected raster
components are a conservative proxy and must be disclosed as such.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _validate_fraction(value: float, name: str = "holdout_fraction") -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return value


def _segment_map(fault_mask: np.ndarray, segment_ids=None) -> tuple[np.ndarray, list[int], str]:
    """Return an integer segment map, its positive IDs, and provenance text."""
    if segment_ids is None:
        labels, count = ndimage.label(fault_mask, structure=np.ones((3, 3), dtype=int))
        return labels.astype(np.int64, copy=False), list(range(1, count + 1)), "8-connected components (proxy)"

    raw = np.asarray(segment_ids)
    if raw.shape != fault_mask.shape:
        raise ValueError(f"segment_ids shape {raw.shape} != fault_mask shape {fault_mask.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("segment_ids must be an integer raster")
    if np.any(raw[fault_mask] <= 0):
        raise ValueError("every positive fault pixel must have a positive segment ID")
    seg = np.zeros(fault_mask.shape, dtype=np.int64)
    seg[fault_mask] = raw[fault_mask].astype(np.int64, copy=False)
    ids = sorted(int(x) for x in np.unique(seg[fault_mask]))
    return seg, ids, "provided integer segment IDs"


def spatial_block_ids(shape, block_px: int) -> np.ndarray:
    """Integer ID per square block of ``block_px`` cells (row-major)."""
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must be a positive 2-D shape")
    if not isinstance(block_px, (int, np.integer)) or block_px <= 0:
        raise ValueError("block_px must be a positive integer")
    h, w = shape
    rows = np.arange(h) // block_px
    cols = np.arange(w) // block_px
    n_cols = int(cols.max()) + 1
    return rows[:, None] * n_cols + cols[None, :]


def split_fault_segments(fault_mask, holdout_fraction: float, buffer_px: int, seed: int,
                         segment_ids=None) -> dict:
    """Split a binary fault raster into visible and withheld whole segments.

    If ``segment_ids`` is omitted, segments are 8-connected components. When a held segment
    is selected, all of its pixels are withheld; nearby visible faults within ``buffer_px``
    are excluded from feature derivation and reported as ``buffer_dropped`` (not scored).
    """
    fm = np.asarray(fault_mask, dtype=bool)
    if fm.ndim != 2:
        raise ValueError("fault_mask must be 2-D")
    fraction = _validate_fraction(holdout_fraction)
    if not isinstance(buffer_px, (int, np.integer)) or buffer_px < 0:
        raise ValueError("buffer_px must be a non-negative integer")
    seg, ids, id_source = _segment_map(fm, segment_ids)
    rng = np.random.default_rng(seed)
    k = 0 if not ids or fraction == 0 else max(1, int(round(fraction * len(ids))))
    held_ids = rng.choice(np.asarray(ids, dtype=np.int64), size=k, replace=False) if k else np.array([], int)
    held_segment = fm & np.isin(seg, held_ids)
    holdout = held_segment.copy()
    visible_raw = fm & ~held_segment
    if held_segment.any() and buffer_px:
        buffer_zone = ndimage.distance_transform_edt(~held_segment) <= buffer_px
    else:
        buffer_zone = np.zeros_like(fm)
    visible = visible_raw & ~buffer_zone
    buffer_dropped = visible_raw & buffer_zone
    return {
        "visible": visible,
        "holdout": holdout,
        "held_segment": held_segment,
        "unscored_withheld": np.zeros_like(fm),
        "buffer_dropped": buffer_dropped,
        "held_segment_ids": sorted(int(x) for x in held_ids),
        "n_segments": len(ids),
        "segment_id_source": id_source,
    }


def spatial_block_holdout(fault_mask, lidar_mask, block_px: int, holdout_fraction: float,
                          buffer_px: int, seed: int, min_lidar_fraction: float = 0.9,
                          segment_ids=None) -> dict:
    """Withhold whole segments selected through lidar-covered spatial blocks.

    Eligible blocks meet ``min_lidar_fraction`` lidar coverage and contain at least one fault
    pixel that is itself lidar-covered. The selected block determines which segments are held;
    each such segment is then removed from visible faults in its entirety, even where it
    crosses outside the selected block. A buffer around those complete held segments removes
    nearby visible faults from feature derivation. ``holdout`` is the scoring mask: positive
    fault pixels inside selected blocks *and* lidar coverage. Positive cells in a held segment
    outside that scoring mask are returned as ``unscored_withheld`` and must not be scored.

    ``segment_ids`` should be authoritative integer segment labels where available. Without
    them, 8-connected components are used as a documented proxy; touching distinct faults may
    be merged, but a held component is never split between visible and holdout sets.
    """
    fm = np.asarray(fault_mask, dtype=bool)
    lm = np.asarray(lidar_mask, dtype=bool)
    if fm.ndim != 2 or lm.ndim != 2:
        raise ValueError("fault_mask and lidar_mask must be 2-D")
    if fm.shape != lm.shape:
        raise ValueError(f"fault mask {fm.shape} and lidar mask {lm.shape} differ")
    fraction = _validate_fraction(holdout_fraction)
    if not isinstance(buffer_px, (int, np.integer)) or buffer_px < 0:
        raise ValueError("buffer_px must be a non-negative integer")
    min_lidar_fraction = float(min_lidar_fraction)
    if not np.isfinite(min_lidar_fraction) or not 0.0 <= min_lidar_fraction <= 1.0:
        raise ValueError("min_lidar_fraction must be between 0 and 1 inclusive")

    block_ids = spatial_block_ids(fm.shape, block_px)
    seg, all_segment_ids, id_source = _segment_map(fm, segment_ids)
    flat_ids = block_ids.ravel()
    n_blocks = int(flat_ids.max()) + 1
    lidar_count = np.bincount(flat_ids, weights=lm.ravel().astype(float), minlength=n_blocks)
    cell_count = np.bincount(flat_ids, minlength=n_blocks).astype(float)
    lidar_fault_count = np.bincount(flat_ids, weights=(fm & lm).ravel().astype(float), minlength=n_blocks)
    coverage = lidar_count / np.maximum(cell_count, 1.0)
    eligible = np.flatnonzero((coverage >= min_lidar_fraction) & (lidar_fault_count > 0))

    rng = np.random.default_rng(seed)
    k = 0 if eligible.size == 0 or fraction == 0 else max(1, int(round(fraction * eligible.size)))
    chosen = rng.choice(eligible, size=min(k, eligible.size), replace=False) if k else np.array([], int)
    held_block = np.isin(block_ids, chosen)

    # A segment touched by any selected block is hidden everywhere to prevent long-trace leakage.
    touched_faults = fm & held_block
    held_ids = np.unique(seg[touched_faults])
    held_ids = held_ids[held_ids > 0]
    held_segment = fm & np.isin(seg, held_ids)
    holdout = fm & lm & held_block
    unscored_withheld = held_segment & ~holdout
    visible_raw = fm & ~held_segment
    if held_segment.any() and buffer_px:
        buffer_zone = ndimage.distance_transform_edt(~held_segment) <= buffer_px
    else:
        buffer_zone = np.zeros_like(fm)
    visible = visible_raw & ~buffer_zone
    buffer_dropped = visible_raw & buffer_zone

    return {
        "visible": visible,
        "holdout": holdout,
        "held_segment": held_segment,
        "unscored_withheld": unscored_withheld,
        "buffer_dropped": buffer_dropped,
        "held_block": held_block,
        "held_block_ids": sorted(int(x) for x in chosen),
        "held_segment_ids": sorted(int(x) for x in held_ids),
        "segment_id_source": id_source,
        "n_segments": len(all_segment_ids),
        "eligible_block_count": int(eligible.size),
        "holdout_cells": int(held_block.sum()),
        "holdout_fault_cells": int(holdout.sum()),
        "held_segment_fault_cells": int(held_segment.sum()),
        "unscored_withheld_fault_cells": int(unscored_withheld.sum()),
        "buffer_dropped_cells": int(buffer_dropped.sum()),
        "lidar_coverage_by_block": {int(b): float(coverage[b]) for b in chosen},
    }
