"""Grid, footprint, catalogue segments and spatially blocked hide-and-recover folds.

Grid facts are read from the files themselves (see registry/grid_facts.json, written by
scripts/inspect_inputs.py). Expected: EPSG:32611, 100 m, 3292 x 3730, transform
(100, 0, 243350, 0, -100, 4508550), labels int8 with nodata -1 outside the footprint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GEMS_DATA_DIR", "/tmp/gems_data"))
N_FOLDS = 5
BLOCKS = 4          # 4x4 spatial super-blocks -> fold = (block index) mod N_FOLDS
BUFFER_PX = 10      # 1 km buffer around withheld segments (no training pixels, no visible labels)
NEG_CLEAR_PX = 3    # background negatives must be > 3 px from any catalogue pixel


@dataclass
class Grid:
    labels: np.ndarray        # int8 raw labels (-1 outside footprint)
    footprint: np.ndarray     # bool, labels != -1
    catalogue: np.ndarray     # bool, labels == 1
    transform: tuple
    crs: str
    shape: tuple
    segments: np.ndarray      # int32, 8-connected component id per catalogue pixel (0 = none)
    n_segments: int
    fold_of_segment: np.ndarray  # int32 array len n_segments+1 (index 0 unused)


def load_grid(labels_path: Path | None = None) -> Grid:
    labels_path = labels_path or (DATA_DIR / "labels.tif")
    with rasterio.open(labels_path) as src:
        raw = src.read(1)
        transform = tuple(src.transform)[:6]
        crs = src.crs.to_string()
        shape = (src.height, src.width)
        nodata = src.nodata
    footprint = raw != -1 if nodata == -1 else np.ones(raw.shape, bool)
    catalogue = (raw == 1) & footprint
    seg, n = ndimage.label(catalogue, structure=np.ones((3, 3), int))
    fold = np.zeros(n + 1, dtype=np.int32)
    if n:
        yy, xx = np.nonzero(seg)
        cy = ndimage.mean(yy, labels=seg[yy, xx], index=np.arange(1, n + 1))
        cx = ndimage.mean(xx, labels=seg[yy, xx], index=np.arange(1, n + 1))
        H, W = shape
        bi = np.clip((np.asarray(cy) * BLOCKS // H).astype(int), 0, BLOCKS - 1)
        bj = np.clip((np.asarray(cx) * BLOCKS // W).astype(int), 0, BLOCKS - 1)
        fold[1:] = (bi * BLOCKS + bj) % N_FOLDS
    return Grid(raw, footprint, catalogue, transform, crs, shape, seg, n, fold)


def withheld_mask(g: Grid, k: int) -> np.ndarray:
    """Catalogue pixels of fold k (withheld in fold k)."""
    fold_pix = g.fold_of_segment[g.segments]
    return g.catalogue & (fold_pix == k)


def visible_mask(g: Grid, k: int) -> np.ndarray:
    """Catalogue pixels visible to the model in fold k (= known, masked by the organizers)."""
    return g.catalogue & ~withheld_mask(g, k)


def buffer_zone(withheld: np.ndarray) -> np.ndarray:
    """Pixels within BUFFER_PX (Euclidean) of withheld truth; excluded from training."""
    if not withheld.any():
        return np.zeros_like(withheld)
    d = ndimage.distance_transform_edt(~withheld)
    return d <= BUFFER_PX


def clearance_zone(g: Grid) -> np.ndarray:
    """Pixels within NEG_CLEAR_PX of ANY catalogue pixel (never used as background negatives)."""
    if not g.catalogue.any():
        return np.zeros_like(g.catalogue)
    d = ndimage.distance_transform_edt(~g.catalogue)
    return d <= NEG_CLEAR_PX
