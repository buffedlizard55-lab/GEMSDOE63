"""Catalogue-free feature stack (reused by every experiment; nothing here reads labels.tif).

Inputs: training_features.tif (19 provided bands, EPSG:32611, 100 m). Band names are taken from
the band descriptions in the file's tags. Derived layers are computed only from provided bands:
  * Hessian "lineness" |lambda_big| - |lambda_small| at sigma = 2 px for bands tmi_hg, tilt_tc,
    elev_slope (sign-agnostic ridge/valley response; large on linear contacts).
  * gradient magnitude at sigma = 2 px for tmi_hg and elev_slope.
Missing sentinels (< -1e38) become NaN and are imputed with the footprint median per layer.
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy import ndimage

from .grid import DATA_DIR

BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_slope", "tilt_tc", "geod_shear",
    "geod_dilat", "tmi_vg", "eq_dist", "grav_vg", "elev_detrend", "iso_grav", "tmi",
    "basement_depth", "eq_density", "conductivity", "grav_hg", "elev_slope",
]  # index i -> band i+1 in training_features.tif
SIGMA = 2.0


def lineness(f: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    hxx = ndimage.gaussian_filter(f, sigma, order=(0, 2))
    hyy = ndimage.gaussian_filter(f, sigma, order=(2, 0))
    hxy = ndimage.gaussian_filter(f, sigma, order=(1, 1))
    tr = hxx + hyy
    disc = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy ** 2)
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    big = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
    small = np.where(np.abs(l1) >= np.abs(l2), l2, l1)
    return (np.abs(big) - np.abs(small)) * sigma ** 2


def load_bands(path=None):
    path = path or (DATA_DIR / "training_features.tif")
    with rasterio.open(path) as src:
        assert src.count == 19, src.count
        arr = src.read().astype(np.float32)
    arr[arr < -1e38] = np.nan
    return arr  # (19, H, W)


def build_feature_stack(footprint: np.ndarray, path=None):
    """Return X (n_footprint, F) float32 and feature names. Memory-conscious: each full-grid
    layer is reduced to its footprint column and released before the next is built."""
    bands = load_bands(path)
    n_fp = int(footprint.sum())
    names = list(BAND_NAMES) + [f"lineness_{n}" for n in ["tmi_hg", "tilt_tc", "elev_slope"]] \
        + [f"gradmag_{n}" for n in ["tmi_hg", "elev_slope"]]
    X = np.empty((n_fp, len(names)), dtype=np.float32)
    layers = {}
    for i, n in enumerate(BAND_NAMES):
        a = bands[i]
        fin = np.isfinite(a)
        med = float(np.nanmedian(a[footprint])) if fin[footprint].any() else 0.0
        layers[n] = np.where(fin, a, np.float32(med)).astype(np.float32)
        X[:, i] = layers[n][footprint]
    del bands
    j = len(BAND_NAMES)
    for n in ["tmi_hg", "tilt_tc", "elev_slope"]:
        X[:, j] = lineness(layers[n]).astype(np.float32)[footprint]
        j += 1
    for n in ["tmi_hg", "elev_slope"]:
        X[:, j] = ndimage.gaussian_gradient_magnitude(layers[n], SIGMA).astype(np.float32)[footprint]
        j += 1
    return X, names


def to_grid(vec: np.ndarray, footprint: np.ndarray, fill=np.nan) -> np.ndarray:
    out = np.full(footprint.shape, fill, dtype=np.float32)
    out[footprint] = vec
    return out
