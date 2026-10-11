"""Lane-drift and uniqueness checks against earlier competition rasters."""

from __future__ import annotations
import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.stats import spearmanr


def check_uniqueness(cand_dots: np.ndarray, comparator_path: str, foot: np.ndarray) -> dict:
    """Check 3px dilation overlap (< 70%) and Spearman rank-correlation (< 0.90)."""
    with rasterio.open(comparator_path) as src:
        c_arr = src.read(1)

    c_dots = (c_arr > 0.5)
    struct = ndi.generate_binary_structure(2, 2)
    c_dil = ndi.binary_dilation(c_dots, structure=struct, iterations=3)

    overlap = float((cand_dots & c_dil).sum()) / float(max(1, cand_dots.sum()))

    s1 = cand_dots[foot].astype(np.float64)
    s2 = np.nan_to_num(c_arr[foot], nan=0.0).astype(np.float64)

    if np.all(s1 == s1[0]) or np.all(s2 == s2[0]):
        rho = 0.0
    else:
        rho, _ = spearmanr(s1, s2)
        if np.isnan(rho):
            rho = 0.0

    return {
        "comparator": comparator_path.split("/")[-1],
        "overlap_3px": float(overlap),
        "overlap_pass": bool(overlap < 0.70),
        "spearman_rho": float(rho),
        "rho_pass": bool(rho < 0.90),
        "passed": bool(overlap < 0.70 and rho < 0.90),
    }
