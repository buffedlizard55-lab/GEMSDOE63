"""Internal exactness tests for the GEMSDOE63 instrument (template cross-check lives
in scripts/verify_instrument.py; these run without the template checkout)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.ndimage import distance_transform_edt

from gems63 import instrument
from gems63.metric import dti_from_counts, kernel


def test_kmax_grid_matches_edt():
    rng = np.random.default_rng(1)
    tg = rng.random((80, 90)) < 0.002
    kg = instrument._kmax_grid(tg)
    ref = kernel(distance_transform_edt(~tg)).astype(np.float32)
    interior = np.zeros_like(tg, bool)
    interior[4:-4, 4:-4] = True
    assert np.abs(kg - ref)[interior].max() < 1e-6


def test_dti_from_counts_identity():
    # TP=5, FP=10, |G|=20 -> 5/(5+2+12)
    assert abs(dti_from_counts(5.0, 10.0, 20) - 5.0 / 19.0) < 1e-9


def test_prevalence_calibration_shrinks_dense_advantage():
    n = np.array([1000.0])
    s = np.array([0.5])
    tp_sparse, fp_sparse = np.array([300.0]), np.array([300.0])
    tp_dense, fp_dense = np.array([450.0]), np.array([2500.0])
    d_sparse = instrument.dti_prevalence_calibrated(tp_sparse, fp_sparse, n, s)[0]
    d_dense = instrument.dti_prevalence_calibrated(tp_dense, fp_dense, n, s)[0]
    assert d_sparse > d_dense  # 0.2-weighted FP mass dominates at calibrated prevalence


if __name__ == "__main__":
    test_kmax_grid_matches_edt()
    test_dti_from_counts_identity()
    test_prevalence_calibration_shrinks_dense_advantage()
    print("3/3 internal tests PASS")
