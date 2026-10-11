import pytest
import numpy as np
from pathlib import Path
from src.gems63.metric import dti_binary, cvar_penalized_score
from src.gems63.submission_writer import audit_submission
import rasterio

def test_dti_worked_example():
    # Worked example from DrivenData page:
    # TP_w=3.00, FP_w=1.89, FN_w=2.00 -> 3.00/(3.00+0.2*1.89+0.8*2.00) = 0.60265
    from src.gems63.metric import ALPHA, BETA, EPS
    tp, fp, fn = 3.0, 1.89, 2.0
    dti = tp / (tp + ALPHA * fp + BETA * fn + EPS)
    assert abs(dti - 0.60265) < 1e-4

def test_cvar_penalized_score():
    scores = [0.10, 0.08, 0.06, 0.04]
    res = cvar_penalized_score(scores, worst_quantile=0.25, penalty_weight=0.5)
    assert res["min"] == 0.04
    assert res["worst_tail_cvar"] == 0.04
    assert res["mean_dti"] == 0.07
    # penalized = 0.07 - 0.5 * (0.07 - 0.04) = 0.055
    assert abs(res["penalized_cvar"] - 0.055) < 1e-6

def test_submission_raster_integrity():
    tif_path = Path("docs/downloads/gems63-cvar-robust-greedy-pruned-10290dots-20261011T220000Z-f49a21c0-zeros.tif")
    assert tif_path.exists(), "Submission GeoTIFF does not exist"
    
    with rasterio.open("data/labels.tif") as src:
        labels = src.read(1)
        foot = (labels != src.nodata)
        cat = (labels == 1)
        
    audit = audit_submission(tif_path, foot, cat)
    assert audit["all_checks_passed"] is True
    assert audit["emitted_positive_pixels"] == 10290
    assert audit["on_catalogue_positive_pixels"] == 0
    assert audit["checks"]["in_footprint_range_0_1"] is True
    assert audit["checks"]["outside_footprint_zeros"] is True
