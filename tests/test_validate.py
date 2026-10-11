"""Validator must accept a correct file and reject each known failure mode."""
import sys
from pathlib import Path
import numpy as np
import rasterio
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems63_stack.validate import validate

LABELS = Path("/tmp/gems_data/labels.tif")
pytestmark = pytest.mark.skipif(not LABELS.exists(), reason="competition labels not placed (see scripts/fetch_data.py)")


def _write_like(path, arr, crs=None):
    with rasterio.open(LABELS) as t:
        prof = dict(driver="GTiff", height=t.height, width=t.width, count=1, dtype="float32",
                    crs=crs or t.crs, transform=t.transform, nodata=np.nan)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype(np.float32), 1)


def _base():
    with rasterio.open(LABELS) as t:
        raw = t.read(1)
    arr = np.zeros(raw.shape, np.float32)
    arr[raw == -1] = np.nan
    arr[1000, 1000] = 1.0
    return arr


def test_valid_file_passes(tmp_path):
    p = tmp_path / "ok.tif"
    _write_like(p, _base())
    v = validate(p, LABELS)
    assert v["all_checks_passed"], v["errors"]


def test_value_above_one_fails(tmp_path):
    a = _base(); a[1000, 1001] = 1.2
    p = tmp_path / "bad_range.tif"; _write_like(p, a)
    assert "inside_values_in_0_1" in validate(p, LABELS)["errors"]


def test_nan_inside_footprint_fails(tmp_path):
    a = _base(); a[1000, 1002] = np.nan  # (1000,1002) is inside footprint? ensure inside below
    with rasterio.open(LABELS) as t:
        raw = t.read(1)
    assert raw[1000, 1002] != -1
    p = tmp_path / "bad_nan.tif"; _write_like(p, a)
    assert "inside_bounds_all_finite" in validate(p, LABELS)["errors"]


def test_wrong_crs_fails(tmp_path):
    from rasterio.crs import CRS
    p = tmp_path / "bad_crs.tif"; _write_like(p, _base(), crs=CRS.from_epsg(4326))
    assert "crs_matches_template" in validate(p, LABELS)["errors"]
