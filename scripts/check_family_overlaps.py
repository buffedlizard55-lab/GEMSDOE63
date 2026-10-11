#!/usr/bin/env python3
"""Measure 3px dot overlaps of the family's live-scored artifacts against ens12/r13
controls and dotted submissions - establishes how the 70% lane rule behaves on the
organizer-confirmed best artifacts (context for the registry-breach interpretation)."""
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with rasterio.open(path) as ds:
        a = ds.read(1)
    return np.isfinite(a) & (a > 0)


def overlap(a, b):
    if not a.any() or not b.any():
        return float("nan")
    d = distance_transform_edt(~b)
    return float((d[a] <= 3.0).mean())


def main():
    foot = np.isfinite(rasterio.open(ROOT / "data/sample_submission.tif").read(1))
    ens12 = load(ROOT / "inputs/calibration/ens12_7f00890a.tif")
    r13 = load(ROOT / "inputs/calibration/r13_lattice_s5_v2_nan.tif")
    h195 = load(ROOT / "inputs/h19_5_nan.tif")
    names = {
        "d28(0.2600)": ROOT / "registry_rasters/gemsdoe32-d28-ref02600-repro-44090-20261004T183200Z-426073b6-zeros.tif",
        "h33(0.2778)": ROOT / "registry_rasters/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif",
        "h32d": ROOT / "registry_rasters/gemsdoe32-h32d-submodular-multipysics-46090-20261004T183200Z-4de30601-zeros.tif",
        "d15(0.2477)": ROOT / "inputs/calibration/d15_scored_nan.tif",
        "MY-FINAL": ROOT / "docs/downloads/gemsdoe63-cvarlw-thin-s3p17-55591-20261011T215754Z-bf9b39b1-zeros.tif",
    }
    print(f"{'artifact':14s} {'pos px':>8s} {'ens12':>8s} {'r13':>8s} {'h19_5':>8s} {'d28':>8s} {'h33':>8s}")
    d28m = load(names["d28(0.2600)"])
    h33m = load(names["h33(0.2778)"])
    for nm, p in names.items():
        m = load(p)
        print(f"{nm:14s} {int(m.sum()):8d} {overlap(m, ens12):8.4f} {overlap(m, r13):8.4f} "
              f"{overlap(m, h195):8.4f} {overlap(m, d28m):8.4f} {overlap(m, h33m):8.4f}")


if __name__ == "__main__":
    main()
