#!/usr/bin/env python3
"""Recompute uniqueness of the final raster against the extended registry
(every fetched earlier raster incl. the other top-scoring submissions),
update evidence/uniqueness.json, run_card.json and hash-collision list."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems63 import uniqueness

FINAL_GLOB = "gemsdoe63-*-zeros.tif"


def main():
    rc = json.loads((ROOT / "evidence/run_card.json").read_text())
    final = ROOT / "docs/downloads" / rc["raster"]["filename"]
    foot = None
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        foot = np.isfinite(ds.read(1))
    with rasterio.open(final) as ds:
        values = ds.read(1).astype(np.float32)

    registry = {}
    mine_names = {rc["raster"]["filename"], rc["raster"]["filename"].replace("-zeros.tif", "-nan.tif")}
    parallel_dirs = [ROOT / "docs/downloads", ROOT / "held"]
    for pdir in parallel_dirs:
        if pdir.exists():
            for p in sorted(pdir.glob("*.tif")):
                if p.name in mine_names or p.name.startswith("checks-"):
                    continue
                try:
                    with rasterio.open(p) as ds:
                        a = ds.read(1).astype(np.float32)
                    if a.shape == values.shape:
                        registry[f"parallel::{p.parent.name}::{p.name}"] = a
                except Exception:
                    pass
    for p in sorted((ROOT / "registry_rasters").glob("*.tif")):
        try:
            with rasterio.open(p) as ds:
                a = ds.read(1).astype(np.float32)
            if a.shape == values.shape:
                registry[p.name] = a
            else:
                print(f"  shape-skip {p.name} {a.shape}")
        except Exception as e:
            print(f"  unreadable {p.name}: {e}")
    for p in sorted((ROOT / "inputs").glob("**/*.tif")):
        try:
            with rasterio.open(p) as ds:
                a = ds.read(1).astype(np.float32)
            if a.shape == values.shape:
                registry[f"inputs::{p.name}"] = a
        except Exception:
            pass

    uniq_final = uniqueness.check_uniqueness(values, foot, registry)

    # sha256 byte-collision check across all registry + input files
    my_sha = hashlib.sha256(final.read_bytes()).hexdigest()
    collisions = []
    for p in sorted((ROOT / "registry_rasters").glob("*.tif")) + sorted((ROOT / "inputs").glob("**/*.tif")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h == my_sha:
            collisions.append(str(p))
    uniq_final["sha256_collisions"] = collisions
    uniq_final["n_registry_rasters_checked"] = len(registry)

    uj = json.loads((ROOT / "evidence/uniqueness.json").read_text())
    uj["final_extended"] = uniq_final
    (ROOT / "evidence/uniqueness.json").write_text(json.dumps(uj, indent=1))

    rc["correlation_overlap_vs_registry"] = {
        "max_abs_spearman": uniq_final["max_abs_spearman"],
        "max_dot_overlap_3px": uniq_final["max_dot_overlap_3px"],
        "verdict": uniq_final["verdict"],
        "breaches": uniq_final["breaches"],
        "n_registry_rasters_checked": len(registry),
        "sha256_collisions": collisions,
        "controls_flagged": list(uniq_final["flagged_controls"].keys()),
    }
    rc["ok_to_download_and_submit"] = bool(rc["raster"]["all_checks_passed"] and uniq_final["verdict"] == "UNIQUE" and not collisions)
    rc["verdict"] = "promote" if rc["ok_to_download_and_submit"] else "negative"
    (ROOT / "evidence/run_card.json").write_text(json.dumps(rc, indent=1))

    print(f"checked {len(registry)} rasters; collisions={collisions}")
    print(f"max|rho|={uniq_final['max_abs_spearman']:.4f}  max3pxOverlap={uniq_final['max_dot_overlap_3px']:.4f}  -> {uniq_final['verdict']}")
    worst = sorted(uniq_final["per_registry"].items(), key=lambda kv: -kv[1]["dot_overlap_3px"])[:8]
    for n, r in worst:
        print(f"   {n[:72]:72s} overlap={r['dot_overlap_3px']:.4f} rho={r['spearman_rho']:+.3f} control={r['control']}")


if __name__ == "__main__":
    main()
