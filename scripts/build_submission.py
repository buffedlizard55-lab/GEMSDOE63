#!/usr/bin/env python3
"""E3 - build the submission raster from the CVaR-selected policy (full catalogue, no holdout).

Policy: registry/submission_policy.json (chosen in E2 by CVaR over label worlds).
Output: docs/downloads/<name>.tif (float32, EPSG:32611, 100 m, NaN outside footprint, 0/1 inside)
        registry/submissions/<name>.json (validator output, sha256, policy, evidence labels)
The organiser's 'Predicted values must be in range [0,1]' rejection is prevented by construction
(values are only 0.0 or 1.0 inside the footprint) and re-checked by gems63.validate.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems63.emit import ridge_score_vec, score_to_grid, policy_dots  # noqa: E402
from gems63.features import build_feature_stack  # noqa: E402
from gems63.grid import REPO, DATA_DIR, load_grid  # noqa: E402
from gems63.validate import validate  # noqa: E402


def main():
    pol = json.loads((REPO / "registry" / "submission_policy.json").read_text())
    p = pol["policy"]
    assert p["family"] == "ridge", "this build implements the selected 'ridge' family"
    g = load_grid()
    X, names = build_feature_stack(g.footprint)
    ridge_grid = score_to_grid(ridge_score_vec(X, names), g.footprint)
    del X
    # full catalogue is KNOWN (masked by the organisers), so it is the visible set here
    dots = policy_dots(ridge_grid, g, g.catalogue, p["q"], p["s"], p["r"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"gems63-cvar-ridge-q{int(round(p['q'] * 1000)):03d}-s{p['s']}-r{p['r']}-{stamp}"
    out_tif = REPO / "docs" / "downloads" / f"{name}.tif"
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(g.shape, dtype=np.float32)
    arr[~g.footprint] = np.nan
    arr[dots & g.footprint] = 1.0
    with rasterio.open(DATA_DIR / "labels.tif") as t:
        profile = dict(driver="GTiff", height=g.shape[0], width=g.shape[1], count=1,
                       dtype="float32", crs=t.crs, transform=t.transform, nodata=np.nan,
                       compress="deflate", tiled=False)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(1, description="GEMS63 CVaR-robust ridge dotted emission; 1=predicted fault dot, 0=no dot, NaN=outside footprint")
    v = validate(out_tif, DATA_DIR / "labels.tif")
    n_dots = int(np.count_nonzero(arr == 1.0))
    note = ("CVaR-robust ridge-lineness dots, 2% budget, NMS 2px, 3px catalogue-flank cut; "
            "holdout ~ random-dot control (NEGATIVE, not promoted)")
    rec = dict(
        submission_name=name, file=f"docs/downloads/{out_tif.name}", sha256=v["sha256"],
        bytes=out_tif.stat().st_size, note=note, note_chars=len(note),
        policy=p, dot_count=n_dots, validator=v,
        evidence=dict(
            holdout_nominal_dti=dict(value=pol["holdout_nominal_dti"], ci95=pol["holdout_nominal_ci95"],
                                      n_withheld_positives=pol["n_withheld_positives"],
                                      evaluator="gems63.dti v1 (exact binary, alpha 0.2, beta 0.8, R 300 m)",
                                      label="HOLDOUT-DTI"),
            random_dot_control_dti=dict(label="HOLDOUT-DTI control",
                                        source="registry/experiments/controls.json"),
            organizer_score=dict(value=None, label="ORGANIZER-CONFIRMED", status="none - not submitted"),
        ),
        generated_utc=stamp, generator="scripts/build_submission.py",
    )
    dest = REPO / "registry" / "submissions" / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ["submission_name", "sha256", "bytes", "dot_count", "note_chars"]}, indent=2))
    print("validator:", "PASS" if v["all_checks_passed"] else "FAIL", v["errors"])


if __name__ == "__main__":
    main()
