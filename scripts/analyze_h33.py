#!/usr/bin/env python3
"""Measure the group's best-scoring artifact and work out what its score implies.

`h33-h33-2-b2-...-zeros` is owner-reported at 0.2778, the highest number any
repository in this group has recorded.  This script does not take anyone's word for
what that file contains: it opens the bytes, measures them, and then derives what the
owner-reported score arithmetically implies about the truth set.

    python scripts/analyze_h33.py [--alias gemsdoe32_h33_2_b2_02778_zeros]

The derivation, all of it from the published metric:

    DTI = T / (0.2 T + 0.2 S + 0.8 N)          (identity 1, docs/method)
    =>  T = S * 0.2*DTI/(1 - 0.2*DTI) + N * 0.8*DTI/(1 - 0.2*DTI)

Every emitted dot contributes at most 1 to S, so S <= |dots|.  A dot more than 3 px
from every truth pixel has kmax = 0 and contributes exactly 1.  A dot that lands on a
straight truth trace earns at most 3.0 of T -- that is the competition's own worked
example, asserted in tests/test_metric.py -- which bounds the credit any set of dots
can earn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems63.metric import ALPHA, BETA  # noqa: E402
from gems63.raster_io import catalogue, footprint, sha256  # noqa: E402

# Credit a single dot can earn on an ideal straight trace: the published worked
# example, TP_w = 3.00 for one dot centred on a vertical line.
MAX_CREDIT_PER_DOT = 3.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="gemsdoe32_h33_2_b2_02778_zeros")
    ap.add_argument("--score", type=float, default=0.2778,
                    help="owner-reported DTI; NOT an organizer receipt")
    ap.add_argument("--out", default=str(REPO / "evidence" / "h33_analysis.json"))
    a = ap.parse_args()

    cache = json.loads((REPO / "registry" / "cache.json").read_text())
    entry = next(e for e in cache["entries"] if e["alias"] == a.alias)
    path = REPO / entry["local_path"]

    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
    finite = np.isfinite(arr)
    pos = arr > 0
    n_dots = int(pos.sum())
    uniq = np.unique(arr[finite])

    cat = catalogue()
    foot = footprint()
    d = distance_transform_edt(~cat)
    on_cat = int((pos & cat).sum())
    near = int((pos & (d <= 3) & ~cat).sum())
    far = int((pos & (d > 3)).sum())

    # ---- what the owner-reported score implies -------------------------------
    dti = a.score
    kT = ALPHA * dti / (1 - ALPHA * dti)          # coefficient on S
    kN = BETA * dti / (1 - ALPHA * dti)           # coefficient on N
    # Case A: all scoreable truth lies within 3 px of the mapped catalogue.
    # Then every far dot has kmax = 0, so S >= far, and T <= near * 3.0.
    s_min = far
    t_max = near * MAX_CREDIT_PER_DOT
    # dti <= t_max / (0.2 t_max + 0.2 s_min + 0.8 N)  =>  N <= ...
    denom_fixed = ALPHA * t_max + ALPHA * s_min
    n_max_case_a = (t_max / dti - denom_fixed) / BETA
    # recall floor with S = 0 (the sibling derivation, re-derived here)
    recall_floor = kN                              # T/N >= kN when S = 0

    h, w = arr.shape
    blocks = [[int(pos[i * h // 3:(i + 1) * h // 3,
                       j * w // 2:(j + 1) * w // 2].sum()) for j in range(2)]
              for i in range(3)]

    out = {
        "artifact": {
            "alias": a.alias, "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size, "sha256": sha256(path),
            "manifest_sha256": entry["sha256"],
            "sha256_matches_manifest": sha256(path) == entry["sha256"],
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "crs": str(meta["crs"]), "dtype": str(arr.dtype),
            "n_nan": int((~finite).sum()),
            "distinct_values": [float(v) for v in uniq[:10]],
            "is_binary": bool(uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}),
        },
        "measured": {
            "positive_pixels": n_dots,
            "on_catalogue_pixels": on_cat,
            "within_3px_of_catalogue": near,
            "beyond_3px_of_catalogue": far,
            "fraction_beyond_3px": far / max(1, n_dots),
            "mean_distance_to_catalogue_px": float(d[pos].mean()),
            "median_distance_to_catalogue_px": float(np.median(d[pos])),
            "spatial_blocks_3x2": blocks,
            "catalogue_pixels_total": int(cat.sum()),
            "footprint_pixels_total": int(foot.sum()),
            "dot_density_of_footprint": n_dots / int(foot.sum()),
        },
        "implied_by_owner_reported_score": {
            "score_used": dti,
            "score_class": "OWNER-REPORTED, not an organizer receipt",
            "T_equals": f"T = {kT:.6f}*S + {kN:.6f}*N",
            "recall_floor_T_over_N_at_S_zero": recall_floor,
            "case_A_assumption": ("all scoreable truth lies within 3 px (300 m) of the "
                                  "mapped catalogue, i.e. this repository's "
                                  "offset-ribbon hypothesis is the whole truth"),
            "case_A_min_S": s_min,
            "case_A_max_T": t_max,
            "case_A_max_truth_pixels_N": n_max_case_a,
            "case_A_max_N_as_fraction_of_catalogue":
                n_max_case_a / max(1, int(cat.sum())),
            "reading": ("If the scoreable truth were confined to within 300 m of the "
                        "mapped catalogue, this artifact could only have reached "
                        f"{dti} if the entire truth set were <= "
                        f"{n_max_case_a:,.0f} pixels, about "
                        f"{100*n_max_case_a/max(1,int(cat.sum())):.0f}% of the "
                        "catalogue's own pixel count. A larger truth set makes that "
                        "impossible, which would mean a substantial share of the truth "
                        "lies further out than 300 m -- exactly where "
                        f"{100*far/max(1,n_dots):.0f}% of this artifact's dots are."),
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
