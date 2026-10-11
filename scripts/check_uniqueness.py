#!/usr/bin/env python3
"""Run the lane-drift gate on a written submission, with chance normalisation.

    python scripts/check_uniqueness.py [--tif docs/downloads/<name>.tif]

Why chance normalisation is not optional here
--------------------------------------------
The brief's gate is "more than 70 % of your dots fall within 3 px of one registry
raster's dots". Taken literally against a **dense** registry raster the test is
vacuous: `gems13_r13_lattice_s5` is a spacing-5 lattice covering the footprint, so
99.88 % of *uniformly random* dots land within 3 px of it. Any emission at all
"fails" against that comparator, including one with nothing in common with it.

So this script reports both numbers for every comparator:

* ``overlap`` -- the literal statistic the brief names.
* ``overlap_random`` -- the same statistic for a uniformly random emission of the
  same size inside the same footprint, measured not modelled.
* ``lift`` -- overlap / overlap_random.  A lift near 1 means co-location is fully
  explained by the comparator's density.  A lift well above 1 is a real signal.

The literal gate result is reported unchanged and is never overridden.  The lift is
reported alongside it so the selector step can tell a lane collision from an artifact
of comparing a sparse field to a dense one.
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

from gems63.raster_io import footprint, sha256          # noqa: E402
from gems63.uniqueness import (OVERLAP_GATE, RANK_CORR_GATE,  # noqa: E402
                               spearman_rank_corr)

LIFT_GATE = 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", default="")
    ap.add_argument("--registry", default=str(REPO / "registry" / "cache.json"))
    ap.add_argument("--card", default=str(REPO / "evidence" / "run_card.json"))
    ap.add_argument("--seed", type=int, default=20261011)
    a = ap.parse_args()

    tif = Path(a.tif) if a.tif else max((REPO / "docs" / "downloads").glob("*.tif"),
                                        key=lambda p: p.stat().st_mtime)
    with rasterio.open(tif) as src:
        mine = src.read(1) > 0
    foot = footprint()
    mine &= foot
    n_mine = int(mine.sum())
    nf = int(foot.sum())

    rng = np.random.default_rng(a.seed)
    ys, xs = np.nonzero(foot)
    pick = rng.choice(ys.size, size=n_mine, replace=False)
    rand = np.zeros(foot.shape, dtype=bool)
    rand[ys[pick], xs[pick]] = True

    entries = json.loads(Path(a.registry).read_text())["entries"]
    rows = []
    for e in entries:
        p = REPO / e["local_path"]
        if not p.exists():
            print(f"  MISSING {e['alias']}")
            continue
        with rasterio.open(p) as src:
            other = (src.read(1) > 0) & foot
        if not other.any():
            continue
        d = distance_transform_edt(~other)
        ovl = float((d[mine] <= 3).mean())
        ovl_r = float((d[rand] <= 3).mean())
        rho, _ = spearman_rank_corr(mine.astype(np.float32), other.astype(np.float32))
        rows.append({
            "alias": e["alias"],
            "owner_reported_score": e.get("owner_reported_score"),
            "sha256": e.get("sha256", ""),
            "other_dots": int(other.sum()),
            "other_density": float(other.sum()) / nf,
            "rank_corr": float(rho),
            "overlap": ovl,
            "overlap_random": ovl_r,
            "lift": (ovl / ovl_r) if ovl_r > 0 else None,
            "literal_gate_tripped": bool(rho > RANK_CORR_GATE or ovl > OVERLAP_GATE),
            "lift_gate_tripped": bool(rho > RANK_CORR_GATE or
                                      (ovl > OVERLAP_GATE and
                                       ovl_r > 0 and ovl / ovl_r > LIFT_GATE)),
        })
        del other, d

    rows.sort(key=lambda r: -(r["lift"] or 0))
    lit = [r["alias"] for r in rows if r["literal_gate_tripped"]]
    adj = [r["alias"] for r in rows if r["lift_gate_tripped"]]

    print(f"submission: {tif.name}   dots={n_mine:,}   "
          f"density={n_mine/nf:.5f}   sha256={sha256(tif)[:16]}")
    print(f"{'alias':<42s} {'dots':>8s} {'rho':>7s} {'ovl':>6s} {'ovlRnd':>7s} "
          f"{'lift':>5s}  gate")
    for r in rows:
        print(f"{r['alias']:<42s} {r['other_dots']:>8,} {r['rank_corr']:>+7.4f} "
              f"{r['overlap']:>6.3f} {r['overlap_random']:>7.3f} "
              f"{(r['lift'] or 0):>5.2f}  "
              f"{'LITERAL+LIFT' if r['lift_gate_tripped'] else ('literal' if r['literal_gate_tripped'] else '-')}")

    gate = {
        "n_registry_rasters": len(rows),
        "rank_corr_gate": RANK_CORR_GATE,
        "overlap_gate": OVERLAP_GATE,
        "overlap_radius_px": 3.0,
        "lift_gate": LIFT_GATE,
        "max_rank_corr": max((r["rank_corr"] for r in rows), default=0.0),
        "max_overlap": max((r["overlap"] for r in rows), default=0.0),
        "max_lift": max((r["lift"] or 0) for r in rows),
        "literal_gate_clear": len(lit) == 0,
        "literal_gate_tripped_by": lit,
        "lift_gate_clear": len(adj) == 0,
        "lift_gate_tripped_by": adj,
        "clear": len(lit) == 0,
        "duplicates": lit,
        "interpretation": (
            "The literal gate is the brief's test and is reported unchanged. "
            "Comparators whose overlap_random is close to their overlap are dense "
            "enough that co-location is explained by density alone; only a lift well "
            "above 1 is evidence of a lane collision."),
    }
    print("\n" + json.dumps(gate, indent=1))

    out = {"gate": gate, "comparisons": rows,
           "submission": {"path": str(tif.relative_to(REPO)), "dots": n_mine,
                          "sha256": sha256(tif)}}
    (REPO / "evidence" / "uniqueness_gate.json").write_text(json.dumps(out, indent=1))

    card_path = Path(a.card)
    if card_path.exists():
        card = json.loads(card_path.read_text())
        card["registry"] = gate
        card["registry_comparisons"] = rows
        card_path.write_text(json.dumps(card, indent=1, default=str))
        print(f"\nrun card updated -> {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
