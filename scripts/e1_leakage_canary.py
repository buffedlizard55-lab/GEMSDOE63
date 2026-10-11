#!/usr/bin/env python3
"""E1 - leakage canary (protocol item 4).

Each catalogue-free feature is tested ALONE on the spatially blocked hide-and-recover holdout:
  positives  = withheld catalogue pixels of fold k (truth the model must recover)
  negatives  = footprint pixels farther than 3 px from ANY catalogue pixel (background)
  AUC        = Mann-Whitney AUC of the feature value (no model, no training)
Rule: AUC > 0.90 on any fold means leakage until proven otherwise (flagged, feature dropped).
Output: registry/experiments/e1_canary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems63.grid import REPO, load_grid, withheld_mask, clearance_zone, N_FOLDS  # noqa: E402
from gems63.features import build_feature_stack  # noqa: E402

LEAK_AUC = 0.90
RNG = np.random.default_rng(63)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    g = load_grid()
    X, names = build_feature_stack(g.footprint)
    fp_idx = np.flatnonzero(g.footprint.ravel())
    clear = clearance_zone(g).ravel()[fp_idx]
    bg_pool = np.flatnonzero(~clear)
    bg = RNG.choice(bg_pool, size=min(200_000, bg_pool.size), replace=False)
    out = {"leak_auc_threshold": LEAK_AUC, "features": {}, "folds": N_FOLDS}
    fold_pos = {}
    for k in range(N_FOLDS):
        wh = withheld_mask(g, k).ravel()[fp_idx]
        fold_pos[k] = np.flatnonzero(wh)
        out.setdefault("withheld_positives", {})[str(k)] = int(wh.sum())
    out["withheld_positives_total"] = int(sum(v for v in out["withheld_positives"].values()))
    for j, n in enumerate(names):
        col = X[:, j]
        per_fold = {}
        for k in range(N_FOLDS):
            pos = col[fold_pos[k]]
            pos = pos[np.isfinite(pos)]
            per_fold[str(k)] = auc(pos, col[bg])
        vals = list(per_fold.values())
        out["features"][n] = {
            "auc_per_fold": per_fold,
            "auc_min": min(vals), "auc_max": max(vals),
            # AUC is one-sided; a feature that is strongly anti-correlated is also informative,
            # so the canary uses the distance from 0.5 as the leakage test statistic.
            "auc_sided_max": max(max(vals), 1 - min(vals)),
            "leak_flag": bool(max(max(vals), 1 - min(vals)) > LEAK_AUC),
        }
    flagged = [n for n, v in out["features"].items() if v["leak_flag"]]
    out["flagged_leakage"] = flagged
    out["verdict"] = "PASS: no single catalogue-free feature exceeds AUC 0.90" if not flagged else "FLAG"
    dest = REPO / "registry" / "experiments" / "e1_canary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    top = sorted(out["features"].items(), key=lambda kv: -kv[1]["auc_sided_max"])[:8]
    print("E1 canary (single-feature holdout AUC; sided max over folds):")
    for n, v in top:
        print(f"  {n:22s} sided_max={v['auc_sided_max']:.3f}  per-fold={[round(x,3) for x in v['auc_per_fold'].values()]}")
    print("withheld positives per fold:", out["withheld_positives"], "total", out["withheld_positives_total"])
    print(out["verdict"])


if __name__ == "__main__":
    main()
