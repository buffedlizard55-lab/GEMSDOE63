#!/usr/bin/env python3
"""Uniqueness check of the candidate raster against every earlier raster in the owner's public repos.

Registry = every *.tif at HEAD of the 73 GEMSDOE* repositories of buffedlizard55-lab
(index built with the GitHub API; files fetched by blobless git clone; HEAD only, so rasters
deleted from history are NOT covered - see registry/irregularities.md).
Tests per registry raster with the same grid (3730 x 3292):
  T1 exact duplicate: SHA-256 equality
  T2 correlation: Pearson r over footprint pixels (gate: r > 0.90 = duplicate, PROTOCOL item 1)
  T3 dot overlap: share of OUR dots within 3 px of the registry raster's dot set, where a
     registry dot = cell >= max(0.5, its 98th percentile) (gate: > 70% = duplicate)
Output: registry/uniqueness_check.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems63.grid import REPO  # noqa: E402

REG_ROOT = Path("/tmp/regs")
R_GATE = 0.90
OVERLAP_GATE = 0.70
DOT_R = 3.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(cand_path: Path):
    with rasterio.open(cand_path) as s:
        cand = s.read(1)
    fp = np.isfinite(cand)
    cand_dots = np.argwhere(cand == 1.0)
    cand_sha = sha(cand_path)
    index = json.load(open("/tmp/tif_index.json"))
    rows, errors = [], []
    for repo, path, size in index:
        p = REG_ROOT / repo / path
        if not p.exists() or p.stat().st_size != size:
            errors.append({"repo": repo, "path": path, "why": "missing or size mismatch"})
            continue
        try:
            with rasterio.open(p) as s:
                if (s.height, s.width) != cand.shape or s.count < 1:
                    continue
                r = s.read(1).astype(np.float32)
        except Exception as e:  # unreadable -> record, don't hide
            errors.append({"repo": repo, "path": path, "why": f"read error: {e}"})
            continue
        rec = {"repo": repo, "path": path, "sha256": sha(p)}
        m = fp & np.isfinite(r)
        if m.sum() < 1000:
            continue
        a, b = cand[m].astype(np.float64), r[m].astype(np.float64)
        sa, sb = a.std(), b.std()
        pear = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else float("nan")
        rec["pearson_footprint"] = pear
        rv = r[np.isfinite(r)]
        p98 = float(np.percentile(rv, 98)) if rv.size else np.inf
        thr = max(0.5, p98)
        reg_dots = np.argwhere(np.nan_to_num(r, nan=-1.0) >= thr)
        if cand_dots.size and reg_dots.size:
            t = cKDTree(reg_dots.astype(float))
            d, _ = t.query(cand_dots.astype(float), k=1, distance_upper_bound=DOT_R + 1e-9)
            rec["dot_overlap_within_3px"] = float(np.isfinite(d).mean())
        else:
            rec["dot_overlap_within_3px"] = 0.0
        rec["registry_dot_cells"] = int(reg_dots.shape[0])
        rows.append(rec)
    # Spearman only for the top correlates (expensive on 5M cells)
    top = sorted(rows, key=lambda x: -(x["pearson_footprint"] if np.isfinite(x["pearson_footprint"]) else -9))[:25]
    for rec in top:
        p = REG_ROOT / rec["repo"] / rec["path"]
        with rasterio.open(p) as s:
            r = s.read(1).astype(np.float32)
        m = fp & np.isfinite(r)
        rec["spearman_footprint"] = float(spearmanr(cand[m], r[m]).correlation)
    # exact duplicates
    cand_hash_hits = [x for x in rows if x.get("sha256") == cand_sha]
    max_p = max(rows, key=lambda x: (x["pearson_footprint"] if np.isfinite(x["pearson_footprint"]) else -9))
    max_o = max(rows, key=lambda x: x["dot_overlap_within_3px"])
    dup_p = [x for x in rows if np.isfinite(x["pearson_footprint"]) and x["pearson_footprint"] > R_GATE]
    dup_o = [x for x in rows if x["dot_overlap_within_3px"] > OVERLAP_GATE]
    out = dict(candidate=cand_path.name, candidate_sha256=cand_sha, rasters_compared=len(rows),
               unreadable_or_missing=len(errors), errors=errors[:50], n_error_total=len(errors),
               gates=dict(pearson_gt=R_GATE, dot_overlap_gt=OVERLAP_GATE),
               max_pearson=dict(value=max_p["pearson_footprint"], repo=max_p["repo"], path=max_p["path"]),
               max_dot_overlap=dict(value=max_o["dot_overlap_within_3px"], repo=max_o["repo"], path=max_o["path"]),
               exact_duplicate_hits=cand_hash_hits,
               duplicates_by_pearson=[(x["repo"], x["path"], x["pearson_footprint"]) for x in dup_p],
               duplicates_by_overlap=[(x["repo"], x["path"], x["dot_overlap_within_3px"]) for x in dup_o],
               top25_by_pearson=top,
               verdict=("DUPLICATE - log and stop" if (cand_hash_hits or dup_p or dup_o) else
                        "UNIQUE under protocol gates (HEAD-only registry; see limits)"))
    (REPO / "registry" / "uniqueness_check.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ["candidate", "rasters_compared", "n_error_total", "max_pearson",
                                           "max_dot_overlap", "verdict"]}, indent=2, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
