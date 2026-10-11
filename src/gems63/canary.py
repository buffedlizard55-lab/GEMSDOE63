"""Leakage canaries (parallel-run protocol rule 4).

Each feature is tested ALONE on the hide-and-recover holdout: AUC > 0.90 for
predicting withheld-truth presence means leakage until proven otherwise.
Features tested:

* ``k_visible``  -- triangular-kernel credit from distance to the VISIBLE
                    catalogue (the geometric context every policy uses);
* ``rad_ratio``  -- radiometric K/Th log-ratio z-score (hypothesis H63-B);
                    skipped unless band provenance is confirmed.

The AUC is computed on the scored (active) domain of the pooled "all" cells,
with negatives sub-sampled for speed.  No holdout DTI is produced here; this
is a diagnostic AUC only.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from gems57.metric import kernel


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC."""
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_canaries(cells, hidden_by_cell, grid, rad_field=None,
                 max_negatives: int = 200_000, seed: int = 63) -> dict:
    """Per-feature solo AUC across the pooled 'all'-mode cells."""
    rng = np.random.default_rng(seed)
    feats = {"k_visible": []}
    if rad_field is not None:
        feats["rad_ratio"] = []
    labels_acc = []
    for cell in cells:
        if not cell.key.endswith("_all"):
            continue
        vis_full = grid.catalogue & ~hidden_by_cell[cell.key]
        dv = ndi.distance_transform_edt(~vis_full)
        kv = kernel(dv)
        truth = np.zeros(cell.active.shape, bool)
        ty, tx = cell.truth_yx
        truth[ty, tx] = True
        act = cell.active
        pos = truth[act]
        kv_act = kv[cell.bbox][act]
        n_pos = int(pos.sum())
        if n_pos == 0:
            continue
        neg_idx = np.nonzero(~pos)[0]
        if len(neg_idx) > max_negatives:
            neg_idx = rng.choice(neg_idx, max_negatives, replace=False)
        keep = np.zeros(len(pos), bool)
        keep[neg_idx] = True
        sel = pos | keep
        labels_acc.append(pos[sel])
        feats["k_visible"].append(kv_act[sel])
        if rad_field is not None:
            feats["rad_ratio"].append(rad_field[cell.bbox][act][sel])
    out = dict(evidence_class="CANARY-AUC", leakage_threshold=0.90,
               n_positive_px=int(np.concatenate(labels_acc).sum()),
               features={})
    for name, parts in feats.items():
        if not parts:
            out["features"][name] = dict(auc=None, note="no data")
            continue
        s = np.concatenate(parts)
        y = np.concatenate(labels_acc)
        a = auc(s, y)
        out["features"][name] = dict(
            auc=a, leakage_flag=bool(a > 0.90),
            note="LEAKAGE until proven otherwise" if a > 0.90 else
                 "below the 0.90 leakage bar")
    return out
