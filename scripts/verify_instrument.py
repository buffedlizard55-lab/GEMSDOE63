#!/usr/bin/env python3
"""Cross-verify GEMSDOE63 instrument bit-for-bit against the shared template.

1. Per-cell unperturbed holdout DTI of a test mask must match the template's
   evaluate_candidate_holdout folds exactly.
2. provider_scan top-two structure must reproduce EDT-exact TP exactly.
3. The 25-offset kmax grid must reproduce kernel(EDT-to-truth) exactly inside bbox.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/home/user/mirror32/src")

from gems63 import holdout, instrument, worlds
from gems63.metric import kernel
import gems32.holdout as t_holdout


def main():
    foot, labels, sgmc_off, near3, transform, crs = holdout.load_grid(ROOT / "data")

    # template context + evaluation
    tctx = t_holdout.load_holdout_context(ROOT / "data")
    rng = np.random.default_rng(7)
    test_mask = (rng.random(foot.shape) < 0.001) & foot & ~labels
    t_eval = t_holdout.evaluate_candidate_holdout(test_mask, tctx, name="verify")

    # my cells + baseline
    cells = holdout.build_cells(foot, labels)
    cws = [worlds.generate_worlds(c, n_worlds=2, seed=63000) for c in cells]
    uni = instrument.build_universe([test_mask], foot.shape)
    my = {}
    for c, ws in zip(cells, cws):
        cwd = instrument.prepare_cell_worlds(c, ws, uni, foot.shape)
        # baseline DTI via the universe-independent exact path (EDT), template logic
        sub = test_mask[c.bbox] & c.active_sub
        brr, bcc = cwd.baseline_truth
        n_t = brr.size
        if n_t and sub.any():
            dp = distance_transform_edt(~sub)
            tp = float(kernel(dp[brr, bcc]).sum())
            present = np.zeros(uni.N, dtype=bool)
            gi = cwd.emap[sub]
            present[gi[gi >= 0]] = True
            Sdot = present[cwd.cidx]
            fp = float(((1.0 - cwd.baseline_kvals) * Sdot).sum())
            import gems63.metric as m
            my[c.key] = m.dti_from_counts(tp, fp, n_t)
        else:
            my[c.key] = 0.0
    ok = True
    print(f"{'cell':16s} {'template':>12s} {'mine':>12s} {'absdiff':>12s}")
    for k in sorted(my):
        tv = t_eval["catalogue_hidden_folds"][k]["dti"]
        d = abs(tv - my[k])
        ok &= d < 1e-9
        print(f"{k:16s} {tv:12.8f} {my[k]:12.8f} {d:12.2e}")
    print("CHECK 1 (per-cell baseline DTI vs template):", "PASS" if ok else "FAIL")

    # check 2: provider TP == EDT TP for a random hard emission in one world
    c = cells[0]
    ws = worlds.generate_worlds(c, n_worlds=3, seed=63000)
    uni2 = instrument.build_universe([test_mask], foot.shape)
    cwd = instrument.prepare_cell_worlds(c, ws, uni2, foot.shape)
    S = np.zeros(uni2.N, dtype=bool)
    flat = np.flatnonzero(test_mask.ravel())
    S[np.searchsorted(uni2.pos_flat, flat)] = True
    sub = test_mask[c.bbox] & c.active_sub
    dp = distance_transform_edt(~sub) if sub.any() else None
    ok2 = True
    for wi, w in enumerate(ws):
        rr, cc = w.truth_coords
        tp_edt = float(kernel(dp[rr, cc]).sum()) if rr.size and dp is not None else 0.0
        ps = instrument.provider_scan(cwd, S, w.truth_coords)
        tp_prov = float(ps.best.sum())
        d = abs(tp_edt - tp_prov)
        ok2 &= d < 1e-8
        print(f"world {wi}: TP_edt={tp_edt:.6f} TP_provider={tp_prov:.6f} absdiff={d:.2e}")
    print("CHECK 2 (provider TP == EDT TP):", "PASS" if ok2 else "FAIL")

    # check 3: kmax grid == kernel(EDT to truth) away from bbox border
    tg = np.zeros_like(c.active_sub)
    rr, cc = ws[0].truth_coords
    tg[rr, cc] = True
    kg = instrument._kmax_grid(tg)
    de = distance_transform_edt(~tg)
    kref = kernel(de).astype(np.float32)
    d3 = np.abs(kg - kref)
    # away from the 3px border band the two must agree exactly
    interior = np.zeros_like(tg)
    interior[4:-4, 4:-4] = True
    md = float(d3[interior].max())
    print(f"CHECK 3 (kmax vs EDT-kernel, interior max abs diff = {md:.2e}):", "PASS" if md < 1e-6 else "FAIL")

    if not (ok and ok2 and md < 1e-6):
        sys.exit(1)


if __name__ == "__main__":
    main()
