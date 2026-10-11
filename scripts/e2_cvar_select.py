#!/usr/bin/env python3
"""E2 - CVaR robust selection of the emission policy across label worlds (robust-selection lane).

Objective per policy theta, over N_WORLDS perturbed label worlds (pooled across the 5 spatially
blocked hide-and-recover folds; each fold uses its own out-of-fold score map):
    J(theta) = (1 - LAMBDA) * mean_w DTI_w(theta) + LAMBDA * CVaR_ALPHA({DTI_w(theta)})
with CVaR_ALPHA = mean of the worst ceil(ALPHA*W) worlds (lower tail; Rockafellar & Uryasev 2000).
LAMBDA = 0.5, ALPHA = 0.2. Selection uses worlds only; the nominal (unperturbed) holdout score is
reported but NOT used for selection.

Search: grid over family x budget q x NMS spacing s (r = 0), then greedy coordinate ascent over
flank radius r and then (q, s) neighbours. Greedy pruning is applied to the RULE parameters
(r, q, s), not to individual dots, because per-dot choices made on fold-specific score maps
do not transfer to the full-catalogue raster that is actually submitted (see README).

Outputs: registry/experiments/e2_cvar.json and registry/submission_policy.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems63_stack.dti import components_binary, dti_from_components, cvar_lower, R_PX  # noqa: E402
from gems63_stack.emit import ridge_score_vec, logit_score_vec, score_to_grid, policy_dots  # noqa: E402
from gems63_stack.features import build_feature_stack  # noqa: E402
from gems63_stack.grid import REPO, load_grid, withheld_mask, N_FOLDS  # noqa: E402
from gems63_stack.worlds import FoldSegments, make_world, nominal_truth  # noqa: E402

N_WORLDS = 40
LAMBDA = 0.5
ALPHA = 0.2
Q_GRID = [0.005, 0.01, 0.02]
S_GRID = [2, 3, 4]
R_GRID = [0, 1, 2, 3, 4]
SEED = 63


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    g = load_grid()
    X, names = build_feature_stack(g.footprint)
    ridge_vec = ridge_score_vec(X, names)
    ridge_grid = score_to_grid(ridge_vec, g.footprint)
    folds = []
    for k in range(N_FOLDS):
        fs = FoldSegments(g, k)
        wh = withheld_mask(g, k)
        logit_vec = logit_score_vec(X, names, g, fs.visible, wh, rng)
        logit_grid = score_to_grid(logit_vec, g.footprint)
        worlds = [make_world(fs, rng) for _ in range(N_WORLDS)]
        nom = nominal_truth(fs)
        folds.append(dict(k=k, fs=fs, logit=logit_grid, worlds=worlds, nominal=nom))
        print(f"fold {k}: segments={len(fs.coords)} truth_px={fs.n_truth_px} "
              f"nominal_active={nom.shape[0]} worlds_mean_px={np.mean([w.shape[0] for w in worlds]):.0f}", flush=True)
    del X
    # truth KD-trees per fold/world (cached across policies)
    truth_trees = []
    for f in folds:
        f["trees"] = [cKDTree(w.astype(np.float64)) if w.shape[0] else None for w in f["worlds"]]
        f["nom_tree"] = cKDTree(f["nominal"].astype(np.float64)) if f["nominal"].shape[0] else None
        truth_trees.append(f)

    cache = {}

    def evaluate(theta):
        fam, q, s, r = theta
        if theta in cache:
            return cache[theta]
        n_dots = 0
        dots_k = []
        for f in folds:
            sg = ridge_grid if fam == "ridge" else f["logit"]
            vis = f["fs"].visible
            dots = policy_dots(sg, g, vis, q, s, r)
            coords = np.argwhere(dots & f["fs"].active)  # never emit inside the known/visible mask
            n_dots += coords.shape[0]
            dots_k.append(cKDTree(coords.astype(np.float64)) if coords.shape[0] else None)
        # pooled components across folds per world
        W = N_WORLDS
        tp = np.zeros(W + 1)
        fp = np.zeros(W + 1)
        fn = np.zeros(W + 1)
        for kk, f in enumerate(folds):
            dt = dots_k[kk]
            dcoords = dt.data if dt is not None else np.zeros((0, 2))
            for w in range(W):
                tp_t = f["worlds"][w]
                c = _pair_components(dcoords, dt, tp_t, f["trees"][w])
                tp[w] += c[0]; fp[w] += c[1]; fn[w] += c[2]
            c = _pair_components(dcoords, dt, f["nominal"], f["nom_tree"])
            tp[W] += c[0]; fp[W] += c[1]; fn[W] += c[2]
        dti_w = np.array([dti_from_components(tp[w], fp[w], fn[w]) for w in range(W)])
        nominal = dti_from_components(tp[W], fp[W], fn[W])
        mean_w = float(dti_w.mean())
        cvar_w = cvar_lower(dti_w, ALPHA)
        J = (1 - LAMBDA) * mean_w + LAMBDA * cvar_w
        res = dict(theta=dict(family=fam, q=q, s=s, r=r), n_dots=int(n_dots),
                   mean_world_dti=mean_w, cvar_world_dti=cvar_w, J=J,
                   nominal_pooled_dti=float(nominal),
                   nominal_tp=float(tp[W]), nominal_fp=float(fp[W]), nominal_fn=float(fn[W]),
                   worlds_min=float(dti_w.min()), worlds_max=float(dti_w.max()))
        cache[theta] = res
        print(f"  {fam:5s} q={q:<6} s={s} r={r}  dots={n_dots:6d}  mean={mean_w:.4f} cvar={cvar_w:.4f} "
              f"J={J:.4f}  nominal={nominal:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        return res

    print("STAGE 1: grid over family x budget x spacing (r=0)")
    grid_results = []
    for fam in ["logit", "ridge"]:
        for q in Q_GRID:
            for s in S_GRID:
                grid_results.append(evaluate((fam, q, s, 0)))
    best = max(grid_results, key=lambda d: d["J"])
    cur = (best["theta"]["family"], best["theta"]["q"], best["theta"]["s"], 0)
    print("STAGE 2: greedy coordinate ascent over r, then (q, s) neighbours")
    improved = True
    while improved:
        improved = False
        base = evaluate(cur)
        for r in R_GRID:
            cand = (cur[0], cur[1], cur[2], r)
            if evaluate(cand)["J"] > base["J"] + 1e-9:
                cur = cand; base = evaluate(cur); improved = True
        for dq in [-1, 1]:
            i = Q_GRID.index(cur[1]) if cur[1] in Q_GRID else None
            if i is not None and 0 <= i + dq < len(Q_GRID):
                cand = (cur[0], Q_GRID[i + dq], cur[2], cur[3])
                if evaluate(cand)["J"] > base["J"] + 1e-9:
                    cur = cand; base = evaluate(cur); improved = True
        for ds in [-1, 1]:
            i = S_GRID.index(cur[2]) if cur[2] in S_GRID else None
            if i is not None and 0 <= i + ds < len(S_GRID):
                cand = (cur[0], cur[1], S_GRID[i + ds], cur[3])
                if evaluate(cand)["J"] > base["J"] + 1e-9:
                    cur = cand; base = evaluate(cur); improved = True
    chosen = evaluate(cur)

    # bootstrap CI over folds for the chosen policy's NOMINAL holdout DTI
    per_fold_counts = []
    for kk, f in enumerate(folds):
        fam, q, s, r = cur
        sg = ridge_grid if fam == "ridge" else f["logit"]
        dots = policy_dots(sg, g, f["fs"].visible, q, s, r) & f["fs"].active
        c = components_binary(np.argwhere(dots), f["nominal"])
        per_fold_counts.append((c["tp"], c["fp"], c["fn"], c["n_truth"]))
    pc = np.array(per_fold_counts)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, N_FOLDS, N_FOLDS)
        tp_b, fp_b, fn_b = pc[idx, 0].sum(), pc[idx, 1].sum(), pc[idx, 2].sum()
        boots.append(dti_from_components(tp_b, fp_b, fn_b))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    chosen["nominal_ci95_fold_bootstrap"] = [float(lo), float(hi)]
    chosen["n_withheld_positives_scored"] = int(pc[:, 3].sum())

    out = dict(
        protocol=dict(n_worlds=N_WORLDS, lambda_=LAMBDA, cvar_alpha=ALPHA, folds=N_FOLDS,
                      shift_px=4, drop_range=[0.10, 0.30], decoy_range=[0.5, 1.0], buffer_px=10,
                      kernel_R_px=R_PX, alpha_fp=0.2, beta_fn=0.8, seed=SEED),
        candidates=sorted(cache.values(), key=lambda d: -d["J"]),
        chosen=chosen,
        note=("Nominal scores are HOLDOUT-DTI (evaluator = gems63_stack.dti, exact binary DTI, 300 m triangular kernel, "
              "pooled over 5 spatial folds). Not organiser scores."),
        runtime_s=round(time.time() - t0, 1),
    )
    dest = REPO / "registry" / "experiments" / "e2_cvar.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    (REPO / "registry" / "submission_policy.json").write_text(json.dumps(
        dict(policy=chosen["theta"], selection="CVaR-robust over label worlds, E2 (registry/experiments/e2_cvar.json)",
             holdout_nominal_dti=chosen["nominal_pooled_dti"],
             holdout_nominal_ci95=chosen["nominal_ci95_fold_bootstrap"],
             n_withheld_positives=chosen["n_withheld_positives_scored"],
             evaluator="gems63_stack.dti (exact binary DTI, alpha=0.2, beta=0.8, R=300 m)"), indent=2))
    print("CHOSEN:", json.dumps(chosen["theta"]), "J=%.4f mean=%.4f cvar=%.4f nominal=%.4f CI=%s" % (
        chosen["J"], chosen["mean_world_dti"], chosen["cvar_world_dti"], chosen["nominal_pooled_dti"],
        [round(x, 4) for x in chosen["nominal_ci95_fold_bootstrap"]]))


def _pair_components(dcoords, dtree, truth_pts, truth_tree):
    """(tp, fp, fn) of one fold/world using precomputed trees (dots vs truth)."""
    n_g = truth_pts.shape[0]
    if n_g == 0:
        return 0.0, float(dcoords.shape[0]), 0.0
    if dtree is None:
        return 0.0, 0.0, float(n_g)
    d_gp, _ = dtree.query(truth_pts.astype(np.float64), k=1, distance_upper_bound=R_PX + 1e-9)
    tp = float(np.maximum(1.0 - np.where(np.isfinite(d_gp), d_gp, np.inf) / R_PX, 0.0).sum())
    d_pg, _ = truth_tree.query(dcoords.astype(np.float64), k=1, distance_upper_bound=R_PX + 1e-9)
    fp = float((1.0 - np.maximum(1.0 - np.where(np.isfinite(d_pg), d_pg, np.inf) / R_PX, 0.0)).sum())
    return tp, fp, float(n_g) - tp


if __name__ == "__main__":
    main()
