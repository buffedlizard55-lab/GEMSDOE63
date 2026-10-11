#!/usr/bin/env python3
"""GEMSDOE63 robust-selection pipeline.

One command runs: leakage canary -> corroboration posterior -> candidate scan on
the label-world ensemble -> CVaR selection -> exact greedy CVaR pruning ->
validated TIF + uniqueness + run card. Everything is deterministic (fixed seeds)
and logged to evidence/.

Budget protocol: experiment 1 = canary+solo weights; experiment 2 = candidate
world-scan + selection; experiment 3 = greedy prune + ship. Stop after three.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems63 import emission, holdout, instrument, posterior, uniqueness, writer, worlds
from gems63.metric import ALPHA, BETA, EPS, _OFFS, dti_from_counts, kernel

EVID = ROOT / "evidence"
EVID.mkdir(exist_ok=True)
OUT = ROOT / "docs" / "downloads"
OUT.mkdir(parents=True, exist_ok=True)

RUN_TS = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
SEED = 63_000
ESTIMATED_LB_HIDDEN_TRUTH_PX = 12691.0  # template calibration constant (GEMSDOE32 holdout.py)

INPUTS = {
    "h19_5": ROOT / "inputs/h19_5_nan.tif",
    "h19_4": ROOT / "inputs/h19_4_nan.tif",
    "h16_1": ROOT / "inputs/h16_1_nan.tif",
    "ens12": ROOT / "inputs/calibration/ens12_7f00890a.tif",
}
LIVE_ANCHORS = {  # name -> (path, live_score)
    "d28_repro_0.2600": (ROOT / "registry_rasters/gemsdoe32-d28-ref02600-repro-44090-20261004T183200Z-426073b6-zeros.tif", 0.2600),
    "h33_2_b2_0.2778": (ROOT / "registry_rasters/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif", 0.2778),
    "h32d_submod_46090": (ROOT / "registry_rasters/gemsdoe32-h32d-submodular-multipysics-46090-20261004T183200Z-4de30601-zeros.tif", None),
    "d15_scored_0.2477": (ROOT / "inputs/calibration/d15_scored_nan.tif", 0.2477),
    "r13_blind_lattice_0.0904": (ROOT / "inputs/calibration/r13_lattice_s5_v2_nan.tif", 0.0904),
    "h19_5_input_0.1922": (ROOT / "inputs/h19_5_nan.tif", 0.1922),
    "h19_4_input_0.1894": (ROOT / "inputs/h19_4_nan.tif", 0.1894),
    "h16_1_input_0.1855": (ROOT / "inputs/h16_1_nan.tif", 0.1855),
    "ens12_input_0.1563": (ROOT / "inputs/calibration/ens12_7f00890a.tif", 0.1563),
}
HARD_ANCHORS = ("d28_repro_0.2600", "h33_2_b2_0.2778", "h32d_submod_46090")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    r = rankdata(scores)
    n1 = int(labels.sum())
    n0 = int(labels.size - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def soft_exact_baseline(px: np.ndarray, cells, foot) -> tuple:
    """Exact soft DTI vs each cell's unperturbed hidden truth; returns (mean, per_cell).

    TP = per-truth max over kernel offsets of p*k ; FP = sum_active p*(1-kmax).
    """
    per_cell = []
    for cell in cells:
        sl = cell.bbox
        psub = px[sl].astype(np.float64)
        act = cell.active_sub
        bt = np.zeros_like(act)
        for rr, cc in cell.hidden_comp_coords:
            keep = act[rr, cc]
            bt[rr[keep], cc[keep]] = True
        n_t = int(bt.sum())
        if n_t == 0:
            per_cell.append(0.0)
            continue
        # kmax of truth
        kmaxg = np.zeros_like(psub)
        trr, tcc = np.nonzero(bt)
        credit = np.zeros(n_t, dtype=np.float64)
        for dy, dx, k in _OFFS:
            # truth-side credit
            nr = trr + dy
            nc = tcc + dx
            ok = (nr >= 0) & (nr < psub.shape[0]) & (nc >= 0) & (nc < psub.shape[1])
            np.maximum.at(credit, np.flatnonzero(ok), psub[nr[ok], nc[ok]] * k)
            # x-side kmax
            shifted = np.zeros_like(act)
            ys = slice(max(0, dy), act.shape[0] + min(0, dy))
            xs = slice(max(0, dx), act.shape[1] + min(0, dx))
            yd = slice(max(0, -dy), act.shape[0] + min(0, -dy))
            xd = slice(max(0, -dx), act.shape[1] + min(0, -dx))
            shifted[yd, xd] = bt[ys, xs]
            np.maximum(kmaxg, shifted.astype(np.float64) * k, out=kmaxg)
        tp = float(credit.sum())
        fp = float((psub[act] * (1.0 - kmaxg[act])).sum())
        per_cell.append(dti_from_counts(tp * 1.0, fp, n_t))
    return float(np.mean(per_cell)), per_cell


def load_bool_or_support(path: Path, foot: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1)
    return np.isfinite(a) & (a > 0) & foot


def main():
    t0 = time.time()
    log("loading grid + labels + sgmc")
    foot, labels, sgmc_off, near3, transform, crs = holdout.load_grid(ROOT / "data")
    collar2 = binary_dilation(labels, iterations=2)  # the proven catalogue-collar exclusion
    allowed_all = foot & ~collar2

    log("building hide-and-recover cells (draws 20/21 x 4 quadrants, hide 20%)")
    cells = holdout.build_cells(foot, labels)
    n_withheld = sum(c.n_hidden_truth for c in cells)
    log(f"cells built: {len(cells)}; withheld positive pixels total = {n_withheld}")

    # ---------------- experiment 1: leakage canary + solo holdout weights ----------------
    log("EXPERIMENT 1: leakage canary (solo AUC vs hidden truth) + solo baseline DTI weights")
    canary = {}
    surfaces = {}
    for name, path in INPUTS.items():
        s = posterior.load_soft(path, foot)
        surfaces[name] = s
        ys_list, xs_list = [], []
        for cell in cells:
            sl = cell.bbox
            act = cell.active_sub
            bt = np.zeros_like(act)
            for rr, cc in cell.hidden_comp_coords:
                keep = act[rr, cc]
                bt[rr[keep], cc[keep]] = True
            ys_list.append(bt[act])
            xs_list.append(s[sl][act])
        y = np.concatenate(ys_list)
        x = np.concatenate(xs_list)
        auc = auc_mann_whitney(x, y)
        solo_mean, solo_cells = soft_exact_baseline(s, cells, foot)
        canary[name] = {"solo_auc_vs_hidden_baseline": auc, "solo_catalogue_hidden_mean_dti": solo_mean,
                        "solo_per_cell": solo_cells,
                        "leakage_flag": bool(auc > 0.90),
                        "verdict": "LEAKAGE-SUSPECT" if auc > 0.90 else "TRUSTABLE"}
        log(f"  {name}: AUC={auc:.4f} solo DTI={solo_mean:.5f} -> {canary[name]['verdict']}")
    flagged = [k for k, v in canary.items() if v["leakage_flag"]]
    if flagged:
        log(f"  WARNING: leakage-suspect surfaces excluded from posterior: {flagged}")
    usable = {k: v for k, v in surfaces.items() if not canary[k]["leakage_flag"]}
    weights = {k: max(canary[k]["solo_catalogue_hidden_mean_dti"], 0.002) for k in usable}
    (EVID / "canary.json").write_text(json.dumps(canary, indent=1))

    # ---------------- posterior + candidates ----------------
    log("building corroboration posterior (weights from solo holdout, collar-excluded)")
    post = posterior.build_posterior(usable, weights, foot, collar2)
    P = post["P"]
    ys, xs = [], []
    for cell in cells:
        sl = cell.bbox
        act = cell.active_sub
        bt = np.zeros_like(act)
        for rr, cc in cell.hidden_comp_coords:
            keep = act[rr, cc]
            bt[rr[keep], cc[keep]] = True
        ys.append(bt[act])
        xs.append(P[sl][act])
    p_auc = auc_mann_whitney(np.concatenate(xs), np.concatenate(ys))
    log(f"  posterior AUC={p_auc:.4f}; weights={ {k: round(v,4) for k,v in post['weights'].items()} }")
    (EVID / "posterior.json").write_text(json.dumps(
        {"weights": post["weights"], "thresholds": post["thresholds"], "auc": p_auc}, indent=1))

    P_aug = posterior.sgmc_augment(P, sgmc_off, allowed_all)
    # ens12-free posterior variant (flagged-control decorrelation insurance; IR-63-REG-01)
    usable_ne = {k: v for k, v in usable.items() if k != "ens12"}
    weights_ne = {k: weights[k] for k in usable_ne}
    P_ne = posterior.build_posterior(usable_ne, weights_ne, foot, collar2)["P"] if usable_ne else P
    pos_vals = P[foot & (P > 0)]
    support_tau = float(np.median(pos_vals)) if pos_vals.size else 0.0
    log("emitting candidates (posterior-ordered thinning; spacings/budgets/anti-speckle)")
    cands = emission.build_candidates(P, P_aug, allowed_all, support_tau)
    for s in (2.83, 3.17):
        cands[f"ne-thin-s{s:.2f}"] = emission.thin_ordered(P_ne, allowed_all, min_sep=s)
    # clean shipped artifacts of earlier runs in this session dir (keep run deterministic outputs only)
    for pat in ("gemsdoe63-*", "checks-gemsdoe63-*"):
        for old in OUT.glob(pat):
            old.unlink()
    cand_info = {k: int(v.sum()) for k, v in cands.items()}
    log("  candidate dot counts: " + json.dumps(cand_info))

    # anchors for instrument calibration + uniqueness context
    anchors = {}
    for name, (path, live) in LIVE_ANCHORS.items():
        anchors[name] = load_bool_or_support(path, foot)

    # ---------------- universe + worlds + heavy precompute ----------------
    log("building universe (union of candidates + hard-anchor supports)")
    universe_masks = list(cands.values()) + [anchors[n] for n in HARD_ANCHORS]
    uni = instrument.build_universe(universe_masks, foot.shape)
    log(f"  universe N = {uni.N:,} px")

    log(f"generating label worlds (shift +/-4px, drop 10-30%, decoys @ observed density; seed {SEED})")
    cell_worlds = [worlds.generate_worlds(c, n_worlds=instrument.N_WORLDS, seed=SEED) for c in cells]
    wstats = [{"cell": c.key, "n_worlds": len(ws),
               "mean_drop": float(np.mean([w.drop_frac for w in ws])),
               "mean_decoys": float(np.mean([w.n_decoys for w in ws])),
               "mean_truth_px": float(np.mean([w.truth_coords[0].size for w in ws]))}
              for c, ws in zip(cells, cell_worlds)]
    log("precomputing per-(cell,world) kmax at universe dots (the heavy pass)")
    cwds = []
    for c, ws in zip(cells, cell_worlds):
        cwd = instrument.prepare_cell_worlds(c, ws, uni, foot.shape)
        cwds.append(cwd)
    (EVID / "worlds_stats.json").write_text(json.dumps(wstats, indent=1))

    # ---------------- experiment 2: candidate scan + CVaR selection ----------------
    log("EXPERIMENT 2: scoring candidates in every world with the exact metric")
    grid_shape = foot.shape

    def S_of_mask(mask):
        idx = np.searchsorted(uni.pos_flat, np.flatnonzero(mask.ravel()))
        S = np.zeros(uni.N, dtype=bool)
        S[idx] = True
        return S

    # prevalence-calibrated scale factors (same worlds for every candidate)
    _tmp = instrument.evaluate_emission(next(iter(cands.values())), cwds)
    world_n = _tmp["world_n"]
    s_cal = instrument.prevalence_scale_factors(world_n, cwds, ESTIMATED_LB_HIDDEN_TRUTH_PX)

    scan = {}
    for name, mask in cands.items():
        r = instrument.evaluate_emission(mask, cwds)
        v_cal = instrument.dti_prevalence_calibrated(r["world_tp"], r["world_fp"], world_n, s_cal)
        j_cal, m_cal, w_cal = instrument.j_from_dti(v_cal)
        r = {k: v for k, v in r.items() if k not in ("world_dtis", "world_tp", "world_fp", "world_n", "domain_share")}
        r["J_calibrated"] = float(j_cal)
        r["mean_calibrated"] = float(m_cal)
        r["worst20_calibrated"] = float(w_cal)
        r["emitted_pixels"] = int(mask.sum())
        scan[name] = r
        log(f"  {name:22s} dots={r['emitted_pixels']:6d} J_cal={r['J_calibrated']:.5f} "
            f"(J_nat={r['J']:.5f}) mean_cal={r['mean_calibrated']:.5f} worst20_cal={r['worst20_calibrated']:.5f} "
            f"catHidden={r['baseline_catalogue_hidden_mean']:.5f}")
    (EVID / "candidate_scan.json").write_text(json.dumps(scan, indent=1))
    best = max(scan, key=lambda k: scan[k]["J_calibrated"])
    log(f"  SELECTED by prevalence-calibrated CVaR J_cal (G={ESTIMATED_LB_HIDDEN_TRUTH_PX:.0f}): {best}")

    # hard anchors through the same instrument (calibration context)
    log("scoring hard anchors through the same label-world instrument")
    anchor_J = {}
    for name in HARD_ANCHORS:
        r = instrument.evaluate_emission(anchors[name], cwds)
        v_cal = instrument.dti_prevalence_calibrated(r["world_tp"], r["world_fp"], world_n, s_cal)
        j_cal, m_cal, w_cal = instrument.j_from_dti(v_cal)
        anchor_J[name] = {k: v for k, v in r.items() if k not in ("world_dtis", "world_tp", "world_fp", "world_n", "domain_share")}
        anchor_J[name]["J_calibrated"] = float(j_cal)
        anchor_J[name]["mean_calibrated"] = float(m_cal)
        anchor_J[name]["emitted_pixels"] = int(anchors[name].sum())
        log(f"  {name:22s} dots={anchor_J[name]['emitted_pixels']:6d} J_cal={j_cal:.5f} "
            f"(J_nat={r['J']:.5f}) live={LIVE_ANCHORS[name][1]}")

    # soft-exact catalogue-hidden calibration for ALL anchors (live-score table)
    log("instrument-vs-live calibration table (soft-exact catalogue-hidden baseline)")
    calib = {}
    for name, (path, live) in LIVE_ANCHORS.items():
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, 0, 1)
        arr[~foot] = 0
        m, pc = soft_exact_baseline(arr, cells, foot)
        calib[name] = {"live": live, "holdout_catalogue_hidden_mean": m, "per_cell": pc}
    scored = [(v["holdout_catalogue_hidden_mean"], v["live"]) for v in calib.values() if v["live"] is not None]
    if len(scored) >= 4:
        ins = np.array([s[0] for s in scored])
        liv = np.array([s[1] for s in scored])
        ri = rankdata(ins)
        rl = rankdata(liv)
        rho = float(np.corrcoef(ri, rl)[0, 1])
        pear = float(np.corrcoef(ins, liv)[0, 1])
    else:
        rho = pear = None
    calib_summary = {"per_anchor": calib, "spearman_instrument_vs_live": rho, "pearson": pear,
                     "note": "catalogue-hidden instrument is optimistic for surfaces trained on the full catalogue; see irregularities"}
    (EVID / "anchors_calibration.json").write_text(json.dumps(calib_summary, indent=1))
    log(f"  instrument-vs-live spearman rho = {rho}")

    # ---------------- experiment 3: exact greedy CVaR pruning ----------------
    log("EXPERIMENT 3: exact greedy CVaR pruning of the selected candidate")
    sel_mask = cands[best]
    S = S_of_mask(sel_mask)

    # state: providers per (cell, world); objective = prevalence-calibrated CVaR J_cal
    s_by_cell = []
    n_w_per_cell = len(cwds[0].worlds)
    for ci, cwd in enumerate(cwds):
        s_by_cell.append(s_cal[ci * n_w_per_cell:(ci + 1) * n_w_per_cell])

    def build_state(S):
        st = []
        for ci, cwd in enumerate(cwds):
            cst = {"tp": np.zeros(len(cwd.worlds)), "fp": None, "n": None, "s": s_by_cell[ci], "prov": []}
            Sdot = S[cwd.cidx]
            cst["fp"] = ((1.0 - cwd.kvals) * Sdot[None, :]).sum(axis=1)
            cst["n"] = np.array([w.truth_coords[0].size for w in cwd.worlds], dtype=np.int64)
            for wi, w in enumerate(cwd.worlds):
                ps = instrument.provider_scan(cwd, S, w.truth_coords)
                cst["prov"].append(ps)
                cst["tp"][wi] = ps.best.sum()
            st.append(cst)
        return st

    def dti_vector_and_J(st):
        vs = []
        for cst in st:
            tp, fp, n, s = cst["tp"], cst["fp"], cst["n"].astype(np.float64), cst["s"]
            v = instrument.dti_prevalence_calibrated(tp, fp, n, s)
            vs.append(v)
        v = np.concatenate(vs)
        j, m, wc = instrument.j_from_dti(v)
        return j, m, wc, v

    st = build_state(S)
    J0, m0, w0, _ = dti_vector_and_J(st)
    log(f"  pre-prune: J_cal={J0:.5f} mean={m0:.5f} worst20={w0:.5f} dots={int(S.sum())}")
    prune_log = [{"commit": 0, "J_cal": J0, "mean": m0, "worst20": w0, "dots": int(S.sum()), "removed": 0}]

    MAX_COMMITS = 12
    for commit in range(1, MAX_COMMITS + 1):
        # per-dot exact removal deltas (top-two structures are exact for single removals)
        dot_gain = np.zeros(uni.N) - np.inf
        for ci, cwd in enumerate(cwds):
            cst = st[ci]
            tp_base, fp_base, n_base, s_b = cst["tp"], cst["fp"], cst["n"].astype(np.float64), cst["s"]
            dti_base = instrument.dti_prevalence_calibrated(tp_base, fp_base, n_base, s_b)
            # TP loss per dot per world
            loss = np.zeros((len(cwd.worlds), len(cwd.cidx)))
            for wi, ps in enumerate(cst["prov"]):
                if ps.n_truth == 0:
                    continue
                valid = ps.prov >= 0
                if not valid.any():
                    continue
                pg = ps.prov[valid]
                # map global universe idx -> local cidx position
                pos = np.searchsorted(cwd.cidx, pg)
                pos_ok = (pos < len(cwd.cidx)) & (cwd.cidx[np.minimum(pos, len(cwd.cidx) - 1)] == pg)
                contrib = (ps.best - ps.second)[valid][pos_ok]
                np.add.at(loss[wi], pos[pos_ok], contrib)
            dfp = (1.0 - cwd.kvals)  # (worlds, dots)
            tp_new = tp_base[:, None] - loss
            fp_new = fp_base[:, None] - dfp
            dti_new = instrument.dti_prevalence_calibrated(
                np.clip(tp_new, 0, None), np.clip(fp_new, 0, None), n_base[:, None] + np.zeros_like(tp_new), s_b[:, None])
            gain = (dti_new - dti_base[:, None]).sum(axis=0)  # summed over worlds -> comparable
            dot_gain[cwd.cidx] = gain
        order = np.argsort(-dot_gain)
        order = order[np.isfinite(dot_gain[order]) & (dot_gain[order] > 0)]
        if order.size == 0:
            log(f"  commit {commit}: no improving dot; stop")
            break
        tried = []
        bestJ = J0
        bestB = None
        for bsz in (1, 25, 100, 250, 500, 1000):
            B = order[:bsz]
            S2 = S.copy()
            S2[B] = False
            st2 = build_state(S2)
            J2, m2, w2, _ = dti_vector_and_J(st2)
            tried.append({"batch": int(bsz), "J": J2, "mean": m2, "worst20": w2})
            if J2 > bestJ + 1e-12:
                bestJ = J2
                bestB = (B, st2, m2, w2)
        if bestB is None:
            log(f"  commit {commit}: no batch improved exact J; stop. tried={tried}")
            break
        B, st, m2, w2 = bestB
        S[B] = False
        J0 = bestJ
        nd = int(S.sum())
        prune_log.append({"commit": commit, "J_cal": float(J0), "mean": float(m2), "worst20": float(w2),
                          "dots": nd, "removed": int(B.size)})
        log(f"  commit {commit}: removed {B.size} dots -> J_cal={J0:.5f} mean={m2:.5f} worst20={w2:.5f} dots={nd}")
    (EVID / "prune_log.json").write_text(json.dumps(prune_log, indent=1))

    final_mask = np.zeros(grid_shape, dtype=bool)
    final_pos = uni.pos_flat[S]
    final_mask.ravel()[final_pos] = True
    _fe = instrument.evaluate_emission(final_mask, cwds)
    v_cal_f = instrument.dti_prevalence_calibrated(_fe["world_tp"], _fe["world_fp"], world_n, s_cal)
    j_cal_f, m_cal_f, w_cal_f = instrument.j_from_dti(v_cal_f)
    final_eval = {k: v for k, v in _fe.items() if k not in ("world_dtis", "world_tp", "world_fp", "world_n", "domain_share")}
    final_eval["J_calibrated"] = float(j_cal_f)
    final_eval["mean_calibrated"] = float(m_cal_f)
    final_eval["worst20_calibrated"] = float(w_cal_f)
    ci_cal = 1.96 * float(v_cal_f.std(ddof=1)) / np.sqrt(v_cal_f.size)
    final_eval["ci95_calibrated"] = float(ci_cal)
    log(f"  FINAL: dots={int(final_mask.sum())} J_cal={j_cal_f:.5f} mean={m_cal_f:.5f}+-CI{ci_cal:.5f} "
        f"worst20={w_cal_f:.5f} J_nat={final_eval['J']:.5f} catHidden={final_eval['baseline_catalogue_hidden_mean']:.5f}")

    # SGMC prevalence-calibrated instrument for final + hard anchors (template-comparable)
    def sgmc_cal(mask):
        off = mask & ~labels
        n_off = int(off.sum())
        if n_off == 0 or sgmc_off.sum() == 0:
            return {"n_off_catalogue": n_off, "sgmc_prevalence_calibrated_dti": 0.0}
        dp = distance_transform_edt(~off)
        sg_rr, sg_cc = np.nonzero(sgmc_off)
        cov = float(kernel(dp[sg_rr, sg_cc]).sum() / sg_rr.size)
        tp = cov * ESTIMATED_LB_HIDDEN_TRUTH_PX
        fp = max(0.0, float(n_off) - tp)
        return {"n_off_catalogue": n_off, "coverage_sgmc": cov,
                "sgmc_prevalence_calibrated_dti": float(tp / (ALPHA * (tp + fp) + BETA * ESTIMATED_LB_HIDDEN_TRUTH_PX + EPS))}

    sgmc_final = sgmc_cal(final_mask)
    sgmc_anchor = {n: sgmc_cal(anchors[n]) for n in HARD_ANCHORS}
    (EVID / "sgmc_calibrated.json").write_text(json.dumps({"final": sgmc_final, "anchors": sgmc_anchor}, indent=1))
    log(f"  SGMC-calibrated final={sgmc_final['sgmc_prevalence_calibrated_dti']:.5f} "
        f"vs h33={sgmc_anchor['h33_2_b2_0.2778']['sgmc_prevalence_calibrated_dti']:.5f} "
        f"vs d28={sgmc_anchor['d28_repro_0.2600']['sgmc_prevalence_calibrated_dti']:.5f}")

    # ---------------- write + validate + uniqueness ----------------
    values = np.zeros(grid_shape, dtype=np.float32)
    values[final_mask] = 1.0
    values[~foot] = 0.0
    vals_digest = hashlib.sha256(values.tobytes()).hexdigest()
    n_dots = int(final_mask.sum())
    short = f"gemsdoe63-cvarlw-{best.replace('.', 'p')}-{n_dots}-{RUN_TS}-{vals_digest[:8]}"
    tif = OUT / f"{short}-zeros.tif"
    receipt = writer.write_submission(values, tif)
    receipt["content_digest8"] = vals_digest[:8]
    receipt["catalogue_collision_px"] = int((final_mask & labels).sum())
    receipt["collar2_collision_px"] = int((final_mask & binary_dilation(labels, iterations=2)).sum())
    writer.write_checks(receipt, OUT / f"checks-{short}-zeros.tif.json")
    log(f"  wrote {tif.name}: sha256={receipt['sha256'][:16]}... all_checks={receipt['all_checks_passed']}")

    import zipfile
    zpath = OUT / f"{short}-zeros.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tif, tif.name)

    # nan twin for optional preference
    values_nan = np.full(grid_shape, np.nan, dtype=np.float32)
    values_nan[foot] = values[foot]
    tif_nan = OUT / f"{short}-nan.tif"
    with rasterio.open(tif_nan, "w", driver="GTiff", height=grid_shape[0], width=grid_shape[1],
                       count=1, dtype="float32", crs=writer.CRS.from_epsg(32611),
                       transform=writer.GRID_TRANSFORM, compress="deflate") as ds:
        ds.write(values_nan, 1)

    # registry + uniqueness (final raster AND posterior surface)
    log("uniqueness checks vs registry rasters")
    registry = {}
    for p in sorted((ROOT / "registry_rasters").glob("*.tif")):
        try:
            with rasterio.open(p) as ds:
                a = ds.read(1).astype(np.float32)
            if a.shape == grid_shape:
                registry[p.name] = a
        except Exception as e:
            log(f"  skip unreadable registry file {p.name}: {e}")
    for p in sorted((ROOT / "inputs").glob("**/*.tif")):
        try:
            with rasterio.open(p) as ds:
                a = ds.read(1).astype(np.float32)
            if a.shape == grid_shape:
                registry[f"inputs::{p.name}"] = a
        except Exception:
            pass
    uniq_final = uniqueness.check_uniqueness(values, foot, registry)
    uniq_surface = uniqueness.check_uniqueness(np.where(foot, P, 0.0).astype(np.float32), foot, registry)
    (EVID / "uniqueness.json").write_text(json.dumps({"final": uniq_final, "posterior_surface": uniq_surface}, indent=1))
    log(f"  final: max|rho|={uniq_final['max_abs_spearman']:.4f} max3pxOverlap={uniq_final['max_dot_overlap_3px']:.4f} -> {uniq_final['verdict']}")
    log(f"  posterior surface: max|rho|={uniq_surface['max_abs_spearman']:.4f} -> {uniq_surface['verdict']}")
    if uniq_final["verdict"] == "DUPLICATE":
        log("  LANE BREACH: duplicate of another lane; logging and stopping before promotion claims")
    if uniq_surface["verdict"] == "DUPLICATE":
        log("  SURFACE BREACH: posterior too similar to a registry raster; flagged")
    log(f"  controls flagged (excluded from gate, IR-63-REG-01): {list(uniq_final['flagged_controls'].keys())}")

    # run card
    note = (f"GEMSDOE63 CVaR-LW | {best}; {n_dots} dots; CVaR over 160 label worlds "
            f"(shift4/drop10-30/decoys), prevalence-cal J={final_eval['J_calibrated']:.4f}; "
            f"SGMC-cal {sgmc_final['sgmc_prevalence_calibrated_dti']:.4f}")
    note = note[:140]
    run_card = {
        "run_ts_utc": RUN_TS,
        "hypothesis": "H63-A/B/C: a corroboration-weighted multi-physics posterior, emitted by posterior-ordered thinning with spacing/budget chosen to maximize mean-DTI penalized by the worst-tail average (CVaR, Rockafellar-Uryasev 2000) across label worlds shifted +/-4px, dropped 10-30%, decoyed at observed density; then exact greedy CVaR pruning.",
        "mechanism": "Independent-physics coincidence localizes unmapped structures; CVaR selection hedges catalogue drift (known-fault set can change in round 2, rules 3.6.2); pruning removes dots whose cost alpha=0.2 exceeds their robust credit.",
        "non_fault_mimic": "Intersections of independent geophysical edges that are NOT faults: e.g. coincident lithologic/magnetic contacts plus terrace edges from younger sediments can mimic a fault lineament; decoy worlds price this risk into the selection.",
        "holdout": {
            "instrument": "label-world ensemble on the hide-and-recover holdout (8 cells: draws 20/21 x 4 quadrants, 20 worlds/cell = 160 cell-worlds; exact pooled DTI alpha=0.2 beta=0.8 R=300m); truth population prevalence-calibrated to the receipt-estimated hidden size G=12691px (GEMSDOE32 instrument calibration evidence)",
            "withheld_positive_px_total": n_withheld,
            "J_cvar_calibrated": final_eval["J_calibrated"],
            "mean_dti_calibrated": final_eval["mean_calibrated"],
            "ci95_calibrated": final_eval["ci95_calibrated"],
            "worst20_calibrated": final_eval["worst20_calibrated"],
            "J_cvar_natural_prevalence": final_eval["J"],
            "mean_dti_natural": final_eval["mean_dti"],
            "baseline_catalogue_hidden_mean_unperturbed": final_eval["baseline_catalogue_hidden_mean"],
            "sgmc_prevalence_calibrated_dti": sgmc_final["sgmc_prevalence_calibrated_dti"],
            "label": "HOLDOUT-DTI (this engine v0.1.0; CIs are Monte-Carlo/t across cell-worlds)",
        },
        "anchors_on_same_instrument": {
            k: {"live": LIVE_ANCHORS[k][1],
                "J_calibrated": anchor_J[k]["J_calibrated"],
                "J_natural": anchor_J[k]["J"],
                "mean_dti_natural": anchor_J[k]["mean_dti"],
                "catHidden": anchor_J[k]["baseline_catalogue_hidden_mean"],
                "sgmc_cal": sgmc_anchor[k]["sgmc_prevalence_calibrated_dti"]}
            for k in HARD_ANCHORS},
        "instrument_vs_live_spearman": rho,
        "correlation_overlap_vs_registry": {
            "max_abs_spearman": uniq_final["max_abs_spearman"],
            "max_dot_overlap_3px": uniq_final["max_dot_overlap_3px"],
            "verdict": uniq_final["verdict"],
            "breaches": uniq_final["breaches"],
        },
        "chosen_candidate": best,
        "prune_commits": len(prune_log) - 1,
        "raster": {
            "filename": tif.name,
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
            "dots": n_dots,
            "catalogue_collision_px": receipt["catalogue_collision_px"],
            "validator": receipt["checks"],
            "all_checks_passed": receipt["all_checks_passed"],
        },
        "submission_name": short,
        "submission_note_140c": note,
        "ok_to_download_and_submit": bool(receipt["all_checks_passed"] and uniq_final["verdict"] == "UNIQUE"),
        "verdict": "promote" if (receipt["all_checks_passed"] and uniq_final["verdict"] == "UNIQUE") else "negative",
        "negative_is_deliverable": True,
        "elapsed_min": round((time.time() - t0) / 60.0, 1),
    }
    (EVID / "run_card.json").write_text(json.dumps(run_card, indent=1))
    log(f"run card: verdict={run_card['verdict']} ok_to_submit={run_card['ok_to_download_and_submit']}")
    log(f"elapsed {run_card['elapsed_min']} min")
    return run_card


if __name__ == "__main__":
    main()
