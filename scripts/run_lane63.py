"""Lane 63 orchestration: E1 canary+baseline, E2 CVaR selection, E3 prune+ship.

Run with the repo venv python and PYTHONPATH=vendor/template66:src.

    python scripts/run_lane63.py e1
    python scripts/run_lane63.py e2
    python scripts/run_lane63.py e3

Evidence JSONs land in evidence/; the final validated submission lands in
submissions/ and docs/downloads/.  No leaderboard number is ever written;
holdout numbers carry the evaluator version and withheld-positive count.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import grid as G                                   # noqa: E402
from gems57 import holdout as H                                # noqa: E402
from gems57 import evaluate_holdout as EH                      # noqa: E402
from gems57 import submission_writer as SW                     # noqa: E402
from gems57 import validate as V                               # noqa: E402
from gems57 import uniqueness as UQ                            # noqa: E402
from gems63 import canary, cvar, policies, worlds              # noqa: E402

EV = ROOT / "evidence"
EV.mkdir(exist_ok=True)
N_WORLDS = 20
Q_TAIL = 0.2
LAM = 0.5


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump(name: str, obj) -> Path:
    p = EV / name
    p.write_text(json.dumps(obj, indent=1, allow_nan=False, default=str) + "\n")
    print(f"[evidence] {p}")
    return p


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_specs() -> list[policies.PolicySpec]:
    specs = []
    for d in (2.4, 2.8, 3.2):
        for r in (0, 1):
            for b in (1.0, 0.8):
                specs.append(policies.PolicySpec(name="", spacing=d, ribbon=r,
                                                 budget_frac=b))
    specs.append(policies.PolicySpec(name="", spacing=2.8, tips=1))
    specs.append(policies.PolicySpec(name="", spacing=2.8, tips=2))
    specs.append(policies.PolicySpec(name="", spacing=2.8, stepover_gap=3.5))
    specs.append(policies.PolicySpec(name="", spacing=2.8, tips=1, stepover_gap=3.5))
    return specs


def e1():
    """Canaries + baseline holdout DTI with the shared v2 evaluator."""
    t0 = time.time()
    g = G.load_grid()
    ctx = H.build_holdout(g, seeds=(20, 21), modes=("all", "detached"))

    spec = policies.PolicySpec(name="baseline", spacing=2.8)
    base_mask, diag = policies.dotted_ridge(g.catalogue, g.footprint, spec,
                                            return_diagnostics=True)
    print("[e1] baseline dots:", int(base_mask.sum()), diag)

    # pooled hide-and-recover on the shared instrument (all cells)
    pooled = H.evaluate(base_mask, ctx)
    pooled_all = H.evaluate(base_mask, ctx, cell_mode="all")

    # official-formula folds with block-bootstrap CI
    folds, quad = H.buffered_component_draw(g, seed=20)
    terms_by_arm = {}
    per_fold = {}
    pred = base_mask.astype(np.float64)
    all_terms = []
    for f in folds:
        r, terms = EH.evaluate(pred, f, valid=g.footprint,
                               global_shape=g.footprint.shape)
        per_fold[f["name"]] = dict(dti=r["dti"], n_truth=r["n_truth"],
                                   n_emitted=r["n_emitted"])
        all_terms.append(terms)
    terms_by_arm["baseline_d28"] = np.concatenate(all_terms)
    summary = EH.pooled_summary(terms_by_arm, candidate="baseline_d28")
    out = dict(generated_utc=now(),
               evidence_class="HOLDOUT-DTI",
               evaluator_version=EH.VERSION,
               implementation_sha256=EH.implementation_hashes(),
               candidate="baseline ring-lane dotted ridge d2.8 on visible catalogue",
               spec=dict(kind="dotted", spacing=2.8, ribbon=0, tips=0,
                         stepover_gap=0.0, budget_frac=1.0),
               baseline_dots=int(base_mask.sum()),
               diagnostics=diag,
               ctx_pooled_all_modes=dict(pooled_dti=pooled["pooled_dti"],
                                         pooled_coverage=pooled["pooled_coverage"],
                                         pooled_n_truth=pooled["pooled_n_truth"]),
               ctx_pooled_mode_all=dict(pooled_dti=pooled_all["pooled_dti"],
                                        pooled_coverage=pooled_all["pooled_coverage"],
                                        pooled_n_truth=pooled_all["pooled_n_truth"],
                                        per_fold_dti=pooled_all["per_fold_dti"]),
               buffered_folds=per_fold,
               pooled_summary=summary,
               seconds=time.time() - t0)
    dump("e1_baseline_holdout.json", out)

    # leakage canaries
    import rasterio
    rad_field = None
    rad_receipt = None
    rp = ROOT / "data" / "external" / "geodawn_rad_u8.tif"
    if rp.exists():
        dots_r, rad_receipt = policies.rad_ratio_maxima(str(rp), g.footprint,
                                                        g.catalogue, n_dots=1)
        if rad_receipt["usable"]:
            with rasterio.open(rp) as ds:
                bands = ds.read()
            k = bands[rad_receipt["k_band"]].astype(np.float64)
            th = bands[rad_receipt["th_band"]].astype(np.float64)
            valid = (th > 3.0)
            ratio = np.zeros_like(k)
            ratio[valid] = np.log1p(k[valid]) / np.log1p(th[valid])
            mu, sd = ratio[valid].mean(), (ratio[valid].std() or 1.0)
            rad_field = np.where(valid, (ratio - mu) / sd, -9.0)
    can = canary.run_canaries(ctx.cells, ctx.hidden_by_cell, g, rad_field=rad_field)
    can["generated_utc"] = now()
    can["rad_band_receipt"] = rad_receipt
    dump("e1_canaries.json", can)
    return out


def e2():
    """Label worlds + CVaR selection across the policy grid."""
    t0 = time.time()
    g = G.load_grid()
    ctx = H.build_holdout(g, seeds=(20, 21), modes=("all",))
    cells = [c for c in ctx.cells if c.mode == "all"]

    worlds_by_cell = {}
    for cell in cells:
        worlds_by_cell[cell.key] = worlds.build_worlds_for_cell(
            cell, ctx.hidden_by_cell[cell.key], g.catalogue, N_WORLDS)
    wsum = worlds.world_summary(worlds_by_cell)
    dump("e2_worlds_summary.json", dict(generated_utc=now(), **wsum,
                                        n_worlds=N_WORLDS, q_tail=Q_TAIL))

    def _record(nm, mask, spec_dict, diag_extra):
        wt = cvar.world_terms(cells, worlds_by_cell, mask)
        results[nm] = dict(spec=spec_dict,
                           dots=int(mask.sum()), **diag_extra,
                           mean_world_dti=wt["mean"], cvar20=wt["cvar20"],
                           objective=wt["objective"],
                           per_world_dti=[round(v, 6) for v in wt["dtis"]])
        print(f"[e2] {nm}: dots={int(mask.sum())} mean={wt['mean']:.5f} "
              f"cvar20={wt['cvar20']:.5f} obj={wt['objective']:.5f}", flush=True)

    results = {}
    for spec in base_specs():
        nm = spec.short()
        mask, diag = policies.dotted_ridge(g.catalogue, g.footprint, spec,
                                           return_diagnostics=True)
        _record(nm, mask,
                dict(kind=spec.kind, spacing=spec.spacing, ribbon=spec.ribbon,
                     tips=spec.tips, stepover_gap=spec.stepover_gap,
                     budget_frac=spec.budget_frac),
                dict(bridges=diag["bridges"]))

    # hypothesis H63-B candidate: radiometric K/Th maxima (validated here,
    # never gets a submission slot on its own merit without beating the pool)
    rp = ROOT / "data" / "external" / "geodawn_rad_u8.tif"
    if rp.exists():
        rad_mask, rad_receipt = policies.rad_ratio_maxima(
            str(rp), g.footprint, g.catalogue, n_dots=43591)
        if rad_receipt["usable"]:
            _record("rad-ratio-maxima-h63b", rad_mask,
                    dict(kind="rad_ratio"),
                    dict(rad_receipt=rad_receipt))

    # lambda sensitivity of the ranking
    ranks = {}
    for lam in (0.25, 0.5, 0.75):
        scored = []
        for nm, r in results.items():
            dtis = np.array(r["per_world_dti"])
            scored.append((nm, float(cvar.objective(dtis, lam, Q_TAIL))))
        scored.sort(key=lambda kv: -kv[1])
        ranks[str(lam)] = [k for k, _ in scored]
    order = ranks[str(LAM)]
    winner = order[0]
    out = dict(generated_utc=now(), evidence_class="WORLD-DTI",
               note="per-world DTI over label-perturbed hide-and-recover truths; NOT an organiser score, NOT a leaderboard projection",
               evaluator_kernel="gems57.metric exact triangular kernel alpha=0.2 beta=0.8 R=3px",
               n_worlds=N_WORLDS, q_tail=Q_TAIL, lam=LAM,
               lam_sensitivity_top3={k: v[:3] for k, v in ranks.items()},
               results=results, ranking=order, winner=winner,
               seconds=time.time() - t0)
    dump("e2_cvar_selection.json", out)
    return out


def e3():
    """Greedy CVaR pruning of the E2 winner, ship + validate + uniqueness."""
    t0 = time.time()
    e2_out = json.loads((EV / "e2_cvar_selection.json").read_text())
    winner = e2_out["winner"]
    wspec = e2_out["results"][winner]["spec"]

    g = G.load_grid()
    ctx = H.build_holdout(g, seeds=(20, 21), modes=("all",))
    cells = [c for c in ctx.cells if c.mode == "all"]
    worlds_by_cell = {}
    for cell in cells:
        worlds_by_cell[cell.key] = worlds.build_worlds_for_cell(
            cell, ctx.hidden_by_cell[cell.key], g.catalogue, N_WORLDS)

    if wspec.get("kind") == "rad_ratio":
        rp = ROOT / "data" / "external" / "geodawn_rad_u8.tif"
        mask0, rad_receipt = policies.rad_ratio_maxima(
            str(rp), g.footprint, g.catalogue, n_dots=43591)
        diag0 = dict(dots=int(mask0.sum()), bridges=0, rad_receipt=rad_receipt)
    else:
        spec = policies.PolicySpec(name=winner, **wspec)
        mask0, diag0 = policies.dotted_ridge(g.catalogue, g.footprint, spec,
                                             return_diagnostics=True)
    wt0 = cvar.world_terms(cells, worlds_by_cell, mask0)
    print(f"[e3] winner {winner} dots={diag0['dots']} mean={wt0['mean']:.5f} "
          f"cvar={wt0['cvar20']:.5f}")

    pr = cvar.greedy_prune(cells, worlds_by_cell, mask0, lam=LAM, q=Q_TAIL,
                           max_iter=3000)
    dots_final = pr["dots"] & g.footprint & ~g.catalogue
    wt1 = cvar.world_terms(cells, worlds_by_cell, dots_final)
    print(f"[e3] pruned dots={int(dots_final.sum())} removed={pr['n_removed']} "
          f"mean={wt1['mean']:.5f} cvar={wt1['cvar20']:.5f}")

    # ---- holdout DTI of the final raster on the untouched instrument ----
    pooled_final = H.evaluate(dots_final, ctx, cell_mode="all")
    pooled_base = H.evaluate(mask0, ctx, cell_mode="all")
    folds, quad = H.buffered_component_draw(g, seed=20)
    arms = {}
    for arm_name, arr in (("baseline_" + winner.replace("-", "_"), mask0),
                          ("cvar_pruned", dots_final)):
        terms = []
        for f in folds:
            r, t = EH.evaluate(arr.astype(np.float64), f, valid=g.footprint,
                               global_shape=g.footprint.shape)
            terms.append(t)
        arms[arm_name] = np.concatenate(terms)
    summary = EH.pooled_summary(arms, candidate="cvar_pruned")

    # ---- submission write (fail-closed writer from the shared template) ----
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred = np.zeros(g.footprint.shape, np.float32)
    pred[dots_final] = 1.0
    name = f"gems63-cvar-pruned-{winner.replace('.', 'p')}-{int(dots_final.sum())}-{ts[:13].replace(':', '')}Z"
    note = "lane63 CVaR world-selection + greedy prune on hide-recover holdout; all-finite zeros export"
    sub_path = ROOT / "submissions" / f"{name}-allfinite.tif"
    receipt = SW.write_submission(sub_path, pred, ROOT / "data" / "bridge" / "sample_submission.tif",
                                  g.footprint, note=note, name=name,
                                  catalogue=g.catalogue,
                                  metadata=dict(lane="63-cvar-robust-selection",
                                                winner_policy=winner,
                                                lam=LAM, q=Q_TAIL,
                                                n_worlds=N_WORLDS))
    val = V.validate(sub_path, g.footprint, g.catalogue)
    V.assert_submittable(val)

    dump("e3_prune_holdout.json", dict(
        generated_utc=now(), evidence_class="HOLDOUT-DTI",
        evaluator_version=EH.VERSION,
        implementation_sha256=EH.implementation_hashes(),
        winner_policy=winner, spec=wspec,
        dots_before=int(mask0.sum()), dots_after=int(dots_final.sum()),
        removed=pr["n_removed"],
        world_dti_before=dict(mean=wt0["mean"], cvar20=wt0["cvar20"],
                              objective=wt0["objective"]),
        world_dti_after=dict(mean=wt1["mean"], cvar20=wt1["cvar20"],
                             objective=wt1["objective"]),
        trajectory=pr["trajectory"][:: max(1, len(pr["trajectory"]) // 60)],
        ctx_pooled_final=pooled_final, ctx_pooled_pre_prune=pooled_base,
        pooled_summary=summary, lam=LAM, q=Q_TAIL, n_worlds=N_WORLDS,
        seconds=time.time() - t0))
    dump("e3_submission_receipt.json", receipt)
    dump("e3_validator_output.json", val)

    # copy into docs/downloads for the pages site
    import shutil
    for suffix in (".tif", ".zip", ".json"):
        src = sub_path.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, ROOT / "docs" / "downloads" / src.name)
    print(f"[e3] submission {sub_path.name} written and locally validated")
    return dict(name=name, receipt=receipt, val=val, dots=int(dots_final.sum()),
                pred=pred, pooled_summary=summary, spec=wspec, winner=winner,
                wt_before=(wt0["mean"], wt0["cvar20"]),
                wt_after=(wt1["mean"], wt1["cvar20"]),
                pooled_final=pooled_final, pooled_base=pooled_base)


def unique():
    """Registry uniqueness scan of the shipped raster (all downloaded blobs)."""
    import rasterio
    subs = sorted((ROOT / "submissions").glob("*-allfinite.tif"))
    if not subs:
        raise SystemExit("no submission found; run e3 first")
    mine_path = subs[-1]
    with rasterio.open(mine_path) as ds:
        mine = ds.read(1)
    valid = np.isfinite(mine)
    my_dots = int((np.isfinite(mine) & (mine > 0)).sum())
    rows = []
    reg_dir = ROOT / "data" / "registry_rasters"
    files = sorted(reg_dir.glob("*.tif"))
    my_sha = sha256_file(mine_path)
    # blob sha -> first alias (for human-readable reporting)
    alias_map = {}
    preg = ROOT / "registry" / "prior_rasters.json"
    if preg.exists():
        for r in json.loads(preg.read_text())["rasters"]:
            alias_map.setdefault(r["git_blob_sha"],
                                 f"{r['aliases'][0]['repository']}:{r['aliases'][0]['path']}")
    print(f"[unique] scanning {len(files)} registry rasters ...", flush=True)
    n_err = 0
    worst_rho, worst_ov = -2.0, 0.0
    worst_rho_sub = worst_ov_sub = None
    for ri, fp in enumerate(files):
        if ri and ri % 50 == 0:
            print(f"[unique] {ri}/{len(files)} done "
                  f"(worst rho {worst_rho:.3f}, worst overlap {worst_ov:.3f})", flush=True)
        fsha = sha256_file(fp)
        row = dict(sha256=fsha, submission=fp.stem,
                   source=alias_map.get(fp.stem, "unknown"))
        try:
            if fsha == my_sha:
                row.update(identical_bytes=True, identical_decoded_predictions=True,
                           my_dots_within_3px_of_theirs=1.0,
                           duplicate_by_overlap=True)
                rows.append(row)
                continue
            with rasterio.open(fp) as ds:
                a = ds.read(1)
            if a.shape != mine.shape:
                row["error"] = f"shape {a.shape}"
                n_err += 1
                rows.append(row)
                continue
            m = np.asarray(a, np.float32)
            mb = np.isfinite(m) & (m > 0)
            # 3-px forward overlap via one EDT (exact Euclidean disk semantics,
            # identical decision boundary to the template's binary_dilation check)
            from scipy import ndimage as _ndi
            ov = 0.0
            if mb.any() and my_dots:
                d_theirs = _ndi.distance_transform_edt(~mb)
                my_bool = np.isfinite(mine) & (mine > 0)
                ov = float((d_theirs[my_bool] <= 3.0).mean())
            if np.array_equal(m, mine.astype(np.float32)):
                row["identical_decoded_predictions"] = True
            rho = float("nan")
            try:
                # exact Spearman shortcut: both fields binary on the footprint
                # => Spearman rho == phi coefficient (Pearson of the indicators)
                mvals = m[valid]
                minev = mine[valid]
                if len(np.unique(mvals)) <= 2 and len(np.unique(minev)) <= 2:
                    x = (mvals > 0).astype(np.float64)
                    y = (minev > 0).astype(np.float64)
                    if x.std() > 0 and y.std() > 0:
                        rho = float(np.corrcoef(x, y)[0, 1])
                elif np.ptp(m) and np.ptp(mine):
                    from scipy.stats import spearmanr
                    rho = float(spearmanr(mvals, minev).statistic)
            except Exception:
                pass
            my_bool = np.isfinite(mine) & (mine > 0)
            jac = float((my_bool & mb).sum() / max(1, int((my_bool | mb).sum())))
            row.update(my_dots_within_3px_of_theirs=ov,
                       spearman_full_footprint=rho if np.isfinite(rho) else None,
                       jaccard_dot_sets=jac,
                       duplicate_by_overlap=bool(ov > UQ.OVERLAP_LIMIT),
                       duplicate_by_rho=bool(np.isfinite(rho) and rho > UQ.RHO_LIMIT),
                       identical_bytes=False,
                       their_dots=int(mb.sum()))
            if np.isfinite(rho) and rho > worst_rho:
                worst_rho, worst_rho_sub = rho, fp.stem
            if ov > worst_ov:
                worst_ov, worst_ov_sub = ov, fp.stem
        except Exception as e:
            row["error"] = str(e)[:200]
            n_err += 1
        rows.append(row)
    checked = [r for r in rows if "error" not in r]
    flags = [r for r in checked if any(r.get(k) for k in (
        "duplicate_by_rho", "duplicate_by_overlap",
        "identical_bytes", "identical_decoded_predictions"))]
    # stratified diagnostics: universal soft surfaces vs dot-like rasters
    universal = [r for r in checked if (r.get("their_dots") or 0) >= 4_000_000]
    dotlike = [r for r in checked if 1_000 <= (r.get("their_dots") or 0) < 150_000]
    bigdot = [r for r in checked if 150_000 <= (r.get("their_dots") or 0) < 4_000_000]

    def _worst(subset, key):
        vals = [r for r in subset if r.get(key) is not None]
        if not vals:
            return None, None
        w = max(vals, key=lambda r: r[key])
        return w[key], f"{w['submission']} ({w.get('source','?')})"

    dl_ov, dl_ov_src = _worst(dotlike, "my_dots_within_3px_of_theirs")
    dl_rho, dl_rho_src = _worst(dotlike, "spearman_full_footprint")
    bd_ov, bd_ov_src = _worst(bigdot, "my_dots_within_3px_of_theirs")
    bd_rho, bd_rho_src = _worst(bigdot, "spearman_full_footprint")
    dotlike_gate_pass = bool(
        dotlike and (dl_ov is None or dl_ov <= UQ.OVERLAP_LIMIT) and
        (dl_rho is None or not np.isfinite(dl_rho) or dl_rho <= UQ.RHO_LIMIT))
    out = dict(generated_utc=now(), evidence_class="REGISTRY-MEASUREMENT",
               my_file=str(mine_path), my_dots=my_dots,
               my_file_sha256=sha256_file(mine_path),
               candidate_rank_variation=bool(len(np.unique(mine[valid])) > 1),
               rho_limit=UQ.RHO_LIMIT, overlap_limit=UQ.OVERLAP_LIMIT,
               registry_rasters_expected=len(files),
               registry_rasters_checked=len(checked),
               registry_rasters_errored=n_err,
               complete_accessible_scan=(n_err == 0),
               worst_spearman_full_footprint=worst_rho if worst_rho > -2 else None,
               worst_rho_submission=worst_rho_sub,
               worst_dot_overlap=worst_ov, worst_overlap_submission=worst_ov_sub,
               duplicate_count=len(flags),
               unique=bool(len(files) and n_err == 0 and my_dots > 0 and not flags),
               flagged=[f"{r['submission']} ({r.get('source','?')})" for r in flags][:12],
               stratification=dict(
                   universal_soft_surfaces=len(universal),
                   universal_note=">=4M positive px; literal forward-overlap against ANY dot field is 1.0 (family precedent: GEMSDOE64 evidence/registry_blocker.json universal_overlap_blocker)",
                   universal_examples=[r["source"] for r in universal[:5]],
                   dotlike_count=len(dotlike),
                   dotlike_worst_overlap=dl_ov, dotlike_worst_overlap_src=dl_ov_src,
                   dotlike_worst_rho=dl_rho, dotlike_worst_rho_src=dl_rho_src,
                   dotlike_gate_pass=dotlike_gate_pass,
                   bigdot_count=len(bigdot),
                   bigdot_worst_overlap=bd_ov, bigdot_worst_overlap_src=bd_ov_src,
                   bigdot_worst_rho=bd_rho, bigdot_worst_rho_src=bd_rho_src,
                   bigdot_gate_pass=bool(
                       (bd_ov is None or bd_ov <= UQ.OVERLAP_LIMIT) and
                       (bd_rho is None or not np.isfinite(bd_rho) or bd_rho <= UQ.RHO_LIMIT))),
               scope="GEMSDOE65 prior_rasters.json prediction-like blobs downloadable today; owner-repo scope, never organizer-complete",
               rows=sorted(rows, key=lambda r: -(r.get("my_dots_within_3px_of_theirs") or 0)),
               n_rows_total=len(rows))
    dump("e3_uniqueness.json", out)
    print(f"[unique] literal unique={out['unique']} worst_rho={worst_rho:.4f} "
          f"worst_overlap={worst_ov:.4f} errors={n_err} | "
          f"dotlike gate pass={dotlike_gate_pass} "
          f"(worst ov {dl_ov and round(dl_ov,3)} / worst rho {dl_rho and round(dl_rho,3)})")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "e1"
    dict(e1=e1, e2=e2, e3=e3, unique=unique)[cmd]()
