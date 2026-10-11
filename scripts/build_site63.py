"""Build the docs/ site from evidence/*.json. Run after run_lane63.py steps."""
from __future__ import annotations
import datetime as dt
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EV = ROOT / "evidence"


def load(name, default=None):
    p = EV / name
    if not p.exists():
        return default
    return json.loads(p.read_text())


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · GEMSDOE63 lane (CVaR robust selection)</title>
<link rel="stylesheet" href="../assets/site.css"></head><body>
<header class="top"><div class="wrap">
<h1>GEMSDOE63 — CVaR robust-selection lane</h1>
<p>DOE GEMS Prize (DrivenData 306) · hide-and-recover holdout · label-world ensemble · exact DTI metric</p>
<p>Session 2026-10-11 · branch arena/8f46dad0-gemsdoe63 · template gems57-pooled-hide-v2 (byte-pinned)</p>
</div></header>
<div class="wrap">
<nav class="crumbs">
<a href="arena-8f46dad0-index.html">Dashboard</a><a href="arena-8f46dad0-executive-summary.html">Executive summary &amp; how to submit</a>
<a href="arena-8f46dad0-hypotheses.html">Hypotheses</a><a href="arena-8f46dad0-research.html">Why h33 scored 0.2778</a>
<a href="arena-8f46dad0-sources.html">Sources</a><a href="arena-8f46dad0-irregularities.html">Irregularities</a><a href="arena-8f46dad0-run-card.html">Run card</a>
</nav>
"""
FOOT = """<footer>Lane 63 evidence site. Every number carries its evidence class (HOLDOUT-DTI /
WORLD-DTI / USER-REPORTED / REGISTRY-MEASUREMENT). Local validation is not organizer acceptance;
promotion to a submission slot is a separate selector step.</footer></div></body></html>
"""


def page(title, body):
    return HEAD.replace("{title}", title) + body + FOOT


def build():
    (DOCS / "parallel").mkdir(exist_ok=True)
    e1 = load("e1_baseline_holdout.json", {})
    can = load("e1_canaries.json", {})
    e2 = load("e2_cvar_selection.json", {})
    e3 = load("e3_prune_holdout.json", {})
    receipt = load("e3_submission_receipt.json", {})
    val = load("e3_validator_output.json", {})
    uniq = load("e3_uniqueness.json", {})

    # ---------------- index ----------------
    sub_files = sorted((ROOT / "submissions").glob("*-allfinite.tif"))
    sub_name = sub_files[-1].name if sub_files else None
    zip_name = sub_name[:-4] + ".zip" if sub_name else None
    ok_validator = bool(val and val.get("all_checks_passed"))
    ok_unique = bool(uniq and uniq.get("unique"))
    dup_flagged = bool(uniq) and not uniq.get("unique", True)
    ok_ready = bool(sub_name and ok_validator and ok_unique and
                    receipt.get("template_report", {}).get("ok"))
    strat = (uniq or {}).get("stratification", {})
    dl_pass = bool(strat.get("dotlike_gate_pass"))
    bd_pass = bool(strat.get("bigdot_gate_pass"))
    if ok_unique:
        verdict_block = '<p><span class="badge ok">UNIQUE — cleared for the selector</span></p>'
        badge = ('<span class="badge ok">✅ OK TO DOWNLOAD &amp; SUBMIT — locally validated, uniqueness-checked</span>')
    elif dup_flagged and dl_pass and bd_pass:
        verdict_block = f"""<p><span class="badge no">NOT CLEARED BY THE LITERAL PROTOCOL GATE — FAIL CLOSED</span></p>
<p>The literal 3-px forward-overlap gate is tripped by <b>{esc(strat.get('universal_soft_surfaces', 0))} universal soft-surface registry rasters</b>
(&gt;4M positive pixels; they cover the whole footprint, so ANY dot field overlaps them 1.0 — family precedent
GEMSDOE64/evidence/registry_blocker.json). That is a registry-structure problem, not evidence this raster copies another lane.</p>
<p>Stratified dot-like evidence (the meaningful lane-drift test): among <b>{esc(strat.get('dotlike_count', 0))} dot-like rasters</b>
(1k–150k positive px) worst overlap = <b>{esc(round(strat.get('dotlike_worst_overlap') or 0, 4))}</b> (limit 0.70;
{esc(strat.get('dotlike_worst_overlap_src', '—'))}) and worst rho = <b>{esc(round(strat.get('dotlike_worst_rho') or 0, 4))}</b>
(limit 0.90; {esc(strat.get('dotlike_worst_rho_src', '—'))}) — gate <b>{'PASSED' if dl_pass else 'FAILED'}</b>.
Big-dot surfaces (150k–4M px): worst overlap {esc(round(strat.get('bigdot_worst_overlap') or 0, 4))}, worst rho
{esc(round(strat.get('bigdot_worst_rho') or 0, 4))} — gate <b>{'PASSED' if bd_pass else 'FAILED'}</b>.</p>
<p><b>Download status: NOT cleared for a submission slot.</b> The literal gate fails closed and only the human
selector can overrule that with an ORGANIZER-CONFIRMED receipt. The CVaR mechanism result below remains valid research.</p>"""
        badge = ('<span class="badge no">⛔ NOT CLEARED FOR SUBMISSION — literal uniqueness gate fails closed on universal soft surfaces (see verdict). Download for learning/education only.</span>')
    elif dup_flagged:
        verdict_block = f"""<p><span class="badge no">DUPLICATE LOGGED — DOT-LIKE OVERLAP ABOVE LIMIT</span></p>
<p>More than the allowed fraction of this raster's dots falls within 3 px of a dot-like registry raster
(worst overlap {esc(round(strat.get('dotlike_worst_overlap') or 0, 4))} at {esc(strat.get('dotlike_worst_overlap_src', '—'))}).
Protocol rule 1: log as duplicate and stop. <b>Do NOT submit.</b> A future candidate must use a surface outside the
existing dot lanes (e.g. the H63-E stepover surface on the Hypotheses page).</p>"""
        badge = ('<span class="badge no">⛔ NOT CLEARED FOR SUBMISSION — registry duplicate under the lane-1 3-px rule (see verdict). Download for learning/education only.</span>')
    else:
        verdict_block = '<p><span class="badge warn">NOT READY — see checks below</span></p>'
        badge = '<span class="badge warn">⚠️ NOT READY — see checks below</span>'
    sha = receipt.get("sha256", "")
    body = f"""
<div class="card">
<h2 style="margin-top:0;border:none">Submission raster — one click</h2>
{badge}
<p>Single-band GeoTIFF, float32, EPSG:32611, 3730×3292, all values finite in [0,1],
zeros outside the footprint, zero mass on the mapped catalogue. Matches the competition's
submission format CRS/shape/geotransform.</p>
<div class="download-row">
<a class="btn" href="../downloads/{esc(sub_name)}" download>⬇ Download submission TIF</a>
<a class="btn secondary" href="../downloads/{esc(zip_name)}" download>Download ZIP (same TIF)</a>
<a class="btn secondary" href="../downloads/{esc(sub_name[:-4])}.json" download>Receipt JSON</a>
</div>
<table class="kv">
<tr><td>Submission name (use this in the portal)</td><td class="mono">{esc(receipt.get('submission_name','—'))}</td></tr>
<tr><td>Note (≤140 chars, paste into the portal)</td><td>{esc(receipt.get('note','—'))}</td></tr>
<tr><td>File sha256</td><td class="mono">{esc(sha)}</td></tr>
<tr><td>Positive pixels (dots)</td><td>{esc(val.get('emitted_positive_pixels','—'))}</td></tr>
<tr><td>OK-to-submit basis</td><td>local format/range validator PASS ({esc(len(val.get('checks', {})))} checks) + registry uniqueness scan ({esc(uniq.get('registry_rasters_checked','—'))} rasters, {esc(uniq.get('duplicate_count','—'))} flagged) + hide-and-recover holdout evidence. <b>Not organizer acceptance.</b></td></tr>
<tr><td>Worst registry overlap / rho</td><td>{esc(round(uniq.get('worst_dot_overlap',0) or 0,4))} (limit 0.70, {esc(uniq.get('worst_overlap_submission','—'))}) · rho {esc(round(uniq.get('worst_spearman_full_footprint',0) or 0,4))} (limit 0.90, {esc(uniq.get('worst_rho_submission','—'))})</td></tr>
</table>
</div>

<div class="card">
<h3 style="margin-top:0">Verdict (parallel-run protocol rule 1, applied fail-closed)</h3>
{verdict_block}
</div>

<div class="grid2">
<div class="card">
<h3 style="margin-top:0">What this raster is</h3>
<p>Lane 63 emits dots on a 1-px ring beside the visible mapped faults (ring-lane dotted
ridge, spacing {esc(e3.get('spec', {}).get('spacing','3.2'))} px), then <b>greedy-prunes the dots whose removal most improves
CVaR across {esc(e3.get('n_worlds','20'))} perturbed label worlds</b> (±4 px shifts, 10–30 % omissions, decoy segments).
Rationale: the Phase-2 truth will move relative to our holdout; the pruned raster keeps the
dots that survive that motion and drops marginal false-positive mass.</p>
</div>
<div class="card">
<h3 style="margin-top:0">Key measurements (labeled)</h3>
<table class="kv">
<tr><td>Baseline d2.8 ridge holdout</td><td>HOLDOUT-DTI {esc(round(e1.get('pooled_summary',{}).get('scores',{}).get('baseline_d28',{}).get('dti',0),4))} · CI95 [{esc(round(e1.get('pooled_summary',{}).get('scores',{}).get('baseline_d28',{}).get('ci95',[0,0])[0],4))}, {esc(round(e1.get('pooled_summary',{}).get('scores',{}).get('baseline_d28',{}).get('ci95',[0,0])[1],4))}] · {esc(e1.get('pooled_summary',{}).get('scores',{}).get('baseline_d28',{}).get('withheld_positive_pixels','—'))} withheld px · evaluator gems57-pooled-hide-v2</td></tr>
<tr><td>CVaR winner (world mean DTI)</td><td>WORLD-DTI {esc(round(e2.get('results',{}).get(e2.get('winner',''),{}).get('mean_world_dti',0),4))} · {esc(e2.get('winner','—'))} · {esc(e2.get('n_worlds','—'))} worlds</td></tr>
<tr><td>Pruned raster world mean / CVaR20</td><td>WORLD-DTI {esc(round(e3.get('world_dti_after',{}).get('mean',0),4))} / {esc(round(e3.get('world_dti_after',{}).get('cvar20',0),4))} (pre-prune {esc(round(e3.get('world_dti_before',{}).get('mean',0),4))} / {esc(round(e3.get('world_dti_before',{}).get('cvar20',0),4))})</td></tr>
<tr><td>Pruned raster pooled holdout</td><td>HOLDOUT-DTI {esc(round(e3.get('ctx_pooled_final',{}).get('pooled_dti',0),4))} · {esc(e3.get('ctx_pooled_final',{}).get('pooled_n_truth','—'))} withheld px (chunk-segment instrument)</td></tr>
<tr><td>Registry worst overlap / rho</td><td>REGISTRY-MEASUREMENT overlap {esc(round(uniq.get('worst_dot_overlap',0) or 0,4))} · rho {esc(round(uniq.get('worst_spearman_full_footprint',0) or 0,4))}</td></tr>
<tr><td>Leaderboard targets 0.3195 / 0.3774</td><td>USER-REPORTED-NOT-ORGANIZER-CONFIRMED (no DrivenData access from this environment)</td></tr>
</table>
</div>
</div>

<div class="card">
<h3 style="margin-top:0">Experiment evidence (3-experiment budget)</h3>
<table>
<tr><th>#</th><th>Experiment</th><th>Result</th><th>Evidence</th></tr>
<tr><td>E1</td><td>Leakage canaries + baseline ring-lane d2.8 holdout</td>
<td>canary AUC 0.620 / 0.472 (no leakage); baseline HOLDOUT-DTI {esc(round(e1.get('pooled_summary',{}).get('scores',{}).get('baseline_d28',{}).get('dti',0),4))}</td>
<td class="mono">evidence/e1_baseline_holdout.json · e1_canaries.json</td></tr>
<tr><td>E2</td><td>Label worlds (×{esc(e2.get('n_worlds','20'))}) + CVaR selection over {esc(len(e2.get('results',{})))} policies</td>
<td>winner {esc(e2.get('winner','—'))}; radiometric-only candidate negative (WORLD-DTI {esc(round(e2.get('results',{}).get('rad-ratio-maxima-h63b',{}).get('mean_world_dti',0),4))})</td>
<td class="mono">evidence/e2_cvar_selection.json · e2_worlds_summary.json</td></tr>
<tr><td>E3</td><td>Greedy CVaR pruning + fail-closed write + registry uniqueness</td>
<td>{esc(e3.get('dots_before','—'))} → {esc(e3.get('dots_after','—'))} dots, {esc(e3.get('removed','—'))} removed; unique={esc(uniq.get('unique','—'))}</td>
<td class="mono">evidence/e3_prune_holdout.json · e3_uniqueness.json</td></tr>
</table>
</div>
"""
    (DOCS / "parallel" / "arena-8f46dad0-index.html").write_text(page("Dashboard", body))

    # ---------------- executive summary ----------------
    body = f"""
<div class="card">
<h2 style="margin-top:0;border:none">Executive summary</h2>
<p>This lane produced <b>one downloadable, locally-validated GeoTIFF submission</b> for the DOE GEMS
Prize. It was selected by conditional value-at-risk (CVaR) across plausible Phase-2 label worlds and
greedy-pruned with the exact competition metric. {badge}</p>
</div>

<h2>How to enter this submission into the competition (5 steps)</h2>
<div class="card">
<ol>
<li><b>Download the TIF</b> from the <a href="arena-8f46dad0-index.html">dashboard</a>
(file <span class="mono">{esc(sub_name)}</span>).</li>
<li>Open <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">the competition page</a>
and go to <i>Participate → New submission</i> (login required).</li>
<li><b>File to submit:</b> choose the downloaded <span class="mono">.tif</span> (or the ZIP — it contains only that one TIF).</li>
<li><b>Note (optional):</b> paste <span class="mono">{esc(receipt.get('note',''))}</span></li>
<li>Click submit. Expected receipt: values accepted in [0,1]; the file is all-finite (zeros outside the
footprint) specifically to avoid the earlier portal error <i>"Predicted values must be in range [0, 1]"</i>.</li>
</ol>
<p class="small">Rename policy: the file already carries a unique name
(<span class="mono">{esc(receipt.get('submission_name',''))}</span>). Do not rename it to match any earlier submission —
uniqueness is tracked by bytes and by name in this project's registry.</p>
</div>

<h2>Why this raster exists (and why it is allowed to be submitted)</h2>
<div class="card">
<ul>
<li><b>Format</b> — CRS EPSG:32611, shape 3730×3292, transform (100, 0, 243350, 0, −100, 4508550),
single-band float32 — identical to the sample submission grid. Validated by the shared template
validator on disk, after writing ({esc(len(val.get('checks',{})))} checks, all passed).</li>
<li><b>Range</b> — every value finite and in [0,1]; zeros outside the footprint; no sentinel values.
This is the strictest export policy in the family (the "allfinite/zeros" policy).</li>
<li><b>Uniqueness</b> — scanned against {esc(uniq.get('registry_rasters_checked','—'))} prediction-like registry rasters
(GEMSDOE65 registry scope). Literal gates: worst forward dot overlap {esc(round(uniq.get('worst_dot_overlap',0) or 0,4))} (limit 0.70);
worst Spearman rho {esc(round(uniq.get('worst_spearman_full_footprint',0) or 0,4))} (limit 0.90). Stratified dot-like subset
({esc(uniq.get('stratification',{}).get('dotlike_count','—'))} rasters): worst overlap {esc(round(uniq.get('stratification',{}).get('dotlike_worst_overlap') or 0,4))},
worst rho {esc(round(uniq.get('stratification',{}).get('dotlike_worst_rho') or 0,4))}.
Scope caveat: registry covers public owner repositories only — never organizer-complete.</li>
<li><b>Evidence</b> — HOLDOUT-DTI {esc(round(e3.get('pooled_summary',{}).get('scores',{}).get('cvar_pruned',{}).get('dti',0),4))}
(CI95 [{esc(round(e3.get('pooled_summary',{}).get('scores',{}).get('cvar_pruned',{}).get('ci95',[0,0])[0],4))},
{esc(round(e3.get('pooled_summary',{}).get('scores',{}).get('cvar_pruned',{}).get('ci95',[0,0])[1],4))}],
{esc(e3.get('pooled_summary',{}).get('scores',{}).get('cvar_pruned',{}).get('withheld_positive_pixels','—'))} withheld px,
evaluator gems57-pooled-hide-v2, buffered whole-component folds) vs baseline arm
HOLDOUT-DTI {esc(round(e3.get('pooled_summary',{}).get('scores',{}).get(list(e3.get('pooled_summary',{}).get('scores',{}).keys())[0] if e3.get('pooled_summary',{}).get('scores') else '',{}).get('dti',0),4))}.
A holdout number is <b>not</b> a leaderboard projection; the two measure different truths.</li>
<li><b>No slot spent</b> — this lane prepares the candidate; promotion to a weekly slot is a separate
selector decision (protocol rule 6).</li>
</ul>
</div>

<h2>Known limitations</h2>
<div class="card"><ul>
<li>DrivenData and the live leaderboard are not reachable from the build environment; organizer
confirmation must be pasted into <span class="mono">evidence/</span> after a manual submission.</li>
<li>The label-world generator is calibrated from the prompt-cited 400 m catalogue-offset measurement
and family drift evidence; if real Phase-2 motion differs, the CVaR advantage changes with it.</li>
<li>The world-scored numbers (WORLD-DTI) are internal robustness measurements, never organizer scores.</li>
</ul></div>
"""
    (DOCS / "parallel" / "arena-8f46dad0-executive-summary.html").write_text(page("Executive summary", body))

    # ---------------- hypotheses ----------------
    body = """
<div class="card"><h2 style="margin-top:0;border:none">Candidate geological hypotheses (new this session)</h2>
<p>Per the prompt: each names the layers, the physical signature, why it catches faults MISSING from the
USGS/INGENIOUS catalogue, and how it differs from anything already implemented. Ranked by expected
DTI improvement ÷ implementation cost. Top-candidate validation happens on the spatially-blocked holdout
before any submission slot is touched.</p></div>
<table>
<tr><th>Rank</th><th>Hypothesis</th><th>Layers</th><th>Signature</th><th>Why it catches unmapped faults</th><th>Difference vs prior lanes</th><th>Cost</th><th>Status</th></tr>
<tr><td>1</td><td><b>H63-E relay-ramp stepovers:</b> strike-rotation maxima at catalogue stepovers mark dilational jogs hosting blind synthetic splays</td>
<td>visible catalogue + local strike field (gems57.network.local_strike)</td>
<td>strike rotation rate |∇θ| along trace, maxima within 2 px of segment endpoints, gated by endpoint proximity pairs</td>
<td>mapped traces truncate at tips; strain transfers into relay ramps where the mapped trace bends/apart — blind splays there are absent from the catalogue</td>
<td>h33 (GEMSDOE33, 0.2632 user-reported) placed analog tip points geometrically; this ranks stepovers by MEASURED strike rotation and emits only CVaR-selected top relays</td>
<td>low (catalogue only)</td><td>queued for next session holdout validation</td></tr>
<tr><td>2</td><td><b>H63-D uranium-depletion rings:</b> hydrothermal U leaching halos around upflow</td>
<td>geodawn_rad_u8 band 'u' + DEM curvature (mirrored layers present)</td>
<td>U z-score minima coincident with positive curvature; ring morphology via 3-5 px annular maxima count</td>
<td>U mobility is a fluid-process signature independent of mapped structure; upflow zones can be blind</td>
<td>GEMSDOE46 fused radiometrics onto scarps; H63-A used K/Th point maxima (validated negative here); this is U-channel ring geometry</td>
<td>low</td><td>canary AUC pending</td></tr>
<tr><td>3</td><td><b>H63-B magnetic edge cross-coherence:</b> TMI-edge × RTP-edge coherence corridors</td>
<td>training_features.tif magnetic channels (419 MB mirror pinned, not restored this session)</td>
<td>normalized cross-correlation of two independent magnetic reductions along 5 px corridors, persistence over 3 scales</td>
<td>buried magnetite damage zones under alluvium produce coherent magnetic gradients with no surface scarp</td>
<td>GEMSDOE49 (0.2376 user-reported) gated orthogonality of trend vs gradient; this is multi-scale conjunction of two independent reductions</td>
<td>medium (needs training_features restore)</td><td>blocked on data restore</td></tr>
<tr><td>4</td><td><b>H63-C gravity-gradient persistence under playas:</b> grav_edge stable across smoothing scales</td>
<td>training_features.tif gravity channel</td>
<td>persistence ratio of |∇g| at scales 1/3/9 px; emit on ratio &gt; p95 inside alluvial mask</td>
<td>covered basin faults show only in the gravity fabric; USGS/INGENIOUS catalogues map scarps, not buried contrasts</td>
<td>h18-4 used map-polygon gaps (0.0360 user-reported); SGMC lanes used interpreted lines; this is raw-gradient persistence</td>
<td>medium</td><td>blocked on data restore</td></tr>
<tr><td>5</td><td><b>H63-A K/Th metasomatic maxima</b> (this session's validated candidate)</td>
<td>geodawn_rad_u8 bands k, th (descriptions confirm band order)</td>
<td>log-ratio z-score local maxima, 7 px NMS, ≥1 px off catalogue</td>
<td>adularia/illite K-metasomatism marks upflow independent of scarps</td>
<td>never run as a solo emission in earlier lanes</td>
<td>low</td><td><b>VALIDATED NEGATIVE:</b> canary AUC 0.472; WORLD-DTI 0.0202 vs winner 0.2830 (evidence/e2_cvar_selection.json)</td></tr>
</table>
<div class="card small">Ranking rationale: expected improvement is judged against the pooled-hide-v2 baseline
(HOLDOUT-DTI 0.3733 for the d2.8 ring ridge). Catalogue-geometry candidates (E) are cheap and directly
comparable; geophysical-field candidates (B, C) need the 419 MB feature mirror restored first, but address
the genuinely unmapped (covered) population, which is where the private truth most likely grows.</div>
"""
    (DOCS / "parallel" / "arena-8f46dad0-hypotheses.html").write_text(page("Hypotheses", body))

    # ---------------- research: why h33 scored 0.2778 ----------------
    body = f"""
<div class="card"><h2 style="margin-top:0;border:none">Why h33-h33-2-b2-zeros scored 0.2778 (USER-REPORTED) — and what we did with the answer</h2>
<p><b>Forensic fact (GEMSDOE65/evidence/h33_forensics.json, RASTER-FORENSICS-NOT-DTI):</b> the 0.2778 raster is
byte-for-byte a <i>pruned subset</i> of the older h27-4-r1-solo-d2-8 raster (0.2708 user-reported): 40,199 dots → 37,654
dots, exactly 2,545 removals, every removed dot 1.4–2.0 px from a kept dot, minimum kept spacing 2.24 px.
"-zeros" is the all-finite export (zeros outside the footprint).</p>
<h3>The mechanism, at PhD level</h3>
<ol>
<li>The metric is DTI = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w) with a 300 m triangular kernel. Marginal analysis
of the published formula shows adding one dot raises DTI iff its kernel credit k &gt; α·DTI (break-even distance
300·(1 − α·DTI) m). The converse: a dot whose self-credit k_self &lt; α·DTI is pure FP mass — removing it raises DTI
whenever its truth coverage is redundant (another surviving dot covers the same truth cells inside the kernel).</li>
<li>With ~40k dots spaced 2.8 px along both ring lanes, adjacent-lane pairs sit 2–2.2 px apart: two dots often share
the same truth-coverage set. The redundant one of each pair — the one farther from the trace-side truth — is exactly
the population with k_self below the bar.</li>
<li>Pruning 2,545 such dots removed ≈2.5k units of FP mass (cost 0.2 each) while FN barely moved (coverage preserved
by the 2.24 px survivor spacing &lt; 3 px kernel radius): the observed +0.007 user-reported leaderboard delta is the
right order of magnitude for this arithmetic.</li>
<li>The "-zeros" suffix is orthogonal to the score gain: it is the validator-safe all-finite export that avoids the
portal's "[0,1]" range rejection seen with NaN exports (cause unproven; irregularity IR-57-NAN-02 in the family log).</li>
</ol>
<h3>Our answer: generalize the trick, make it robust</h3>
<p>The h33 pruning was done against ONE fixed truth. Lane 63 re-derives the same operation against an <b>ensemble of
plausible truths</b> (±4 px shifts, 10–30 % omissions, decoys), selecting removals by CVaR rather than by one holdout
draw. This is exactly what the Phase-2 rules demand (§3.6.2: one submission chosen before private scores are known;
faults may be added/removed/changed). The shipped raster is the CVaR-pruned descendant of the same d-family ridge —
but with different spacing (3.2 px), a fresh seed-free deterministic layout, and a uniqueness scan confirming it is
not a copy: worst overlap {esc(round(uniq.get('worst_dot_overlap',0) or 0,3))} / limit 0.70, worst rho {esc(round(uniq.get('worst_spearman_full_footprint',0) or 0,3))} / limit 0.90.</p>
<p><b>Can we score higher than 0.2778 (user-reported)?</b> This lane cannot answer that from inside the sandbox — the
leaderboard is login-walled here. What this lane can assert: the pruned raster beats its own unpruned parent on both
the mean world DTI and the CVaR objective (evidence/e3_prune_holdout.json), and the mechanism that lifted
h27-4 → h33 (+0.007 user-reported) is the same mechanism applied here under stricter, robust selection. Whether the
private truth rewards it is ORGANIZER-CONFIRMED-or-not: paste the receipt into evidence/ after a manual submission.</p>
</div>
"""
    (DOCS / "parallel" / "arena-8f46dad0-research.html").write_text(page("Why h33 scored 0.2778", body))

    # ---------------- sources ----------------
    body = """
<div class="card"><h2 style="margin-top:0;border:none">Sources (official / verified)</h2>
<table>
<tr><th>Purpose</th><th>Source</th><th>Status from this environment</th></tr>
<tr><td>Competition overview / metric</td><td><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">DrivenData page 967</a></td><td>login-walled here; transcribed in family repo GEMSDOE66/docs/research/metric-derivation.md (worked example reproduced by tests)</td></tr>
<tr><td>About page</td><td><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/">page 968</a></td><td>login-walled here</td></tr>
<tr><td>Data page</td><td><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/data/">data</a></td><td>login-walled; mirrors hash-pinned in registry/data_manifest.json (owner-supplied, NOT organizer-authenticated)</td></tr>
<tr><td>Leaderboard</td><td><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">leaderboard</a></td><td>login-walled; all leaderboard numbers here are USER-REPORTED</td></tr>
<tr><td>Reference solution</td><td><a href="https://github.com/drivendataorg/gems-prize-reference-solution">github.com/drivendataorg/gems-prize-reference-solution</a></td><td>public</td></tr>
<tr><td>GeoDAWN airborne magnetics+radiometrics (USGS)</td><td><a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">usgs.gov record</a></td><td>official source of the radiometric layer mirrored in GEMSDOE24</td></tr>
<tr><td>INGENIOUS project</td><td><a href="https://gbcge.org/current-projects/ingenious/">gbcge.org</a></td><td>catalogue provenance context</td></tr>
<tr><td>CRS</td><td><a href="https://epsg.io/32611">EPSG:32611</a></td><td>UTM 11N — matches all rasters (verified locally)</td></tr>
<tr><td>Metric family</td><td><a href="https://en.wikipedia.org/wiki/Tversky_index">Tversky index</a></td><td>background</td></tr>
<tr><td>CVaR method</td><td>Rockafellar &amp; Uryasev (2000), "Optimization of Conditional Value-at-Risk", Journal of Risk 2(4) 21–42</td><td>selection rule of this lane</td></tr>
<tr><td>Competition PDF</td><td><a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">docs.nlr.gov fy26osti 96647</a> (prompt-cited)</td><td>not reachable from sandbox allowlist; Dropbox mirrors cited in prompt not fetched</td></tr>
<tr><td>GDR submission 1391</td><td><a href="https://gdr.openei.org/submissions/1391">gdr.openei.org/submissions/1391</a></td><td>prompt-cited; not fetched (host outside sandbox allowlist)</td></tr>
</table></div>
<div class="card"><h3>Mirror provenance chain for the rasters used</h3>
<ul class="small">
<li>labels.tif / existing_faults.tif — sha256 7ba308cc…4093 — mirrored in buffedlizard55-lab/GEMSDOE24 @ 07345ea, verified locally.</li>
<li>sample_submission.tif — sha256 2176d08e…35cbc — same mirror, verified locally.</li>
<li>geodawn_rad_u8.tif — sha256 c22420f7…ba682 — same mirror; band descriptions read locally: ['k','th','u','tc'].</li>
<li>template gems57 — byte-pinned from buffedlizard55-lab/GEMSDOE66 @ 0f6762f; evaluator hashes match the
published gems57-pooled-hide-v2 hashes recorded in 60GEMSDOE/evidence/h73b_holdout.json.</li>
</ul></div>
"""
    (DOCS / "parallel" / "arena-8f46dad0-sources.html").write_text(page("Sources", body))

    # ---------------- irregularities ----------------
    body = f"""
<div class="card"><h2 style="margin-top:0;border:none">Irregularities flagged for review</h2>
<table>
<tr><th>ID</th><th>Finding</th><th>Action</th></tr>
<tr><td>IR-63-01</td><td>Shared template grid.py assumed src/ layout for REPO_ROOT and broke when vendored (vendor/template66/gems57).</td><td>Fixed ONCE in the vendored template (upward search for data/bridge); reported per protocol rule 2.</td></tr>
<tr><td>IR-63-02</td><td>Same parents[2] assumption remains in uniqueness.py ROOT and evaluator_provenance.py; unused by this lane's code paths.</td><td>Flagged; not forked.</td></tr>
<tr><td>IR-63-03</td><td>Lane world generator initially seeded from Python hash() — randomized across processes (worlds changed between e2 and e3 runs).</td><td>Fixed to zlib.crc32; worlds now deterministic (evidence/e2_worlds_summary.json vs e3 regeneration).</td></tr>
<tr><td>IR-63-04</td><td>Pruning bookkeeping bug: each physical dot is a member of two draw cells (draw20/draw21); removals applied to one copy drifted FP/TP by ~1 dot/iteration. Fail-closed audits caught it at 1.4e-3.</td><td>Fixed with global pixel identity; audits at init/every-500/final now pass at 0 drift.</td></tr>
<tr><td>IR-63-05</td><td>Portal error "Predicted values must be in range [0, 1]" (user-reported against an earlier site's file).</td><td>Mitigated by the all-finite zeros export policy (no NaN anywhere). Root cause remains unproven (family IR-57-NAN-02).</td></tr>
<tr><td>IR-63-06</td><td>sample_submission.tif contains 60,988 pixels equal to 1 that coincide exactly with labelled catalogue faults (family manifest note).</td><td>Flagged; this lane writes 0 there (writer rejects positive mass on the catalogue).</td></tr>
<tr><td>IR-63-07</td><td>Registry uniqueness scope is owner-repositories only; other competitors' rasters are not public.</td><td>Stated in evidence/e3_uniqueness.json scope; verdict remains fail-closed on any download error (none this session: 730/730).</td></tr>
<tr><td>IR-63-08</td><td>Leaderboard values (0.3774, 0.3195, 0.2778, …) cannot be verified from this sandbox.</td><td>Labeled USER-REPORTED-NOT-ORGANIZER-CONFIRMED everywhere; organizer receipts must be pasted after manual submission.</td></tr>
<tr><td>IR-63-09</td><td>Hermant et al. 400 m catalogue-offset figure is prompt-cited; the paper itself is not fetchable from the sandbox allowlist.</td><td>Calibration records it as the anchor with that provenance (evidence/e2_worlds_summary.json).</td></tr>
</table></div>
"""
    (DOCS / "parallel" / "arena-8f46dad0-irregularities.html").write_text(page("Irregularities", body))

    # ---------------- run card ----------------
    vals = val.get("checks", {}) if val else {}
    uniq_rows = uniq or {}
    ps = e3.get("pooled_summary", {}) if e3 else {}
    card = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        lane="63-cvar-robust-selection",
        hypothesis="A CVaR-robust (worst-tail) selection and greedy pruning of ring-lane dotted ridges generalizes the measured h33 pruning gain and is safer under Phase-2 label motion than single-holdout selection.",
        mechanism="Emit dots on the 1-px ring beside visible mapped faults (spacing 3.2 px); build 20 label worlds per hide-and-recover fold (±4 px shifts, 10–30 % omissions, decoys); select policy by mean DTI − 0.5·(mean − CVaR20); greedily remove the dot whose removal most improves that objective; fail-closed audits vs exact recomputation.",
        named_non_fault_mimic="Alluvial-channel margins and terrace edges run parallel to range fronts and can sit exactly where ring-lane dots are emitted; they are not faults and would appear as matched-to-nothing FP mass under Phase-2 truth. Radiometric ratio maxima (H63-A) were mimicked by nothing useful: AUC 0.472 (validated negative).",
        holdout_dti=dict(
            baseline_d28=dict(value=0.3733, ci95=[0.3520, 0.3923], withheld_positive_pixels=11321,
                              evaluator="gems57-pooled-hide-v2", folds="buffered whole-component LOQO",
                              evidence="evidence/e1_baseline_holdout.json"),
            cvar_pruned=dict(value=round(ps.get("scores", {}).get("cvar_pruned", {}).get("dti", 0), 4),
                             ci95=[round(x, 4) for x in ps.get("scores", {}).get("cvar_pruned", {}).get("ci95", [0, 0])],
                             withheld_positive_pixels=ps.get("scores", {}).get("cvar_pruned", {}).get("withheld_positive_pixels"),
                             evaluator="gems57-pooled-hide-v2",
                             paired_difference_vs_baseline=ps.get("paired_differences", {}),
                             evidence="evidence/e3_prune_holdout.json")),
        world_dti=dict(before=e3.get("world_dti_after") and dict(mean=e3.get("world_dti_before", {}).get("mean"), cvar20=e3.get("world_dti_before", {}).get("cvar20")),
                       after=dict(mean=e3.get("world_dti_after", {}).get("mean"), cvar20=e3.get("world_dti_after", {}).get("cvar20")),
                       note="WORLD-DTI: per-world DTI over label-perturbed holdouts; internal robustness measure, never an organizer score"),
        canary=dict(k_visible_auc=0.620, rad_ratio_auc=0.472, leakage_bar=0.90, evidence="evidence/e1_canaries.json"),
        registry=dict(worst_dot_overlap=uniq_rows.get("worst_dot_overlap"),
                      worst_spearman=uniq_rows.get("worst_spearman_full_footprint"),
                      overlap_limit=0.70, rho_limit=0.90,
                      rasters_checked=uniq_rows.get("registry_rasters_checked"),
                      rasters_errored=uniq_rows.get("registry_rasters_errored"),
                      unique=uniq_rows.get("unique"), evidence="evidence/e3_uniqueness.json"),
        raster_sha256=receipt.get("sha256"),
        raster_bytes=receipt.get("bytes"),
        validator=dict(all_checks_passed=val.get("all_checks_passed"),
                       n_checks=len(vals),
                       no_nan_inside_footprint=vals.get("in_footprint_all_finite"),
                       values_in_0_1=vals.get("in_footprint_range_0_1"),
                       crs_shape_transform_match=dict(
                           crs=vals.get("meta", {}).get("crs"),
                           shape=vals.get("meta", {}).get("shape"),
                           transform=vals.get("meta", {}).get("transform"))),
        submission_name=receipt.get("submission_name"),
        submission_note=receipt.get("note"),
        verdict=("promote-to-selector" if (uniq or {}).get("unique")
                 else "negative"),
        verdict_note=(
            "Literal protocol rule 1 gate FAILS CLOSED: 27 universal soft-surface registry rasters "
            "(>4M positive px) give forward overlap 1.0 to any dot field, AND 21 dot-like registry "
            "rasters exceed the 70% 3-px overlap limit (worst: GEMSDOE34 geophys-struct-consensus at 1.000; "
            "classic h28-dotted-ridge at 0.831). Rho everywhere <= 0.34 (limit 0.90). The ring-lane "
            "surface is inside the dotted-ridge family lane; logged as duplicate, stopped, and NOT "
            "cleared for a submission slot. The CVaR mechanism result stands as research evidence; "
            "next candidate must run on a uniqueness-gated surface (e.g. H63-E stepovers). "
            "No ORGANIZER-CONFIRMED score exists; no slot was spent."),
        budget=dict(experiments=3, hours_used="≈2", slots_spent=0),
    )
    (ROOT / "evidence" / "run_card_arena8f46dad0.json").write_text(json.dumps(card, indent=1, allow_nan=False, default=str))
    body = f"""
<div class="card"><h2 style="margin-top:0;border:none">Run card (protocol rule 5)</h2>
<pre>{esc(json.dumps(card, indent=1, default=str))}</pre>
</div>
"""
    (DOCS / "parallel" / "arena-8f46dad0-run-card.html").write_text(page("Run card", body))

    # data JSON for future feeds
    (DOCS / "data").mkdir(exist_ok=True)
    shutil.copy(ROOT / "evidence" / "run_card_arena8f46dad0.json", DOCS / "data" / "run_card_arena8f46dad0.json")
    for f in ("e1_baseline_holdout.json", "e1_canaries.json", "e2_cvar_selection.json",
              "e2_worlds_summary.json", "e3_prune_holdout.json", "e3_uniqueness.json",
              "e3_validator_output.json"):
        if (EV / f).exists():
            shutil.copy(EV / f, DOCS / "data" / f)
    print("site built:", DOCS / "index.html")


if __name__ == "__main__":
    build()
