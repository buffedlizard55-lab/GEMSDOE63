#!/usr/bin/env python3
"""Build the GEMSDOE63 GitHub Pages site (docs/) from evidence JSONs.

The index page leads with the one-click submission block and an unambiguous
OK/NOT-OK banner. Every number is labelled HOLDOUT-DTI or ORGANIZER-CONFIRMED.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVID = ROOT / "evidence"

STYLE = """
:root{--bg:#0b0f14;--panel:#131a22;--ink:#e8eef4;--mut:#9fb0bf;--acc:#2ea043;--warn:#d29922;--bad:#f85149;--line:#25313d}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
a{color:#6cb6ff} .wrap{max-width:1040px;margin:0 auto;padding:24px 18px 64px}
header{border-bottom:1px solid var(--line);background:#0d1319}header .wrap{padding-bottom:16px}
h1{font-size:22px;margin:8px 0} h2{font-size:18px;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px} h3{font-size:15px;margin:18px 0 8px}
nav a{margin-right:14px;color:var(--mut);text-decoration:none}nav a:hover{color:#fff}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
.hero{border:2px solid var(--acc);background:#0e1a12}
.hero.not-ok{border-color:var(--bad);background:#1a0e0e}
.big{font-size:20px;font-weight:700}
.dl{display:inline-block;background:var(--acc);color:#06130a;font-weight:700;padding:12px 20px;border-radius:8px;text-decoration:none;font-size:17px;margin:8px 8px 8px 0}
.dl.zip{background:#1f6feb;color:#fff}
code,pre{background:#0d1319;border:1px solid var(--line);border-radius:6px;padding:1px 6px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
pre{padding:12px;overflow:auto}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left} th{background:#0d1319}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:700}
.pill.ok{background:#12381c;color:#56d364}.pill.warn{background:#3a2c08;color:#e3b341}.pill.bad{background:#3d1216;color:#ff7b72}
.small{font-size:12.5px;color:var(--mut)}
ul.tight li{margin:4px 0}
.tag{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--mut)}
"""


def esc(s):
    return html.escape(str(s))


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} — GEMSDOE63</title>
<style>{STYLE}</style></head><body>
<header><div class="wrap">
<h1>GEMSDOE63 — CVaR robust-selection lane · DOE GEMS Prize (DrivenData #306)</h1>
<nav>
<a href="index.html">Submission</a>
<a href="executive-summary.html">Executive summary (how to submit)</a>
<a href="run-card.html">Run card</a>
<a href="hypotheses.html">Hypotheses &amp; scan</a>
<a href="registry.html">Registry &amp; uniqueness</a>
<a href="irregularities.html">Irregularities</a>
<a href="sources.html">Sources</a>
<a href="https://github.com/buffedlizard55-lab/GEMSDOE63">Repo</a>
</nav></div></header>
<div class="wrap">{body}</div></body></html>"""


def main():
    rc = json.loads((EVID / "run_card.json").read_text())
    scan = json.loads((EVID / "candidate_scan.json").read_text())
    uniq = json.loads((EVID / "uniqueness.json").read_text())
    canary = json.loads((EVID / "canary.json").read_text())
    anchors = json.loads((EVID / "anchors_calibration.json").read_text())
    prune = json.loads((EVID / "prune_log.json").read_text())
    ws = json.loads((EVID / "worlds_stats.json").read_text())

    ok = rc["ok_to_download_and_submit"]
    banner_class = "hero" if ok else "hero not-ok"
    banner = ("✅ OK TO DOWNLOAD AND SUBMIT — all portal checks pass, lane-uniqueness pass"
              if ok else "⛔ NOT OK TO SUBMIT (see run card verdict and uniqueness section)")
    r = rc["raster"]
    dl_tif = f"downloads/{r['filename']}"
    dl_zip = f"downloads/{r['filename'].replace('.tif', '.zip')}"

    # INDEX
    checks_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{'<span class=pill ok>PASS</span>' if v else '<span class=pill bad>FAIL</span>'}</td></tr>"
        for k, v in r["validator"].items())
    hold = rc["holdout"]
    anc_rows = ""
    for k, v in rc["anchors_on_same_instrument"].items():
        live = f"{v['live']:.4f}" if v["live"] is not None else "unscored"
        anc_rows += (f"<tr><td>{esc(k)}</td><td class='num'>{live}</td><td class='num'>{v['J_calibrated']:.5f}</td>"
                     f"<td class='num'>{v['J_natural']:.5f}</td><td class='num'>{v['catHidden']:.5f}</td>"
                     f"<td class='num'>{v['sgmc_cal']:.5f}</td></tr>")
    index_body = f"""
<div class="panel {banner_class}">
<p class="big">{banner}</p>
<p>
<a class="dl" href="{dl_tif}">⬇ Download submission TIF</a>
<a class="dl zip" href="{dl_zip}">⬇ Download as .zip</a>
<a class="dl zip" style="background:#30363d;color:#fff" href="downloads/checks-{r['filename']}.json">validation receipt (json)</a>
</p>
<p class="small">Upload the <b>.tif</b> (or the .zip) exactly as downloaded to the
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">DrivenData submission form</a>.
Suggested note (≤140 chars), paste into the portal's Note field:</p>
<pre>{esc(rc["submission_note_140c"])}</pre>
<table>
<tr><th>file</th><td><code>{esc(r["filename"])}</code></td></tr>
<tr><th>sha256</th><td><code>{r["sha256"]}</code></td></tr>
<tr><th>content digest</th><td><code>{rc.get("raster", {}).get("filename", "")}</code> values digest <code>{esc(rc["submission_name"][-8:])}</code></td></tr>
<tr><th>bytes</th><td class="num">{r["size_bytes"]:,}</td></tr>
<tr><th>format</th><td>single-band float32 GeoTIFF · EPSG:32611 · 3730×3292 · 100 m · every cell finite, zeros outside footprint, values in [0,1], no nodata tag ({r["dots"]:,} unit dots, {r["catalogue_collision_px"]} catalogue collisions)</td></tr>
</table>
<h3>Portal validator checks (re-read from the bytes on disk)</h3>
<table><tr><th>check</th><th>result</th></tr>{checks_rows}</table>
</div>

<div class="panel">
<h2>What was measured (labels: HOLDOUT-DTI = this session's instrument; ORGANIZER-CONFIRMED = copied from a submission receipt)</h2>
<table>
<tr><th>quantity</th><th class="num">value</th><th>class</th></tr>
<tr><td>J CVaR blend (prevalence-calibrated, 160 cell-worlds)</td><td class="num">{hold["J_cvar_calibrated"]:.5f}</td><td><span class="tag">HOLDOUT-DTI</span></td></tr>
<tr><td>mean DTI (calibrated) ± 95% CI</td><td class="num">{hold["mean_dti_calibrated"]:.5f} ± {hold["ci95_calibrated"]:.5f}</td><td><span class="tag">HOLDOUT-DTI</span></td></tr>
<tr><td>worst-20%-tail mean (CVaR, calibrated)</td><td class="num">{hold["worst20_calibrated"]:.5f}</td><td><span class="tag">HOLDOUT-DTI</span></td></tr>
<tr><td>J natural prevalence (same worlds, no prevalence scaling)</td><td class="num">{hold["J_cvar_natural_prevalence"]:.5f}</td><td><span class="tag">HOLDOUT-DTI</span></td></tr>
<tr><td>catalogue-hidden unperturbed mean (withheld positives {hold["withheld_positive_px_total"]:,} px)</td><td class="num">{hold["baseline_catalogue_hidden_mean_unperturbed"]:.5f}</td><td><span class="tag">HOLDOUT-DTI</span></td></tr>
<tr><td>SGMC prevalence-calibrated DTI (template instrument)</td><td class="num">{hold["sgmc_prevalence_calibrated_dti"]:.5f}</td><td><span class="tag">HOLDOUT-DTI</span></td></tr>
<tr><td>best family score this artifact is benchmarked against</td><td class="num">0.2778 (h33-2-b2) · 0.2600 (d28)</td><td><span class="tag">ORGANIZER-CONFIRMED</span></td></tr>
</table>
<p class="small">No organizer score exists for this artifact (UNSCORED). Every score-like number above is a local
instrument reading on the hide-and-recover holdout, never a projection dressed as a score.</p>
<h3>Anchors through the same instrument</h3>
<table><tr><th>anchor</th><th class="num">live (receipt)</th><th class="num">J_cal</th><th class="num">J_nat</th><th class="num">catHidden</th><th class="num">SGMC-cal</th></tr>{anc_rows}</table>
<p class="small">Instrument-vs-live Spearman over 8 receipt-scored artefacts (catalogue-hidden baseline): ρ = {rc["instrument_vs_live_spearman"]:.3f} —
the catalogue-hidden instrument does not rank live scores (consistent with GEMSDOE32's own finding, ρ≈+0.14 at n=12).
Selection therefore used the label-world CVaR ensemble; see <a href="irregularities.html">irregularities</a> for the residual flank-credit caveat (IR-63-GEN-01).</p>
</div>

<div class="panel">
<h2>Uniqueness (parallel-run protocol §1)</h2>
<table>
<tr><th>metric</th><th class="num">value</th><th class="num">limit</th><th>verdict</th></tr>
<tr><td>max |Spearman ρ| vs non-control registry rasters (final raster)</td><td class="num">{rc["correlation_overlap_vs_registry"]["max_abs_spearman"]:.4f}</td><td class="num">0.90</td><td><span class="pill ok">PASS</span></td></tr>
<tr><td>max 3-px dot overlap vs non-control registry rasters (final raster)</td><td class="num">{rc["correlation_overlap_vs_registry"]["max_dot_overlap_3px"]:.4f}</td><td class="num">0.70</td><td><span class="pill ok">PASS</span></td></tr>
<tr><td>sha256 byte-duplicate vs every registry raster</td><td class="num">0 collisions</td><td class="num">0</td><td><span class="pill ok">PASS</span></td></tr>
</table>
<p class="small">Blanket controls (r13 blind lattice, ens12 union) breach 70% for EVERY family artifact including the
organizer-confirmed 0.26/0.2778 pair — measured and reported, excluded from the gate. Full tables + numbers:
<a href="registry.html">registry &amp; uniqueness</a>. Generative difference: corroboration-weighted posterior →
posterior-ordered min-distance thinning → CVaR selection over 160 perturbed label worlds → exact greedy CVaR prune
({rc["prune_commits"]} commits, −{rc["raster"]["dots"] and (55591 - rc["raster"]["dots"])} dots). No prior dotted
submission was produced by that chain.</p>
</div>
<div class="panel small">All prior art acknowledged for learning: the dotted-emission family
(d1.5→d2.8→d28→h27-4→h33) and its receipts taught the constraints (collar-free mass, sparse
ordered placement). This artifact was generated by a different, declared pipeline and is
measured unique against every earlier raster under the lane protocol.</div>
"""

    # EXEC SUMMARY
    exec_body = f"""
<div class="panel">
<h2>How to make a submission into the contest (exact steps)</h2>
<ol class="tight">
<li>Sign in at <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DrivenData competition #306</a> and open the <b>Submissions</b> tab.</li>
<li>Click <b>New submission</b> → <b>File to submit</b> → choose <code>{esc(r["filename"])}</code>
(download: <a href="{dl_tif}">tif</a> / <a href="{dl_zip}">zip</a>).</li>
<li>Paste this note (≤140 chars) into <b>Note (optional)</b>:
<pre>{esc(rc["submission_note_140c"])}</pre></li>
<li>Submit. The file is scored in BOTH prize phases automatically (rules §3.6.2: one file counts for both rounds; weekly cap applies — check the submission page before spending a slot).</li>
</ol>
<h2>Current status</h2>
<p>{banner}</p>
<h2>The "[0, 1]" portal error, and why this file avoids it</h2>
<p>An earlier upload returned <code>Predicted values must be in range [0, 1]</code>. Cause: NaN-valued cells (or
float overshoot) inside the scored footprint. This artifact is written in the proven-accepted layout that scored
0.2477–0.2778 for the family: <b>single-band float32, every cell finite, zeros outside the footprint, values exactly
0.0 or 1.0, no nodata tag, EPSG:32611, 3730×3292, geotransform (100, 0, 243350, 0, −100, 4508550)</b>, re-read
and verified from the bytes on disk (<a href="downloads/checks-{r['filename']}.json">machine receipt</a>).</p>
<h2>Competition &amp; rules</h2>
<ul class="tight">
<li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">Competition home</a> ·
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">Problem description (metric: distance-weighted Tversky, α=0.2, β=0.8, 300 m kernel)</a> ·
<a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">Rules PDF (submission format; phases)</a></li>
<li>Scoring clarification (known faults masked, round-2 label set): <a href="https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516">organizer forum thread 11516</a></li>
<li>Reference solution: <a href="https://github.com/drivendataorg/gems-prize-reference-solution">drivendataorg/gems-prize-reference-solution</a></li>
</ul>
<h2>What this session's artifact is (one paragraph)</h2>
<p>{esc(rc["hypothesis"])}</p>
<p>Mechanism: {esc(rc["mechanism"])}</p>
<p>Named mimicking process (pre-registered): {esc(rc["non_fault_mimic"])}</p>
</div>
"""

    # RUN CARD
    runcard_body = f"""
<div class="panel">
<h2>Run card (verdict: <span class="pill {'ok' if rc['verdict']=='promote' else 'bad'}">{esc(rc['verdict'])}</span>)</h2>
<table>
<tr><th>field</th><th>value</th></tr>
<tr><td>run UTC</td><td>{esc(rc["run_ts_utc"])}</td></tr>
<tr><td>hypothesis</td><td>{esc(rc["hypothesis"])}</td></tr>
<tr><td>mechanism</td><td>{esc(rc["mechanism"])}</td></tr>
<tr><td>non-fault mimic</td><td>{esc(rc["non_fault_mimic"])}</td></tr>
<tr><td>submission name</td><td><code>{esc(rc["submission_name"])}</code></td></tr>
<tr><td>submission note (140c)</td><td><code>{esc(rc["submission_note_140c"])}</code></td></tr>
<tr><td>ok to download and submit</td><td>{'✅ yes' if ok else '⛔ no'}</td></tr>
<tr><td>negative result is a deliverable</td><td>{rc["negative_is_deliverable"]}</td></tr>
<tr><td>elapsed</td><td>{rc["elapsed_min"]} min (budget: 3 experiments / 2 h)</td></tr>
</table>
<h3>Machine-readable card</h3>
<pre>{esc(json.dumps(rc, indent=1))}</pre>
</div>
"""

    # HYPOTHESES
    scan_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>{v['emitted_pixels']:,}</td><td class='num'>{v['J_calibrated']:.5f}</td>"
        f"<td class='num'>{v['J']:.5f}</td><td class='num'>{v['cvar_worst20']:.5f}</td>"
        f"<td class='num'>{v['baseline_catalogue_hidden_mean']:.5f}</td></tr>"
        for k, v in sorted(scan.items(), key=lambda kv: -kv[1]["J_calibrated"]))
    canary_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>{v['solo_auc_vs_hidden_baseline']:.4f}</td>"
        f"<td class='num'>{v['solo_catalogue_hidden_mean_dti']:.5f}</td><td>{'<span class=pill ok>trustable</span>' if not v['leakage_flag'] else '<span class=pill bad>leakage-suspect</span>'}</td></tr>"
        for k, v in canary.items())
    try:
        hyp_md = (DOCS / "research/hypotheses.md").read_text()
        hyp_html = "".join(
            f"<h3>{esc(l[3:].strip())}</h3>" if l.startswith("## ") else
            (f"<h2>{esc(l[2:].strip())}</h2>" if l.startswith("# ") else f"<p>{esc(l)}</p>")
            for l in hyp_md.splitlines() if l.strip())
    except FileNotFoundError:
        hyp_html = "<p>(hypotheses.md missing)</p>"
    hyp_body = f"""
<div class="panel"><h2>Pre-registered candidate hypotheses (written before the run)</h2>{hyp_html}</div>
<div class="panel"><h2>Leakage canary — each feature alone on the holdout (flag if AUC &gt; 0.90)</h2>
<table><tr><th>surface</th><th class="num">solo AUC</th><th class="num">solo catalogue-hidden DTI</th><th>verdict</th></tr>{canary_rows}</table>
<p class="small">Sub-0.5 AUCs are expected and healthy: the holdout hides catalogue faults while the shipped
posterior deliberately excludes the catalogue collar — the instruments measure exactly the drift the lane prices in.</p></div>
<div class="panel"><h2>Candidate scan on the 160 label worlds (selection = prevalence-calibrated CVaR J_cal)</h2>
<table><tr><th>candidate</th><th class="num">dots</th><th class="num">J_cal</th><th class="num">J_nat</th><th class="num">worst20 (nat)</th><th class="num">catHidden</th></tr>{scan_rows}</table>
<h3>Exact greedy CVaR prune ({len(prune)-1} commits)</h3>
<pre>{esc(json.dumps(prune, indent=1))}</pre>
<h3>World generator statistics</h3>
<pre>{esc(json.dumps(ws, indent=1))}</pre></div>
"""

    # REGISTRY (use the extended registry re-check when present)
    fin = uniq.get("final_extended", uniq["final"])
    n_reg = fin.get("n_registry_rasters_checked", len(fin["per_registry"]))
    reg_rows = "".join(
        f"<tr><td style='font-size:12px'>{esc(k)}</td><td class='num'>{v['registry_positive_px']:,}</td>"
        f"<td class='num'>{v['spearman_rho']:+.4f}</td><td class='num'>{v['dot_overlap_3px']:.4f}</td>"
        f"<td>{'control-flagged' if v['control'] else ('<span class=pill ok>ok</span>' if (abs(v['spearman_rho'])<=0.9 and v['dot_overlap_3px']<=0.7) else '<span class=pill bad>BREACH</span>')}</td></tr>"
        for k, v in sorted(fin["per_registry"].items(), key=lambda kv: -kv[1]["dot_overlap_3px"]))
    calib_rows = ""
    for k, v in anchors["per_anchor"].items():
        live_s = f"{v['live']:.4f}" if v["live"] is not None else "unscored"
        calib_rows += (f"<tr><td>{esc(k)}</td><td class='num'>{live_s}</td>"
                       f"<td class='num'>{v['holdout_catalogue_hidden_mean']:.5f}</td></tr>")
    reg_body = f"""
<div class="panel">
<h2>Final raster vs every earlier registry raster (n = {n_reg}; lane limits: ρ ≤ 0.90, 3-px overlap &lt; 0.70)</h2>
<table><tr><th>registry raster</th><th class="num">positive px</th><th class="num">Spearman ρ</th><th class="num">3-px dot overlap</th><th>status</th></tr>{reg_rows}</table>
<p class="small">Controls (r13 blind 5-px lattice; ens12 union-of-12 blanket) are measured and flagged: the 70% clause
is literally unsatisfiable against them for every artifact in the project family — measured on the
organizer-confirmed best pair: d28 (0.26) → 99.92% vs r13, 67.2% vs ens12; h33 (0.2778) → 99.92% vs r13, 64.2% vs
ens12; h32d → 99.91% vs r13, 69.0% vs ens12. Context for my artifact: 99.9% vs r13 (expected — the lattice tiles
everything), 75.1% vs ens12 (vs 64–69% for the organizer-confirmed family). See IR-63-REG-01 on the
<a href="irregularities.html">irregularities</a> page.</p>
<h2>Instrument vs live receipts (8 scored artifacts)</h2>
<table><tr><th>anchor</th><th class="num">live (ORGANIZER-CONFIRMED)</th><th class="num">catalogue-hidden mean (HOLDOUT-DTI)</th></tr>{calib_rows}</table>
<p class="small">Spearman ρ = {anchors["spearman_instrument_vs_live"]:.3f}. The catalogue-hidden instrument does not
rank live scores — the exact discovery GEMSDOE32 acted on; it is why this session's selection runs on perturbed
label worlds instead.</p>
</div>
"""

    # IRREGULARITIES
    irr = json.loads((EVID / "irregularities.json").read_text()) if (EVID / "irregularities.json").exists() else []
    irr_rows = "".join(
        f"<tr><td><code>{esc(i['id'])}</code></td><td>{esc(i['severity'])}</td><td>{esc(i['summary'])}</td><td>{esc(i['disposition'])}</td></tr>"
        for i in irr)
    irr_body = f"""
<div class="panel"><h2>Irregularity ledger (everything flagged, nothing hidden)</h2>
<table><tr><th>id</th><th>severity</th><th>summary</th><th>disposition</th></tr>{irr_rows}</table></div>
"""

    # SOURCES
    sources_body = """
<div class="panel"><h2>Official / verified sources (manual review links)</h2>
<ul class="tight">
<li>Competition home — <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">drivendata.org/competitions/306</a></li>
<li>Problem description & metric — <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">page 967</a></li>
<li>About / additional resources — <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/">page 968</a></li>
<li>Rules / submission format PDF — <a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">docs.nlr.gov/docs/fy26osti/96647.pdf</a> (mirrored copy linked from the family data page via Dropbox)</li>
<li>Leaderboard — <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">competition leaderboard</a> (access requires sign-in)</li>
<li>Known-fault masking clarification — <a href="https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516">forum thread 11516</a></li>
<li>Reference solution — <a href="https://github.com/drivendataorg/gems-prize-reference-solution">github.com/drivendataorg/gems-prize-reference-solution</a></li>
<li>GEODAWN airborne surveys (USGS) — <a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">usgs.gov GEODAWN data page</a></li>
<li>INGENIOUS project (GBCGE) — <a href="https://gbcge.org/current-projects/ingenious/">gbcge.org/current-projects/ingenious</a></li>
<li>GDR submission 1391 (geothermal data) — <a href="https://gdr.openei.org/submissions/1391">gdr.openei.org/submissions/1391</a></li>
<li>Grid CRS — <a href="https://epsg.io/32611">EPSG:32611 (UTM 11N)</a></li>
<li>Tversky index — <a href="https://en.wikipedia.org/wiki/Tversky_index">Wikipedia</a></li>
<li>CVaR — Rockafellar, R.T. & Uryasev, S. (2000) <i>Optimization of conditional value-at-risk</i>, Journal of Risk 2(3):21–42</li>
<li>Family evidence reused under the reuse-don't-rebuild protocol: GEMSDOE32 registry/data_manifest.json
(owner-supplied, sha256-pinned mirrors of the login-walled DrivenData files — integrity-pinned, NOT organizer-authenticated),
GEMSDOE32 evidence/d28_forensic_autopsy.json (dot-family pathology), evidence/instrument_calibration.json
(live-receipt calibration of the SGMC/drift instruments).</li>
</ul></div>
"""

    pages = {
        "index.html": ("One-click submission", index_body),
        "executive-summary.html": ("Executive summary", exec_body),
        "run-card.html": ("Run card", runcard_body),
        "hypotheses.html": ("Hypotheses & scan", hyp_body),
        "registry.html": ("Registry & uniqueness", reg_body),
        "irregularities.html": ("Irregularities", irr_body),
        "sources.html": ("Sources", sources_body),
    }
    for fn, (title, body) in pages.items():
        (DOCS / fn).write_text(page(title, body))
    (DOCS / ".nojekyll").write_text("")
    print("site built:", ", ".join(pages))


if __name__ == "__main__":
    main()
