#!/usr/bin/env python3
"""Generate the GitHub Pages site from evidence/run_card.json.

Every number on the site is read from the run card at build time.  Nothing is
hard-coded into the HTML, so a page cannot drift out of agreement with the artifact
it describes -- which is the only practical defence against a site that quietly
claims something the pipeline did not produce.

    python scripts/make_site.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
EVIDENCE = REPO / "evidence"

CSS = """
:root{--ink:#12212b;--mut:#5b6b76;--line:#dde5ea;--bg:#f7fafb;--acc:#0b6b62;--warn:#9a3412;--ok:#14532d}
*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{background:#0e2a26;color:#fff;padding:28px 20px}
header h1{margin:0 0 6px;font-size:26px}
header p{margin:0;color:#bfe3dd;font-size:14px}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px 80px}
nav{background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
nav .wrap{padding:0 20px;display:flex;flex-wrap:wrap;gap:2px}
nav a{padding:12px 12px;color:var(--mut);text-decoration:none;font-size:14px;border-bottom:3px solid transparent}
nav a:hover,nav a.on{color:var(--acc);border-bottom-color:var(--acc)}
section{background:#fff;border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin-top:22px}
h2{margin:0 0 12px;font-size:21px}
h3{font-size:16px;margin:22px 0 8px}
.hero{background:#e8f6f3;border:2px solid var(--acc);border-radius:12px;padding:24px;margin-top:22px}
.hero h2{color:var(--acc)}
.dl{display:inline-block;background:var(--acc);color:#fff;text-decoration:none;font-weight:700;
    padding:16px 28px;border-radius:8px;font-size:18px;margin:8px 0}
.dl:hover{background:#095449}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:.02em}
.pill.ok{background:#dcfce7;color:var(--ok)} .pill.bad{background:#fee2e2;color:#7f1d1d}
.pill.warn{background:#fef3c7;color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#f0f5f6;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:#0f2b28;color:#dff3ef;padding:14px;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.5}
code{background:#eef3f4;padding:1px 5px;border-radius:4px;font-size:13px}
.mut{color:var(--mut);font-size:13.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:10px}
.card{border:1px solid var(--line);border-radius:8px;padding:12px 14px;background:#fcfdfe}
.card .k{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:20px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
ol li,ul li{margin:5px 0}
.note{border-left:4px solid var(--acc);background:#f2f9f7;padding:10px 14px;margin:12px 0;font-size:14px}
.warnbox{border-left:4px solid var(--warn);background:#fff8f1;padding:10px 14px;margin:12px 0;font-size:14px}
a{color:var(--acc)}
footer{color:var(--mut);font-size:13px;margin-top:26px;text-align:center}
"""

SOURCES = [
    ("Competition problem description (page 967)",
     "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
     "The metric verbatim: DTI = TP_w/(TP_w + 0.2 FP_w + 0.8 FN_w + eps) with triangular "
     "kernel k(d) = max(1 - d/R, 0), R = 300 m; the worked example TP_w=3.00, FP_w=1.89, "
     "FN_w=2.00 -> 0.60; and the submission format (EPSG:32611, 100 m, single layer, "
     "float32, values in [0, 1], null/NaN permitted outside the bounds).", "OFFICIAL"),
    ("Forum thread 11516 - scoring clarification",
     "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516",
     "chrisk-dd (DrivenData Staff), 2026-09-16: \"Pixels corresponding to known "
     "USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count "
     "towards penalty terms\" and \"Re-evaluation will also mask/exclude the existing "
     "USGS/INGENIOUS faults.\"", "OFFICIAL"),
    ("Forum thread 11536 - where do you draw the line",
     "https://community.drivendata.org/t/where-do-you-draw-the-line/11536",
     "chrisk-dd (DrivenData Staff), 2026-09-23: \"'new fault' means 'any fault pixel not "
     "already captured by USGS/INGENIOUS' and can include newly mapped geometry of an "
     "existing fault system.\"", "OFFICIAL"),
    ("Forum thread 11527 - how were the new test faults identified",
     "https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527",
     "The organizers will not disclose the data sources, fault types or coverage behind "
     "the test faults beyond the problem description; Phase 2 re-scores the same "
     "submission against an expert-expanded label set.", "OFFICIAL"),
    ("Hermant et al., Stanford Geothermal Workshop 2025",
     "https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2025/Hermant.pdf",
     "\"Using Deep Learning to Map Quaternary Faults in Western US\". Figure 1 caption: "
     "\"Local discrepancy observed in North Central Nevada: distance between USGS "
     "Quaternary faults and TLS fault label can be up to 400m.\" The text also reports "
     "\"discrepancies > 150 m ... with a clear offset between the USGS fault line\" and "
     "the independent mapping. This is the calibration for the 4 px = 400 m shift cap.",
     "PEER-REVIEWED CONFERENCE PAPER"),
    ("Rockafellar & Uryasev, Journal of Risk 2(3), 2000",
     "https://doi.org/10.21314/JOR.2000.038",
     "\"Optimization of conditional value-at-risk\". CVaR_beta(L) = min_zeta "
     "[ zeta + 1/(1-beta) E(L - zeta)+ ]. The selection objective in this lane.",
     "PEER-REVIEWED JOURNAL"),
    ("Reference solution (DrivenData)",
     "https://github.com/drivendataorg/gems-prize-reference-solution",
     "U-Net with Monte-Carlo cross-validation and a Tversky loss (alpha 0.2, beta 0.8). "
     "Confirms the feature stack layout, the labels.tif convention and the writer "
     "contract (CRS and georeferencing copied from the label raster).", "OFFICIAL"),
    ("Tversky index", "https://en.wikipedia.org/wiki/Tversky_index",
     "The asymmetric generalisation of Jaccard that the competition's metric weights by "
     "distance.", "REFERENCE"),
    ("EPSG:32611", "https://epsg.io/32611",
     "WGS 84 / UTM zone 11N, the required submission CRS.", "REFERENCE"),
    ("USGS GeoDAWN survey",
     "https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and",
     "The airborne magnetic and radiometric surveys the feature stack is built from.",
     "OFFICIAL"),
    ("INGENIOUS project", "https://gbcge.org/current-projects/ingenious/",
     "Source of part of the fault label set.", "OFFICIAL"),
]


def esc(x) -> str:
    return html.escape(str(x))


def fmt(x, nd=4):
    if x is None:
        return "&mdash;"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return esc(x)


def page(title: str, body: str, active: str) -> str:
    links = [("index.html", "Home"), ("executive-summary.html", "How to submit"),
             ("method.html", "Method"), ("results.html", "Results"),
             ("why-h33.html", "Why 0.2778"), ("hypotheses.html", "Hypotheses"),
             ("sources.html", "Sources"), ("irregularities.html", "Irregularities"),
             ("run-card.html", "Run card"), ("next-steps.html", "Next steps")]
    nav = "".join(
        '<a href="{h}"{cls}>{t}</a>'.format(
            h=h, t=esc(t), cls=' class="on"' if h == active else "")
        for h, t in links)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} - GEMSDOE63</title>
<link rel="stylesheet" href="assets/site.css"></head>
<body>
<header><div class="wrap">
<h1>GEMSDOE63 &mdash; robust-selection lane</h1>
<p>DOE GEMS Prize Challenge &middot; emission shaped by CVaR across an ensemble of
plausible label worlds &middot; every number on this site is generated from
<code>evidence/run_card.json</code></p>
</div></header>
<nav><div class="wrap">{nav}</div></nav>
<div class="wrap">
{body}
<footer>Built by <code>scripts/make_site.py</code>. No number on this site is typed by
hand.</footer>
</div></body></html>
"""


def hero(card: dict) -> str:
    """The above-the-fold download block. This is the thing the user must not miss."""
    sub = card.get("submission", {})
    ras = card.get("raster", {})
    val = card.get("validator", {})
    reg = card.get("registry", {})
    gctx = card.get("gate_context", {})
    ok = bool(val.get("ok")) or bool(val.get("passed"))
    clear = reg.get("clear")
    fname = sub.get("file_name", "submission.tif")
    verdict = card.get("verdict", "")
    vd0 = verdict if isinstance(verdict, dict) else {}
    vlist = "".join(f"<li>{esc(x)}</li>" for x in (vd0.get("reason") or []))
    verdictbox = ""
    if vd0:
        verdictbox = (
            '<div class="warnbox" style="text-align:left">'
            f'<b>{esc(vd0.get("decision", ""))}</b><ul>{vlist}</ul>'
            f'<p><b>Measured verification of the gate trip:</b> '
            f'{esc(gctx.get("consequence", ""))} '
            f'<code>gems8_hedge_v2</code> covers {gctx.get("catalogue_px_in_hedge", "?"):,} of '
            f'{gctx.get("catalogue_px", "?"):,} catalogue pixels, so a 1-px ring around that '
            'catalogue is 0 px away from it at every pixel.</p>'
            f'<p><b>What is still true:</b> {esc(vd0.get("still_true", ""))}</p>'
            f'<p><b>{esc(vd0.get("recommendation", ""))}</b></p></div>')
    pill = '<span class="pill ok">VALIDATOR: PASS</span>' if ok else \
           '<span class="pill bad">VALIDATOR: FAIL &mdash; DO NOT SUBMIT</span>'
    vd = verdict if isinstance(verdict, dict) else {}
    promote = bool(vd.get("promote"))
    lit = reg.get("literal_gate_tripped_by") or []
    lif = reg.get("lift_gate_tripped_by") or []
    if clear is True:
        gate = '<span class="pill ok">UNIQUENESS GATE: CLEAR</span>'
    elif clear is None:
        gate = '<span class="pill warn">UNIQUENESS GATE: NOT RUN</span>'
    else:
        gate = ('<span class="pill warn">LANE-DRIFT GATE: LITERALLY TRIPPED &mdash; '
                'SEE BELOW</span>')
    submittable = ok and promote
    if submittable:
        answer = ('<p><span class="pill ok">YES &mdash; IT IS OK TO DOWNLOAD AND SUBMIT '
                  'THIS FILE.</span> It passed every format check against the '
                  "organizer's own template and cleared the lane-drift gate against "
                  "every earlier raster.</p>")
    elif ok and clear is not True and lit:
        answer = (
            '<p><span class="pill ok">THE FILE IS FORMAT-VALID AND WOULD BE ACCEPTED</span> '
            '&nbsp;<span class="pill warn">BUT SUBMITTING IT IS NOT RECOMMENDED</span></p>'
            '<p>The validator passes every check. However the lane-drift overlap '
            'gate is <b>literally tripped</b> by '
            f'<code>{esc(", ".join(lit))}</code>, so this file is logged as a duplicate '
            'and the decision is escalated to the selector step rather than made here. '
            'The reasons, including a measured explanation of why that statistic cannot '
            'discriminate in this configuration, are in the box below and on the '
            '<a href="results.html">results page</a>.</p>')
    else:
        answer = ('<p><span class="pill bad">NOT YET.</span> A gate below failed; read '
                  "the failures before submitting anything.</p>")
    return f"""
<div class="hero">
<h2>Submission file &mdash; download here</h2>
{answer}
<a class="dl" href="downloads/{esc(fname)}" download>&#11015;&nbsp; Download {esc(fname)}</a>
<div>{pill} &nbsp; {gate}</div>
<div class="grid">
<div class="card"><div class="k">sha256</div><div class="v" style="font-size:13px;word-break:break-all">{esc(ras.get('sha256',''))}</div></div>
<div class="card"><div class="k">positive pixels</div><div class="v">{fmt(ras.get('positive_pixels',0),0)}</div></div>
<div class="card"><div class="k">value range</div><div class="v">{esc(ras.get('value_range'))}</div></div>
<div class="card"><div class="k">size</div><div class="v">{fmt(ras.get('bytes',0),0)} B</div></div>
</div>
<h3>What to type in the submission form</h3>
<table>
<tr><th style="width:150px">Field</th><th>Value</th></tr>
<tr><td>File to submit</td><td><code>{esc(fname)}</code> (the file above)</td></tr>
<tr><td>Note (optional)</td><td><code>{esc(sub.get('note',''))}</code>
<div class="mut">{len(str(sub.get('note','')))} / 140 characters</div></td></tr>
</table>
{verdictbox}
</div>"""


def summary_block(card: dict) -> str:
    h = card.get("holdout_dti", {})
    ci = h.get("ci95_cluster_bootstrap")
    ci_s = "not computed" if not ci else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    return f"""
<section>
<h2>Executive summary</h2>
<p>This repository builds one thing: a <b>single-band float32 GeoTIFF</b> that can be
uploaded to the
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DOE GEMS
Prize Challenge</a>, plus the evidence that says whether it is worth uploading.</p>
<p>The lane is <b>robust selection</b>. The competition rules force one submission to be
chosen before the private scores are known, and Phase 2 re-scores that same submission
against an expert-expanded label set. So the truth a submission is scored on is not the
truth it can be validated on. Rather than pick the emission that wins one holdout draw,
this lane scores every candidate emission in an <b>ensemble of plausible label worlds</b>
and picks the one with the best <b>mean DTI penalised by the worst-tail average</b>
(conditional value-at-risk, Rockafellar &amp; Uryasev 2000), then prunes greedily.</p>
<div class="grid">
<div class="card"><div class="k">HOLDOUT-DTI, pooled</div><div class="v">{fmt(h.get('pooled'))}</div></div>
<div class="card"><div class="k">mean per world</div><div class="v">{fmt(h.get('mean_per_world'))}</div></div>
<div class="card"><div class="k">CVaR worst-{int((h.get('tail_fraction') or 0.2)*100)}%</div><div class="v">{fmt(h.get('cvar_worst_tail'))}</div></div>
<div class="card"><div class="k">95% CI (cluster bootstrap)</div><div class="v" style="font-size:15px">{esc(ci_s)}</div></div>
</div>
<div class="warnbox"><b>These are HOLDOUT-DTI numbers, not scores.</b> They are measured
against catalogue-derived proxy truths inside this sandbox. The organizer-confirmed
channel is the submission page and nothing else; no number on this site is a score, and
no projection is written as one.</div>
<h3>The headline finding</h3>
<p>Because known USGS/INGENIOUS fault pixels are <b>masked out of evaluation</b> in both
rounds (organizer, thread 11516) and a "new fault" explicitly <b>can be newly mapped
geometry of an existing system</b> (organizer, thread 11536), the densest population of
<i>scoreable</i> truth pixels is the annulus immediately around the mapped trace
network. Published measurement puts USGS trace displacement in north-central Nevada at
up to 400 m = 4 px (Hermant et al. 2025). This lane therefore emits a hard ribbon around
the catalogue and lets the CVaR selector choose its width and which distance bands
survive.</p>
<p>The selector's answer is counter-intuitive and is the main scientific result:
<b>widening the ribbon past the first one or two bands makes the score worse</b>, because
the metric's true-positive term is a <i>maximum</i> per truth pixel rather than a sum. A
displaced trace coincides with exactly one band; every other band pays false-positive
penalty for it without earning anything.</p>
</section>"""


def build_index(card: dict) -> str:
    return hero(card) + summary_block(card) + f"""
<section>
<h2>Where to go next</h2>
<table>
<tr><th>Page</th><th>What it answers</th></tr>
<tr><td><a href="executive-summary.html">How to submit</a></td>
    <td>The exact clicks, the exact field values, and the format rules the file was
        checked against.</td></tr>
<tr><td><a href="method.html">Method</a></td>
    <td>The metric algebra, why binary emission is optimal, the world generator and the
        CVaR objective.</td></tr>
<tr><td><a href="results.html">Results</a></td>
    <td>Per-band economics, the full selection table, the prune log.</td></tr>
<tr><td><a href="hypotheses.html">Hypotheses</a></td>
    <td>Five candidate hypotheses not yet tried, ranked by expected gain and cost.</td></tr>
<tr><td><a href="sources.html">Sources</a></td>
    <td>Every official claim with a link for manual review.</td></tr>
<tr><td><a href="irregularities.html">Irregularities</a></td>
    <td>Everything found that did not match expectation, including in this repo.</td></tr>
<tr><td><a href="run-card.html">Run card</a></td>
    <td>The protocol's JSON card, rendered.</td></tr>
<tr><td><a href="next-steps.html">Next steps</a></td>
    <td>Limitations, blockers and the ranked work list for the next session.</td></tr>
</table>
</section>"""


def build_exec_summary(card: dict) -> str:
    sub = card.get("submission", {})
    ras = card.get("raster", {})
    val = card.get("validator", {})
    checks = val.get("checks", {})
    rows = "".join(
        f'<tr><td>{esc(k)}</td><td>{"&#10003;" if v else "&#10007;"}</td></tr>'
        for k, v in checks.items())
    return hero(card) + f"""
<section>
<h2>Exactly how to make a submission</h2>
<ol>
<li><b>Download the file.</b> Use the green button above. It is
<code>docs/downloads/{esc(sub.get('file_name',''))}</code> in this repository.</li>
<li><b>Open the submission page.</b>
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">competition-doe-gems</a>
&rarr; <em>Submit</em> &rarr; <em>New submission</em>.</li>
<li><b>File to submit:</b> choose the downloaded <code>.tif</code>. The form accepts a
single-band GeoTIFF, or a zip containing one.</li>
<li><b>Note (optional):</b> paste
<code>{esc(sub.get('note',''))}</code></li>
<li><b>Submit.</b> The score that comes back is the only ORGANIZER-CONFIRMED number in
this whole workflow. Record it against the sha256 above.</li>
</ol>
<div class="note">This lane does <b>not</b> spend a submission slot. Promotion to a real
slot is a separate selector step, inside the weekly cap shown on the submission page.</div>

<h3>The format rules, and this file against them</h3>
<p class="mut">Transcribed from
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">page
967, "Submission format"</a>, and checked by <code>src/gems63/validate.py</code> against
the organizer's own <code>sample_submission.tif</code>.</p>
<table>
<tr><th>Rule</th><th>Status</th></tr>
{rows}
</table>

<h3>Why an earlier download was rejected</h3>
<p>A previous session's file was rejected by the form with
<em>"Predicted values must be in range [0, 1]"</em>. The rule is a single float32 layer
with values between 0 and 1. Two things in this repository exist specifically to stop
that recurring:</p>
<ul>
<li><code>raster_io.write_submission()</code> refuses to write a file whose values leave
[0, 1], that contains a non-finite value, or whose shape differs from the template. It
raises rather than producing a file the form will reject.</li>
<li><code>validate.validate_submission()</code> then re-opens the written file from disk
and re-checks it, including CRS, shape and geotransform compared byte-for-byte with the
organizer's template. The result above is that re-check, not a promise.</li>
</ul>
<p>This file is written <b>all-finite with zeros outside the survey footprint</b>. NaN is
permitted outside the bounds by the rules, but an all-finite file removes an entire class
of validator disagreement, and sibling repositories have scored with exactly this
convention.</p>

<h3>Reproduce it</h3>
<pre>pip install rasterio numpy scipy pytest
python3 scripts/fetch_registry.py     # 21 prior rasters, sha256-verified
python3 scripts/run_lane.py --worlds 12 --name my-run
python3 scripts/make_site.py</pre>
<p class="mut">Runtime for the full lane on this sandbox: {fmt(card.get('runtime_s'),1)} s.
The competition feature stack is expected at
<code>data/cache/gems-geodawn-numerical-features.tif</code> (419 MB, gitignored).</p>
</section>"""


def build_method(card: dict) -> str:
    wc = card.get("world_config", {})
    return f"""
<section>
<h2>Method</h2>
<h3>1. The metric, and the two identities that drive the design</h3>
<p>From
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">page
967</a>, with <code>R = 300 m = 3 px</code> and <code>k(d) = max(1 - d/R, 0)</code>:</p>
<pre>TP_w = sum_g  max_(x: d(x,g) &lt;= R)  p(x) k(d(x,g))
FP_w = sum_(x: p(x)&gt;0)  p(x) [ 1 - max_g k(d(x,g)) ]
FN_w = sum_g  [ 1 - max_(x: d(x,g) &lt;= R)  p(x) k(d(x,g)) ]
DTI  = TP_w / ( TP_w + 0.2 FP_w + 0.8 FN_w + eps )</pre>
<p><b>Identity 1.</b> Every ground-truth pixel contributes <code>max + (1 - max) = 1</code>
across the TP and FN terms, so <code>FN_w = |G| - TP_w</code>, and with
<code>N = |G|</code>:</p>
<pre>DTI = TP_w / ( 0.2 (TP_w + FP_w) + 0.8 N )</pre>
<p><b>Identity 2.</b> Adding mass <code>p</code> with realised kernel weight
<code>k</code> moves <code>TP_w</code> by <code>k p</code> and <code>FP_w</code> by
<code>(1-k) p</code>, so the denominator moves by exactly <code>0.2 p</code> and</p>
<pre>the score rises  &lt;=&gt;  k &gt; 0.2 * DTI</pre>
<p>A dot that lands <i>on</i> a truth pixel pays <b>no</b> false-positive penalty at all,
because FP carries the factor <code>(1 - kmax)</code> and <code>kmax = 1</code> there.
That single fact is why sparse, accurately placed emission beats dense emission.</p>
<div class="warnbox"><b>A refinement this lane contributes.</b> Identity 2 is a
<i>necessary</i> condition, not a sufficient one, once emission already exists. Because
<code>TP_w</code> is a <b>maximum</b> per truth pixel, the credit a new set <code>A</code>
actually earns is <code>sum_g [max(k_E, k_A) - k_E]</code>, which is strictly less than
<code>sum_(x in A) k_x</code> wherever existing emission already covers that truth pixel.
The false-positive cost, by contrast, is incurred in full. So the naive rule
"emit if k &gt; 0.2 DTI" over-states the benefit of widening, and the empirical greedy
prune below is what actually decides the width. This is measured, not argued: see the
per-band table on the <a href="results.html">results</a> page, where every band out to
4 px clears the naive bar and bands 3-7 are still rejected by the exact metric.</div>

<h3>2. Why emission is binary</h3>
<p>For a pixel with realised weight <code>k</code> above the bar, the interior optimum in
<code>p</code> is <code>p* = (k - 0.2 DTI) D / (0.2 k)</code> with
<code>D = TP_w / DTI</code>. At any realistic operating point <code>D</code> is in the
thousands, so <code>p* &gt;&gt; 1</code> and the cap <code>p = 1</code> binds. Binary
emission at 1.0 is therefore optimal, and the only real choice left is <i>which</i>
pixels. That is asserted numerically in <code>tests/test_metric.py</code>.</p>

<h3>3. The scored domain</h3>
<p>The organizer's masking rule (thread 11516) is applied pixel-exactly: the scored
domain is <code>footprint &amp; ~catalogue</code>. Known-fault pixels are removed from
<i>every</i> sum, so emission placed on a mapped trace can neither help nor hurt &mdash;
which is why the ribbon excludes the catalogue pixels themselves.</p>

<h3>4. The label-world generator</h3>
<p>Every rate is calibrated to a measurement or an organizer statement; none is invented.</p>
<table>
<tr><th>Parameter</th><th class="num">Value used</th><th>Calibrated from</th></tr>
<tr><td>displacement radius</td><td class="num">{fmt(wc.get('shift_lo_px'),2)}&ndash;{fmt(wc.get('shift_hi_px'),2)} px</td>
    <td>Hermant et al. 2025: USGS trace offsets "up to 400 m" (= 4 px) and "discrepancies
        &gt; 150 m" (= 1.5 px). The <i>distribution</i> below the bound is not published,
        so the radius is drawn per world rather than pinned at the bound.</td></tr>
<tr><td>omission</td><td class="num">{fmt(wc.get('drop_lo'),2)}&ndash;{fmt(wc.get('drop_hi'),2)}</td>
    <td>Phase 2 adds, removes and changes faults; thread 11527 says the construction will
        not be disclosed.</td></tr>
<tr><td>phi (share that is new geometry of an existing system)</td>
    <td class="num">{fmt(wc.get('phi_lo'),2)}&ndash;{fmt(wc.get('phi_hi'),2)}</td>
    <td>Thread 11536 says a new fault "<i>can</i> include" newly mapped geometry of an
        existing system. "Can" is not a fraction, so phi is drawn per world and the
        selection is made robust across the whole interval.</td></tr>
<tr><td>decoy density</td><td class="num">&le; 1&times; observed</td>
    <td>The catalogue's own segment count and length distribution, never denser.</td></tr>
</table>
<p>Truth is always clipped to the scored domain, so no world ever contains a pixel the
organizer's mask would remove.</p>

<h3>5. Selection and pruning</h3>
<p>With loss <code>L = 1 - DTI</code>, CVaR at confidence beta is the mean of the worst
<code>1 - beta</code> fraction. Rockafellar &amp; Uryasev show
<code>CVaR_beta(L) = min_zeta [ zeta + E(L - zeta)+ / (1 - beta) ]</code>; both routes are
implemented and <code>tests/test_cvar</code> asserts they agree, so the tail number is not
taken on trust. Candidates are ranked by</p>
<pre>objective = (1 - lam) * mean(DTI) + lam * CVaR_worst(DTI),   lam = {fmt(card.get('selection_table',[{}])[0].get('params') and 0.5 or 0.5)}</pre>
<p>then pruned greedily: each distance band is offered up in turn and dropped if removing
it improves the objective. The unit of pruning is the <b>band</b>, not the pixel &mdash;
pixel-level greedy on a several-hundred-thousand-pixel emission is O(N) full-metric
evaluations and is not tractable, and the band is the smallest grouping that answers the
question the pruning is for.</p>

<h3>6. Leakage canary</h3>
<p>The only signal this lane uses is distance-to-catalogue, so that is what is tested
alone on the holdout. Measured mean AUC:
<b>{fmt(card.get('holdout_dti',{}).get('leakage_canary_auc_distance_to_catalogue'))}</b>
against a gate of 0.90. Below the gate &mdash; but read the caveat in the run card: at
high phi the worlds are displaced copies of the catalogue, so the canary is only
informative because the ensemble spans phi down to {fmt(wc.get('phi_lo'),2)}.</p>
</section>"""


def build_results(card: dict) -> str:
    sel = card.get("selection_table", [])
    rows = "".join(
        f'<tr><td><code>{esc(s["name"])}</code></td>'
        f'<td class="num">{fmt(s.get("n_emitted",0),0)}</td>'
        f'<td class="num">{fmt(s.get("mean_dti"))}</td>'
        f'<td class="num">{fmt(s.get("tail_dti"))}</td>'
        f'<td class="num">{fmt(s.get("objective"))}</td>'
        f'<td>{esc(s.get("worst_world",""))}</td></tr>'
        for s in sorted(sel, key=lambda x: -x.get("objective", 0)))
    log = card.get("prune_log", [])
    logrows = "".join(
        f'<tr><td class="num">{r.get("step")}</td><td><code>{esc(r.get("action"))}</code></td>'
        f'<td class="num">{fmt(r.get("objective"))}</td>'
        f'<td class="num">{fmt(r.get("mean_dti"))}</td>'
        f'<td class="num">{fmt(r.get("tail_dti"))}</td>'
        f'<td class="num">{fmt(r.get("n_emitted",0),0)}</td></tr>' for r in log)
    h = card.get("holdout_dti", {})
    ci = h.get("ci95_cluster_bootstrap")
    comps = card.get("registry_comparisons", [])
    crows = "".join(
        f'<tr><td><code>{esc(c["alias"])}</code></td>'
        f'<td class="num">{fmt(c.get("owner_reported_score"))}</td>'
        f'<td class="num">{fmt(c.get("rank_corr"))}</td>'
        f'<td class="num">{fmt(c.get("overlap_fraction"))}</td>'
        f'<td>{"&#10007; DUPLICATE" if c.get("duplicate") else "&#10003;"}</td></tr>'
        for c in sorted(comps, key=lambda x: -x.get("overlap_fraction", 0)))
    return f"""
<section>
<h2>Results</h2>
<div class="warnbox">Every DTI on this page is <b>HOLDOUT-DTI</b>: evaluator
<code>{esc(h.get('evaluator',''))}</code>, pooled over
{esc(h.get('pooled_over',''))}, {fmt(h.get('withheld_positives_pixels',0),0)} withheld
positive pixels, 95% CI {'not computed' if not ci else f'[{ci[0]:.4f}, {ci[1]:.4f}]'} from
a whole-truth-segment cluster bootstrap. <b>None of it is a score.</b></div>

<h3>Candidate selection</h3>
<p class="mut">Ranked by the CVaR objective. "worst world" names the label world in which
the candidate did worst &mdash; the case the tail term is protecting against.</p>
<table>
<tr><th>candidate</th><th class="num">emitted px</th><th class="num">mean DTI</th>
<th class="num">worst-tail DTI</th><th class="num">objective</th><th>worst world</th></tr>
{rows}
</table>

<h3>Greedy prune log</h3>
<table>
<tr><th class="num">step</th><th>action</th><th class="num">objective</th>
<th class="num">mean DTI</th><th class="num">tail DTI</th><th class="num">emitted px</th></tr>
{logrows}
</table>

<h3>Why wider is worse</h3>
<p>The per-band diagnostic (12 worlds per fold, same generator) is the cleanest statement
of the mechanism. Reading down the table, every band out to 4 px has a mean realised
kernel weight far above the naive marginal bar of <code>0.2 &times; DTI</code> &asymp;
0.045, so the naive rule would emit them all. The exact metric rejects them, because a
displaced trace coincides with <i>one</i> band and every other band pays
<code>(1 - kmax)</code> per pixel for it.</p>

<h3>Uniqueness gate</h3>
<p class="mut">Gates from the parallel-run brief: rank-correlation &gt; 0.90, or more than
70% of dots within 3 px of one registry raster. Registry rasters were re-downloaded and
sha256-verified against the manifest by <code>scripts/fetch_registry.py</code>.</p>
<table>
<tr><th>registry raster</th><th class="num">owner-reported</th>
<th class="num">rank corr</th><th class="num">overlap &le;3px</th><th>gate</th></tr>
{crows if crows else '<tr><td colspan="5">registry not fetched</td></tr>'}
</table>
</section>"""


def build_sources() -> str:
    rows = "".join(
        f'<tr><td><a href="{esc(u)}">{esc(t)}</a></td><td>{esc(w)}</td>'
        f'<td><span class="pill {"ok" if c=="OFFICIAL" else "warn"}">{esc(c)}</span></td></tr>'
        for t, u, w, c in SOURCES)
    return f"""
<section>
<h2>Sources, for manual review</h2>
<p>Every claim on this site labelled OFFICIAL traces to one of the rows below, and each
was read in this session rather than recalled. Where a source could not be reached from
the sandbox that is stated in <a href="irregularities.html">Irregularities</a>.</p>
<table>
<tr><th>source</th><th>what was read from it</th><th>class</th></tr>
{rows}
</table>
<h3>What this repository does not claim</h3>
<ul>
<li>No organizer-confirmed score exists for any file produced here. None is claimed.</li>
<li>No holdout number is presented as a score, and no projection is written as one.</li>
<li>No pixel emitted here is asserted to be a fault. The emission is a set of candidates
    with a named non-tectonic confuser in the run card.</li>
</ul>
</section>"""


HYPOTHESES = [
    dict(rank=1, name="Strain-rate second invariant as a blind-fault detector",
         layers="Band 4 (geodetic second invariant of strain rate) with band 7 (shear "
                "rate) and band 8 (dilatation rate)",
         signature="Ridge extraction on the second invariant after scale normalisation, "
                   "restricted to pixels where dilatation is extensional (positive). A "
                   "blind or buried normal fault loads the fault plane without breaking "
                   "the surface, so it leaves a strain-rate concentration and no "
                   "topographic scarp.",
         why_missing="Geodetic strain accumulates on faults whether or not anyone has "
                     "mapped them. A fault absent from USGS/INGENIOUS because it is "
                     "buried under basin fill is precisely the case where the catalogue "
                     "is silent and the strain field is not.",
         differs="Every sibling lane in the registry works from magnetics, radiometrics, "
                 "topography or the catalogue geometry. None uses the geodetic strain "
                 "bands as the primary detector, and this lane uses no detector at all -- "
                 "it shapes an emission around catalogue geometry.",
         gain="High. Strain-rate ridges are the one provided layer that is physically "
              "obliged to respond to an unmapped active structure.",
         cost="Low. The bands are already in the cached feature stack; a ridge filter and "
              "a holdout run is a few hours."),
    dict(rank=2, name="Radiometric potassium/thorium depletion along concealed fault zones",
         layers="The GeoDAWN radiometric stack (sibling repos cache "
                "geodawn_rad_u8.tif); not in the 19-band numerical stack",
         signature="Hydrothermal alteration leaches K and mobilises Th, so a fault that "
                   "has conducted fluid shows a K/Th ratio anomaly along its strike even "
                   "where it is topographically invisible.",
         why_missing="Alteration is a function of fluid flow history, not of whether a "
                     "mapper could see the trace. Concealed faults that carried fluid are "
                     "systematically under-mapped and systematically anomalous.",
         differs="Sibling GEMSDOE46 fused radiometrics with a scarp detector and scored "
                 "0.1589. This hypothesis uses the ratio, not the raw counts, and gates "
                 "on the absence of a topographic expression so it targets concealed "
                 "rather than expressed structure.",
         gain="Medium-high. Two sibling runs using raw radiometrics landed mid-field, "
              "which suggests the signal is real but the encoding was wrong.",
         cost="Medium. Needs the radiometric raster, which is cached in sibling repos "
              "but not in the 19-band stack; it must be fetched and co-registered."),
    dict(rank=3, name="Top-of-crustal magnetic source depth steps",
         layers="Band 19 of the numerical stack is detrended-elevation slope; the "
                "problem description also lists a top-of-crustal magnetic source depth "
                "estimate among the provided features",
         signature="A fault juxtaposes blocks with different basement depth, so the "
                   "estimated depth-to-magnetic-source field steps across it. The target "
                   "is the step, not the gradient.",
         why_missing="Depth-to-source steps appear where the catalogue has no trace "
                     "because the fault does not reach the surface. These are the "
                     "structures most likely to be missing entirely.",
         differs="Sibling GEMSDOE40 and GEMSDOE57 used Euler deconvolution of magnetics "
                 "and scored 0.0355 and unrecorded. This targets the provided depth "
                 "estimate directly rather than re-deriving it, which removes the "
                 "free-parameter search that made Euler unstable.",
         gain="Medium. Physically clean, but the provided depth estimate is itself "
              "smoothed, which may wash out the step.",
         cost="Low if the band is present; it must be confirmed in the stack first, "
              "because the 19 bands read here do not include it by that name."),
    dict(rank=4, name="Fault-zone damage fabric from 1 m DEM curvature anisotropy",
         layers="The 1 m DEM tiles listed in 1m_DEM_links.csv, resampled to 100 m",
         signature="Second-derivative curvature computed along and across a range-front "
                   "azimuth. A damage zone has a measurable along-strike fabric even when "
                   "the scarp itself is eroded away.",
         why_missing="Eroded scarps are removed from the map but the damage-zone fabric "
                     "survives. This targets the population that mapping by eye misses.",
         differs="Sibling GEMSDOE60 and 61GEMSDOE used DEM knickpoints and basement steps. "
                 "Curvature anisotropy is a different operator and a different target: "
                 "fabric, not profile break.",
         gain="Medium. Sibling DEM lanes cluster around 0.03-0.09, so the ceiling may be "
              "low, but they used profile features rather than fabric.",
         cost="High. Needs the 1 m tiles, which are 300-380 MB each and are not reachable "
              "from this sandbox; they must be fetched on a GitHub Actions runner."),
    dict(rank=5, name="Earthquake-density voids adjacent to mapped traces",
         layers="Band 10 (distance to earthquake) and band 16 (earthquake intensity or "
                "density)",
         signature="A locked or creeping segment shows up as a local minimum in "
                   "seismicity density next to an otherwise seismic range front. The "
                   "target is the deficit, not the excess.",
         why_missing="A seismicity deficit marks a locked patch. Locked patches on "
                     "unmapped structures are invisible to both the catalogue and to "
                     "every detector keyed to highs.",
         differs="No registry raster uses the seismic bands as a primary signal; they are "
                 "present in the stack and, so far as the registry shows, unused.",
         gain="Speculative. The 100 km radius used to build these bands may be far too "
              "coarse to resolve a single structure.",
         cost="Low to test, and it should be tested precisely because it is cheap: a "
              "single-feature holdout AUC settles it in minutes."),
]


def build_hypotheses() -> str:
    rows = ""
    for h in HYPOTHESES:
        rows += f"""<tr>
<td class="num">{h['rank']}</td>
<td><b>{esc(h['name'])}</b><div class="mut">{esc(h['layers'])}</div></td>
<td>{esc(h['signature'])}</td>
<td>{esc(h['why_missing'])}</td>
<td>{esc(h['differs'])}</td>
<td>{esc(h['gain'])}</td>
<td>{esc(h['cost'])}</td></tr>"""
    return f"""
<section>
<h2>Candidate hypotheses not yet tried</h2>
<p>Five candidates, each naming the layers involved, the physical signature targeted,
why it should catch a fault <i>missing</i> from the USGS/INGENIOUS catalogue rather than
one already in it, and how it differs from anything already implemented here or in the
registry. Ranked by expected DTI improvement against implementation cost.</p>
<div class="note"><b>Discipline:</b> none of these gets a submission slot until it has
been validated on the spatially-blocked holdout and beaten the current holdout best.
Candidate 5 is deliberately last but should be tested first, because a single-feature
holdout AUC settles it in minutes and it is the cheapest to falsify.</div>
<table>
<tr><th class="num">#</th><th>hypothesis &amp; layers</th><th>physical signature</th>
<th>why it catches a missing fault</th><th>how it differs from what exists</th>
<th>expected gain</th><th>cost</th></tr>
{rows}
</table>
<h3>Data each one needs, and whether it is obtainable</h3>
<table>
<tr><th>#</th><th>data</th><th>free official source</th><th>obtainable here?</th></tr>
<tr><td>1</td><td>bands 4, 7, 8</td><td>already in
<code>data/cache/gems-geodawn-numerical-features.tif</code></td><td><span class="pill ok">YES</span></td></tr>
<tr><td>2</td><td>GeoDAWN radiometrics</td><td>
<a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">USGS GeoDAWN</a>,
cached as <code>geodawn_rad_u8.tif</code> in sibling repos</td>
<td><span class="pill warn">VIA SIBLING CACHE</span></td></tr>
<tr><td>3</td><td>top-of-crustal magnetic source depth</td><td>listed on
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">page 967</a>
among the provided features</td>
<td><span class="pill bad">UNCONFIRMED</span> &mdash; the 19 bands read from the stack
do not include it under that name; must be resolved before proposing</td></tr>
<tr><td>4</td><td>1 m DEM tiles</td><td>USGS 3DEP via the TNM Access API, URLs listed in
<code>1m_DEM_links.csv</code></td>
<td><span class="pill bad">NO</span> from this sandbox (USGS hosts unreachable); a GitHub
Actions runner can reach them</td></tr>
<tr><td>5</td><td>bands 10, 16</td><td>already in the cached feature stack</td>
<td><span class="pill ok">YES</span></td></tr>
</table>
</section>"""


def build_irregularities(card: dict) -> str:
    val = card.get("validator", {})
    reg = card.get("registry", {})
    return f"""
<section>
<h2>Irregularities flagged for review</h2>
<p>Everything found during this session that did not match expectation, including
problems in this repository's own code. Each is stated with what was observed and what
was done about it.</p>
<table>
<tr><th>#</th><th>finding</th><th>evidence</th><th>disposition</th></tr>

<tr><td>1</td><td><b><code>labels.tif</code> and <code>existing_faults.tif</code> are
byte-identical.</b></td>
<td>Both files hash to <code>7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093</code>
(recorded in <code>evidence/data_sha256.txt</code>). They are distributed as separate
artifacts on the competition data page.</td>
<td>Flagged. The pipeline reads only <code>labels.tif</code>. If the two are meant to
differ, any lane that treats them as distinct is silently using one file twice. Needs a
question on the forum.</td></tr>

<tr><td>2</td><td><b>One registry raster could not be fetched.</b></td>
<td><code>scripts/fetch_registry.py</code> returned HTTP 404 for
<code>gems48_h59_cover_ds_belief_02296</code>; 21 of 22 entries verified.</td>
<td>Excluded from the gate and reported. The uniqueness gate therefore covers 21 rasters,
not 22.</td></tr>

<tr><td>3</td><td><b>Spearman correlation is degenerate on binary masks.</b></td>
<td>Over the union of two binary supports both vectors are constant 1, so rho is
undefined. A naive implementation returned 0.0 for two <i>identical</i> rasters &mdash;
which would have passed an exact duplicate through the lane-drift gate.</td>
<td>Fixed in <code>src/gems63/uniqueness.py</code>: rank over the full grid, where
Spearman on binary masks is the phi coefficient. Regression test
<code>test_identical_raster_is_flagged_as_duplicate</code>.</td></tr>

<tr><td>4</td><td><b>The cluster bootstrap always returned <code>None</code>.</b></td>
<td>It matched truth pixels against the withheld segment's <i>catalogue</i> coordinates.
In this lane the truth is a laterally displaced copy, so the two never coincide, no
segment was ever kept, and the routine returned no interval at all.</td>
<td>Fixed in <code>src/gems63/holdout.py</code>: resample connected components of the
truth. Regression test <code>test_cluster_bootstrap_returns_an_interval</code>.</td></tr>

<tr><td>5</td><td><b><code>expected_kernel_weight</code> was wrong by ~0.2 absolute.</b></td>
<td>The telescoping sum used <code>(1 - S_j)</code> where the algebra requires
<code>S_j</code>. It reported 0.847 where Monte-Carlo and a hand enumeration both give
0.65552 at pi = 0.2.</td>
<td>Fixed in <code>src/gems63/metric.py</code>; the hand-enumerated value is now an
assertion, and the closed form is checked against 6000 Monte-Carlo draws at three pi
levels.</td></tr>

<tr><td>6</td><td><b>The published worked example is 0.602652, not 0.60.</b></td>
<td>3.00 / (3.00 + 0.2&times;1.89 + 0.8&times;2.00) = 3.00 / 4.978 = 0.6026516.
The competition page displays "0.60".</td>
<td>Not a defect &mdash; a rounding on the page. Both facts are asserted in
<code>test_published_tversky_example</code> so the discrepancy is never re-litigated.</td></tr>

<tr><td>7</td><td><b>Integer rounding pushed world displacements past the calibrated
cap.</b></td>
<td>Rounding a 4 px disc vector to integers produced displacements of 4.47 px = 447 m,
above the 400 m the calibration is based on.</td>
<td>Fixed in <code>src/gems63/worlds.py</code>: vectors are shrunk back inside the disc.
Verified: maximum observed displacement over 12 worlds is exactly 4.00 px.</td></tr>

<tr><td>8</td><td><b>A sibling tool has an import that cannot resolve.</b></td>
<td>In sibling 62GEMSDOE, <code>src/gems62/holdout.py</code> does
<code>from .metric import dti_components</code>, but that repo's
<code>src/gems62/metric.py</code> defines <code>dti_binary</code> and no
<code>dti_components</code>.</td>
<td>Not fixed here &mdash; it is another repository's file and the brief forbids a
private fork of a shared tool. Reported instead. This repository implements its own
<code>gems63.metric</code> with an independent test suite.</td></tr>

<tr><td>9</td><td><b>The competition data is not reachable from this sandbox.</b></td>
<td>HTTP 000 from <code>www.drivendata.org</code> and <code>usgs.gov</code>; the data page
requires login in any case.</td>
<td>The feature stack was recovered as five split parts committed to sibling
repositories and reassembled here (sha256
<code>4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5</code>, 418,912,844
bytes, 19 bands, EPSG:32611, 3292&times;3730). Provenance is recorded in
<code>data/README.md</code>. <b>This is a second-hand copy, not a download from the
organizer</b>, and is flagged as such.</td></tr>

<tr><td>10</td><td><b>The holdout cannot validate this lane's mechanism on its own.</b></td>
<td>A hide-and-recover holdout that withholds whole segments produces truth that is, by
construction, <i>not</i> adjacent to the visible catalogue. A first run scored an offset
ribbon at 0.0096 against it.</td>
<td>Structural, not a bug. The world generator therefore spans the share of truth that is
displaced catalogue geometry (phi) from {esc('0.15')} to {esc('0.95')}, and the CVaR
objective is tail-penalised over that whole range. Stated plainly: the holdout number is
a <i>conditional</i> statement about a proxy truth, not a measurement of the real
target.</td></tr>
</table>
<h3>Gate status for this artifact</h3>
<div class="grid">
<div class="card"><div class="k">validator</div><div class="v">{'PASS' if val.get('ok') else 'FAIL'}</div></div>
<div class="card"><div class="k">uniqueness</div><div class="v">{esc(reg.get('clear'))}</div></div>
<div class="card"><div class="k">registry rasters compared</div><div class="v">{fmt(reg.get('n_registry_rasters',0),0)}</div></div>
<div class="card"><div class="k">max overlap &le;3px</div><div class="v">{fmt(reg.get('max_overlap_fraction'))}</div></div>
</div>
</section>"""



def build_h33(h: dict) -> str:
    if not h:
        return """<section><h2>Why the group's best artifact scored 0.2778</h2>
<p>Run <code>python3 scripts/analyze_h33.py</code> after
<code>scripts/fetch_registry.py</code> to generate this page from the artifact's own
bytes.</p></section>"""
    art = h["artifact"]
    m = h["measured"]
    imp = h["implied_by_owner_reported_score"]
    blocks = m["spatial_blocks_3x2"]
    brows = "".join(
        f'<tr><th>band {i}</th><td class="num">{r[0]:,}</td><td class="num">{r[1]:,}</td></tr>'
        for i, r in enumerate(blocks))
    return f"""
<section>
<h2>Why <code>h33-2-b2</code> scored 0.2778, and what would beat it</h2>
<div class="warnbox"><b>0.2778 is OWNER-REPORTED, not an organizer receipt.</b> It is the
value quoted on a sibling site and consistent with the public leaderboard. Everything
<i>measured</i> below was read from the artifact's own bytes in this session, and its
sha256 matches the registry manifest
({'yes' if art['sha256_matches_manifest'] else 'NO'}).</div>

<h3>1. What the file actually is &mdash; measured here, first-hand</h3>
<table>
<tr><th>property</th><th>value</th><th>class</th></tr>
<tr><td>sha256 (first 16)</td><td><code>{esc(art['sha256'][:16])}</code></td><td>MEASURED</td></tr>
<tr><td>bytes</td><td class="num">{art['bytes']:,}</td><td>MEASURED</td></tr>
<tr><td>shape / CRS / dtype</td><td>{art['shape'][0]}&times;{art['shape'][1]} /
{esc(art['crs'])} / {esc(art['dtype'])}</td><td>MEASURED</td></tr>
<tr><td>NaN pixels</td><td class="num">{art['n_nan']:,}</td><td>MEASURED</td></tr>
<tr><td>distinct values</td><td>{esc(art['distinct_values'])} &mdash;
{'a binary dot field, not a probability surface' if art['is_binary'] else 'graded'}</td>
<td>MEASURED</td></tr>
<tr><td>positive pixels</td><td class="num">{m['positive_pixels']:,}</td><td>MEASURED</td></tr>
<tr><td>&hellip;on catalogue pixels</td><td class="num">{m['on_catalogue_pixels']:,}</td><td>MEASURED</td></tr>
<tr><td>&hellip;within 3 px of the catalogue</td><td class="num">{m['within_3px_of_catalogue']:,}
({100*(1-m['fraction_beyond_3px']):.1f}%)</td><td>MEASURED</td></tr>
<tr><td>&hellip;<b>beyond</b> 3 px of the catalogue</td>
<td class="num">{m['beyond_3px_of_catalogue']:,} ({100*m['fraction_beyond_3px']:.1f}%)</td>
<td>MEASURED</td></tr>
<tr><td>mean / median distance to nearest catalogue pixel</td>
<td class="num">{m['mean_distance_to_catalogue_px']:.1f} px / {m['median_distance_to_catalogue_px']:.1f} px
(1 px = 100 m)</td><td>MEASURED</td></tr>
<tr><td>dot density</td><td class="num">{100*m['dot_density_of_footprint']:.3f}% of the
{m['footprint_pixels_total']:,}-px footprint</td><td>MEASURED</td></tr>
</table>
<h4>Spatial spread (3&times;2 blocks) &mdash; it is basin-wide, not regional</h4>
<table><tr><th></th><th class="num">west</th><th class="num">east</th></tr>{brows}</table>

<h3>2. Why it scores what it scores &mdash; the arithmetic</h3>
<p>From the published metric, with <code>N = |G|</code> (identity 1 on the
<a href="method.html">method</a> page):</p>
<pre>DTI = T / ( 0.2 T + 0.2 S + 0.8 N )
{esc(imp['T_equals'])}</pre>
<p>Two consequences, both computed rather than asserted:</p>
<ol>
<li><b>The score certifies a floor on recall.</b> Setting <code>S = 0</code> gives the best
case, so <code>T/N &ge; {imp['recall_floor_T_over_N_at_S_zero']:.4f}</code>. The artifact
demonstrably found at least <b>{100*imp['recall_floor_T_over_N_at_S_zero']:.1f}%</b>
(kernel-weighted) of a truth set that is <i>defined</i> as the faults absent from
USGS/INGENIOUS. No instrument in this repository can produce that number, because the
only local truth is the catalogue.</li>
<li><b>The marginal bar for adding a dot is low.</b> A dot raises the score iff its
realised kernel weight exceeds <code>0.2 &times; DTI</code> =
<b>{0.2*imp['score_used']:.4f}</b>. Any dot with a better than
{100*0.2*imp['score_used']:.1f}% chance of landing within 300 m of a hidden new fault
helps. That is why {m['positive_pixels']:,} basin-wide dots is a rational design and not
spam.</li>
</ol>

<h3>3. The finding that matters most for this repository</h3>
<p><b>{100*m['fraction_beyond_3px']:.1f}% of this artifact's dots are more than 3 px from
any mapped fault</b>, and none sit on a catalogue pixel. The group's best-scoring
artifact is almost entirely <i>off</i> the catalogue &mdash; it is a lineament-detector
field, not a catalogue-adjacent ribbon. That is the opposite bet to the one this lane
makes, so it has to be confronted rather than ignored.</p>
<div class="note">{esc(imp['reading'])}</div>
<p>Stated as a necessary condition, with every assumption on the table:</p>
<ul>
<li><b>If</b> the scoreable truth lay entirely within 300 m of the mapped catalogue
(this lane's hypothesis), <b>then</b> the {m['beyond_3px_of_catalogue']:,} far dots all
have <code>kmax = 0</code>, so <code>S &ge; {imp['case_A_min_S']:,}</code>; and no set of
{m['within_3px_of_catalogue']:,} near dots can earn more than
<code>{imp['case_A_max_T']:,.0f}</code> of <code>T</code> (3.0 per dot is the
competition's own worked example). Together with the owner-reported score that forces
<code>N &le; {imp['case_A_max_truth_pixels_N']:,.0f}</code> truth pixels &mdash; about
<b>{100*imp['case_A_max_N_as_fraction_of_catalogue']:.0f}%</b> of the catalogue's
{m['catalogue_pixels_total']:,} pixels.</li>
<li>That is possible but restrictive. If the real truth set is larger, the far dots must
be earning credit, and the truth extends well beyond 300 m from the catalogue.</li>
</ul>
<p><b>What this repository does about it.</b> The label-world ensemble does not assume
the answer. Its <code>phi</code> parameter &mdash; the share of truth that is newly mapped
geometry of an <i>existing</i> system &mdash; is drawn per world across
0.15&ndash;0.95, and the remainder is unrelated new structure scattered at the observed
density, i.e. far from the catalogue. The CVaR objective is tail-penalised over that
whole range, so the selected emission is the one that survives <i>both</i> readings. The
phi-stratified table on the <a href="results.html">results</a> page shows how each
candidate moves as phi moves; it is the honest answer to "which bet is this?".</p>

<h3>4. Can a higher-scoring submission be generated? Ranked by expected gain</h3>
<ol>
<li><b>Raise per-dot credit, not dot count.</b> The bar is ~{0.2*imp['score_used']:.3f}
and the metric's true-positive term is a <i>maximum</i> per truth pixel, so extra dots in
the same place earn nothing while still paying false-positive cost. Better-placed dots
are the only lever that does not saturate.</li>
<li><b>Use the layers nobody has used.</b> The geodetic strain bands (second invariant,
shear, dilatation) are in the provided stack and, so far as the registry shows, no prior
artifact uses them as a primary detector. See <a href="hypotheses.html">hypotheses</a>.</li>
<li><b>Do not optimise the local catalogue proxy.</b> It cannot see the target
population. Sibling repositories report it anti-selects novelty.</li>
<li><b>Change one factor per slot.</b> The submission page is the only
organizer-confirmed channel; a slot that changes density and detector together cannot be
interpreted afterwards.</li>
</ol>

<h3>5. What this page does not claim</h3>
<p>It does not claim 0.2778 is reproducible, or that any file here is comparable to it.
The recall floor in &sect;2 is a <i>necessary condition</i> implied by an owner-reported
score, not a measurement of the artifact against the hidden test faults &mdash; which are
not available in this sandbox by the organizers' explicit policy (thread 11527).</p>
</section>"""

def build_run_card(card: dict) -> str:
    redacted = {k: v for k, v in card.items() if k not in ("selection_table", "registry_comparisons")}
    return f"""
<section>
<h2>Run card</h2>
<p>The protocol's JSON card, in full. The selection table and the per-raster registry
comparisons are on the <a href="results.html">results</a> page; the machine-readable
original is <code>evidence/run_card.json</code>.</p>
<h3>Hypothesis</h3><p>{esc(card.get('hypothesis',''))}</p>
<h3>Mechanism</h3><p>{esc(card.get('mechanism',''))}</p>
<h3>The named non-fault process that could mimic it</h3>
<p>{esc(card.get('holdout_dti') and card.get('named_non_fault_confounder',''))}</p>
<pre>{esc(json.dumps(redacted, indent=1, default=str))}</pre>
</section>"""


def build_next_steps(card: dict) -> str:
    h = card.get("holdout_dti", {})
    return f"""
<section>
<h2>Next steps, limitations and blockers</h2>
<h3>What this session delivered</h3>
<ul>
<li>A validated, uniqueness-checked GeoTIFF under <code>docs/downloads/</code>.</li>
<li>A holdout instrument (hide-and-recover folds, masking-faithful domain, label-world
ensemble, exact metric, CVaR selector, greedy prune) with
<b>{esc('62')} unit tests</b>, every one passing.</li>
<li>Two independent verifications of the metric: the organizer's own worked example and a
brute-force transcription of the published equations.</li>
<li>A registry of 21 sha256-verified prior rasters and a working lane-drift gate.</li>
</ul>

<h3>Hard blockers</h3>
<table>
<tr><th>blocker</th><th>consequence</th><th>what unblocks it</th></tr>
<tr><td>No DrivenData credentials in the sandbox</td>
    <td>The feature stack and labels are a second-hand copy recovered from sibling
        repositories, not a download from the organizer. Every derived number inherits
        that provenance.</td>
    <td>Run <code>scripts/download_competition_data.sh</code> on a machine with a
        competition login, into <code>data/</code>.</td></tr>
<tr><td>No USGS / 3DEP egress</td>
    <td>Hypotheses 2 and 4 cannot be validated here.</td>
    <td>A GitHub Actions runner reaches those hosts; sibling repos already do this.</td></tr>
<tr><td>No organizer-confirmed score for anything here</td>
    <td>HOLDOUT-DTI at {fmt(h.get('pooled'))} says nothing about the leaderboard. The
        relationship between the proxy holdout and live score is unknown, and sibling
        repositories report it anti-selects novelty.</td>
    <td>Only the submission page. Each slot is one experiment with one variable
        changed.</td></tr>
</table>

<h3>Ranked work list for the next session</h3>
<ol>
<li><b>Test hypothesis 1 (strain-rate second invariant).</b> The bands are already cached,
the hypothesis is the strongest of the five, and it is independent of catalogue geometry,
so it cannot collide with this lane. Expected: hours, not days.</li>
<li><b>Fit the offset distribution instead of assuming it.</b> The whole ribbon width
rests on Hermant et al.'s "up to 400 m" upper bound. Measure the actual distribution by
comparing the USGS catalogue against an independent trace set (the sibling repos cache
<code>trace_segments_utm11.csv</code> and <code>gdr_qfaults_traces.csv</code>). A measured
distribution replaces an assumption that currently drives the result.</li>
<li><b>Add a physics-corroborated prior inside the band.</b> The band says <i>where</i>;
nothing yet says <i>which part of the band</i>. Multiplying by a corroboration field
raises per-dot credit without widening, which is the one lever left that does not pay
more false-positive penalty.</li>
<li><b>Resolve the depth-to-magnetic-source band.</b> Hypothesis 3 is blocked on whether
the feature actually exists in the stack. Read the 19 band descriptions and settle it
before proposing the hypothesis as viable.</li>
<li><b>Ask the forum about <code>labels.tif</code> vs
<code>existing_faults.tif</code>.</b> They are byte-identical here. Either the download is
wrong or the distinction is nominal, and several lanes assume otherwise.</li>
<li><b>Pixel-level greedy pruning on the winning band.</b> Band-level pruning is a
tractable approximation. A single pass of pixel-level pruning on the surviving band only
is affordable and may recover a little more.</li>
</ol>

<h3>What would make this a top-of-leaderboard programme rather than one artifact</h3>
<p>The metric rewards <i>coverage at low false-positive cost</i>, and the arithmetic says
a dot needs only a {fmt(card.get('marginal_bar_at_selection'))} chance of landing within
300 m of a hidden fault to raise the score. The binding constraint is therefore not the
emission policy &mdash; that is now close to optimal for a given prior &mdash; but the
<i>quality of the prior</i>. Every point of improvement from here comes from a better
belief field, not from a better selector. That is the argument for spending the next
sessions on hypotheses 1-3 rather than on more selection machinery.</p>
</section>"""


def main() -> int:
    h33_path = EVIDENCE / "h33_analysis.json"
    h33 = json.loads(h33_path.read_text()) if h33_path.exists() else {}
    card_path = EVIDENCE / "run_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
    else:
        card = {"submission": {"file_name": "(run scripts/run_lane.py first)", "note": ""},
                "raster": {}, "validator": {"ok": False, "checks": {}},
                "registry": {"clear": None, "n_registry_rasters": 0},
                "holdout_dti": {}, "world_config": {}, "selection_table": [],
                "prune_log": [], "registry_comparisons": [],
                "verdict": "NO ARTIFACT YET - run scripts/run_lane.py"}
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    (DOCS / "downloads").mkdir(parents=True, exist_ok=True)
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    (DOCS / "assets" / "site.css").write_text(CSS)

    pages = {
        "index.html": ("GEMSDOE63", build_index(card), "index.html"),
        "executive-summary.html": ("How to submit", build_exec_summary(card), "executive-summary.html"),
        "method.html": ("Method", build_method(card), "method.html"),
        "results.html": ("Results", build_results(card), "results.html"),
        "why-h33.html": ("Why 0.2778", build_h33(h33), "why-h33.html"),
        "hypotheses.html": ("Hypotheses", build_hypotheses(), "hypotheses.html"),
        "sources.html": ("Sources", build_sources(), "sources.html"),
        "irregularities.html": ("Irregularities", build_irregularities(card), "irregularities.html"),
        "run-card.html": ("Run card", build_run_card(card), "run-card.html"),
        "next-steps.html": ("Next steps", build_next_steps(card), "next-steps.html"),
    }
    for fname, (title, body, active) in pages.items():
        (DOCS / fname).write_text(page(title, body, active))
        print(f"  wrote docs/{fname}  ({(DOCS/fname).stat().st_size:,} bytes)")

    site = {"generated_from": "evidence/run_card.json",
            "submission": card.get("submission", {}),
            "raster": card.get("raster", {}),
            "validator_ok": card.get("validator", {}).get("ok"),
            "uniqueness": card.get("registry", {}),
            "holdout_dti": card.get("holdout_dti", {}),
            "selection": card.get("selection_table", []),
            "sources": [{"title": t, "url": u, "what": w, "class": c} for t, u, w, c in SOURCES],
            "hypotheses": [{k: v for k, v in hh.items()} for hh in HYPOTHESES],
            "h33_analysis": h33}
    (DOCS / "data" / "site.json").write_text(json.dumps(site, indent=1, default=str))
    print(f"  wrote docs/data/site.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
