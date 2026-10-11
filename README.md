# GEMSDOE63 — Lane 63: CVaR robust-selection for the DOE GEMS Prize

> **Status (2026-10-11):** lane built and validated; one CVaR-pruned raster
> produced. **Verdict: NEGATIVE (duplicate logged)** — the registry uniqueness
> gate fails closed (see below). The raster is downloadable for
> learning/education but is **NOT cleared for submission**. See
> **docs/index.html** for the badge and the full stratified evidence. Read
> this README at the start of every session.

## What this repo does

The DOE GEMS Prize (DrivenData competition 306) asks for a GeoTIFF of
predicted *new* (unmapped) fault location in the northwestern Great Basin —
faults matter because they are the plumbing of geothermal systems. The metric
is a distance-weighted Tversky index (alpha 0.2, beta 0.8, 300 m triangular
kernel): false negatives cost 4x a false positive, and credit decays linearly
to zero at 300 m.

**Lane 63's method (robust selection):** Phase-2 truth will differ from
anything we validated on (organizers may add/remove/change faults; published
catalogue offsets reach ~400 m). So instead of maximizing one holdout score,
we build an ensemble of *label worlds* on the hide-and-recover holdout —
shift withheld segments by up to ±4 px, drop 10–30 % of them, add decoys at
the observed density — score candidate emission policies in every world with
the exact metric, select by mean DTI penalized by the worst-tail average
(**CVaR**, Rockafellar & Uryasev, *Journal of Risk* 2000), then greedily
prune the dot whose removal most improves CVaR. Output: one validated,
uniqueness-checked GeoTIFF.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py e1   # canaries + baseline holdout
PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py e2   # worlds + CVaR policy selection
PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py e3   # greedy CVaR prune + ship
PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py unique  # registry uniqueness scan
.venv/bin/python scripts/build_site63.py                                     # rebuild the docs site
```

## Results this session (labels per protocol rule 3)

| Measurement | Value | Label |
|---|---|---|
| Baseline ring-lane d2.8 ridge, pooled hide-and-recover | DTI 0.3733, CI95 [0.3520, 0.3923], 11,321 withheld px | HOLDOUT-DTI, gems57-pooled-hide-v2 |
| Canary AUC (distance-to-visible-catalogue) | 0.620 | CANARY-AUC (bar 0.90) |
| Canary AUC (radiometric K/Th ratio) | 0.472 | CANARY-AUC (negative result) |
| CVaR policy winner | dotted-d3.2 (mean world DTI 0.2830) | WORLD-DTI, 20 worlds |
| Pruning effect on worlds | mean 0.2826→0.2935, CVaR20 0.2726→0.2831 | WORLD-DTI, 20 worlds |
| Pruning effect on static buffered holdout | 0.3762→0.3622 (paired Δ −0.0141, CI [−0.0243,−0.0056], 11,321 withheld px) | HOLDOUT-DTI, gems57-pooled-hide-v2 |
| Registry uniqueness (730 rasters) | worst rho 0.340 (limit 0.90) PASS; forward overlap 1.000 (limit 0.70) FAIL — 27 universal soft surfaces + 21 dot-like rasters | REGISTRY-MEASUREMENT, fail-closed |
| **Verdict** | **negative — duplicate logged, not cleared for a slot** | protocol rule 1 |
| Leaderboard top cited in the prompt | 0.3774 / 0.3195 | USER-REPORTED-NOT-ORGANIZER-CONFIRMED |

## Limitations & what is needed next

1. **No DrivenData auth in this environment** — training_features.tif (419 MB),
   labels and sample_submission come from owner-mirrored, hash-pinned GitHub
   blobs (see registry/data_manifest.json). Mirrors are integrity-pinned, NOT
   organizer-authenticated.
2. **Leaderboard cannot be read from here** — every leaderboard value in this
   repo is USER-REPORTED. Organizer-confirmed scores require a manual receipt
   paste into evidence/.
3. **Registry uniqueness is owner-repo scoped** (730 prediction-like blobs
   today). Other teams' rasters are not public; the claim is scoped accordingly.
4. **Submission slots are NOT spent by this lane.** Promotion is a separate
   selector step within the weekly cap.
5. **The ring-lane surface is inside the dotted-ridge family lane** (21 dot-like
   registry rasters exceed the 70 % 3-px overlap gate). Any future lane-63
   candidate must be selected from a UNIQUENESS-GATED surface pool.
6. Next session, in order: (a) implement the H63-E stepover/strike-rotation
   surface and pass it through the same CVaR world selection + prune pipeline;
   (b) validate it on the holdout and the registry gate BEFORE any slot use;
   (c) wire H63-B/C (magnetic/gravity channels) once training_features.tif is
   restored from the pinned GEMSDOE parts; (d) optionally continue pruning past
   the pre-registered 3000-removal cap with a fresh calibration decision.

## The prompt (verbatim, session 2026-10-11)

<details>
<summary>Click to expand the full session prompt</summary>

```text
Review the repo. 

THE FOLLOWING IS THE HIGHEST URGENCY AND MUST BE FOLLOWED!

MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION.  DO NOT COPY A PREVIOUS SUBMISSION UNLESS IT'S FOR LEARNING AND EDUCATION.  BUT WE MUST GENERATE A UNIQUE TIF SUBMISSION.  IT MUST BE OBVIOUS WHETHER IT IS OK TO DOWNLOAD AND SUBMIT THE GENERATED TIF SUBMISSION.

There should be an easy to download submission tif file as described by the prompt.  Read the entire prompt.

Robust-selection lane: choose and shape the emission by CVaR across plausible label worlds. The rules force one submission chosen before private scores are known (§3.6.2). Organizers say Phase 2 faults can be added, removed or changed, and Hermant et al. measured catalogue offsets up to 400 m, so the truth you're scored on will differ from the truth you validated on. Build a label-world ensemble on the hide-and-recover holdout: shift withheld segments by up to ±4 px, drop 10–30% of them, and add decoy segments at the observed density. Score candidate rasters or emission policies (spacing, ribbon width, budget) in every world with the exact metric. Select by mean DTI penalized by the worst-tail average, i.e. conditional value-at-risk (Rockafellar and Uryasev, Journal of Risk, 2000), rather than by a single holdout score. Then prune greedily, dropping the dot or segment whose removal most improves CVaR, to produce a new raster. The result is only as honest as the world generator, so calibrate shift and omission rates from measured offsets and never tune against the public leaderboard. Output the standard validated GeoTIFF, uniqueness-checked against every earlier raster.

PARALLEL-RUN PROTOCOL — read first. This session is one of several running from this same prompt.

1. LANE. Your lane is the single method paragraph below. Stay inside it. If your raster's rank-correlation with any registry raster exceeds [0.90], or more than [70%] of your dots fall within 3 px of one registry raster's dots, you have drifted into another lane: log it as a duplicate and stop. Check this on the surface before placement AND on the final dots.

2. REUSE, DON'T REBUILD. Use the template's cached feature stack, evaluate_holdout.py and submission_writer.py. Holdout = hide-and-recover: withhold whole fault segments with a buffer, derive every catalogue-based feature only from the visible faults, mask visible faults pixel-exactly, score pooled DTI (alpha 0.2, beta 0.8, 300 m triangular kernel). If a shared tool is wrong, fix it once in the template and report it; never keep a private fork.

3. LABEL EVERY NUMBER as HOLDOUT-DTI (evaluator version, number of withheld positives, 95% CI) or ORGANIZER-CONFIRMED (copied from a submission-page receipt). A projection is never written as a score.

4. LEAKAGE CANARY. Test each feature alone on the holdout before trusting any result. AUC above [0.90] means leakage until proven otherwise.

5. RUN CARD. End with one JSON card: hypothesis; mechanism; the named non-fault process that could mimic it; holdout DTI + CI; correlation/overlap vs registry; raster sha256; validator output (no NaN inside the footprint, values in [0,1], CRS/shape/transform match); submission name + note of at most 140 characters; verdict promote / negative. Negative results are deliverables.

6. BUDGET. Stop after [3] experiments or [2] hours. Do not pick submissions: promotion to a real slot is a separate selector step, within the weekly cap shown on the submission page.

The following sites should serve as a starting point for understanding how to generate TIF submissions.  These websites are researched, and tested and have generated TIF submissions.  But we need to generate high scoring submissions.

Here are the results from submissions into the competition, separated by ....:

WE NEED TO STUDY, ANALYZE, AND UNDERSTAND THE HIGHEST SCORE FROM THE GEMDOE SITE WHERE THE SUBMISSION TIF IS DOWNLOADED FROM WHICH IS THE FOLLOWING:

https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html

h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros: 0.2778

Why and how did this get the highest score and are we able to generate a submission that scores higher than 0.2778?

Answer the question using Phd level experience, knowledge, and judgement. Then use the answer to generate a unique TIF submission into the competition.  Must be unique submission unlike any within the GEMSDOE sites above.  Verify working line by line no hallucinations.

Current competition leaderboard GEMSDOE high score:

0.3774	

https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html

gems-submission-20260925T001403Z-7f00890a: 0.1563

[... full leaderboard list omitted here for brevity; see docs/data/prompt_leaderboard.json for the machine-readable copy ...]

The following is the leaderboard for the competition:

https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/

See below for more links and information related to the competition:

https://github.com/drivendataorg/gems-prize-reference-solution

https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and

https://gbcge.org/current-projects/ingenious/

https://epsg.io/32611

https://en.wikipedia.org/wiki/Tversky_index

We need to quickly look at the results and results from the GEMSDOE websites above.

Before implementing, generate 3–5 candidate geological hypotheses we haven't tried yet, each naming: the specific layer(s) involved, the physical signature being targeted (e.g., an edge-detection or curvature transform), why it should catch a fault missing from the USGS/INGENIOUS catalogue rather than one already in it, and how it differs from anything already implemented in this repo. Rank them by expected DTI improvement and implementation cost. Validate the top candidate on our spatially-blocked holdout set before touching a weekly submission slot — do not spend a submission slot on an idea that hasn't beaten the current holdout best. If a candidate can't be validated without new external data, name the specific free, official source needed and check it's obtainable before proposing the idea as viable.

Work line by line verifying from official verified trusted sources, provide links for manual review.  There should be no manual input, work on your own to complete tasks.  Flag any irregularities for review.  No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements.  No hallucinations.  Verify line by line.

We have a good understanding of how our hypothesis, methodology, calculations, analysis are done so we should be able to figure out a way to score higher on the leaderboard using previous results and scoring that we have across the sites listed above.  We need to come up with distinct and unique strategies to score higher in this competition leaderboard.  We need to start doing heavy and deep research into the part of the project that matters the most, which is the scientific discovery of geothermal vents.  We should store all of our information and knowledge that we can gather from official verified sources.  This will serve as a starting point for other projects as well.  We need to think outside the box but still be grounded in proper scientific research, we are ultimately aiming for a top prize that many others are competing for.  So it's important to be contrarian but be smart about it.  We need to find sources of data that others are over looking or areas of the project when it comes to geothermal vents.  We need to do deep research and critical thinking and come up with new hypothesis to test.

0.3195	is the highest score right now so we need to design a new strategy, research, testing, analyzing, and generating submission system than the current website.  It should be unique, take unique approaches to generating a submission that can score higher than 0.3195.

Put this prompt into the repo readme and read it everytime we work on the project as a starting point to make sure we are building what we are aiming for and have a strong base to continue building and improving on making something useful for everyday use.  It should solve the problem of having to manually check everything ourselves and having an up to date current feed.

[... session context: Core Values (Maximize P(Win), Own the Outcome), site requirements (obvious download, executive summary subpage, unique name+note), data links (DrivenData login-walled; Dropbox mirrors; GDR 1391; EPSG:32611), the "single remaining blocker" data-placement note, three-pass verification instruction, PR+merge instruction ...]
```

</details>

## Core values guiding this repo

- **Maximize P(Win):** every decision (selection rule, export mode, registry
  scope) is made to maximize the probability that the programme succeeds; we
  report negative results honestly instead of dressing them up.
- **Own the Outcome:** shared tools were fixed upstream (grid.py root
  resolution), the submission is validated end-to-end locally, and every
  claim carries an evidence file and an evidence class.

## Key official sources (manual review)

| Source | URL |
|---|---|
| Competition overview | https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ |
| Competition about | https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/ |
| Data page (login) | https://www.drivendata.org/competitions/306/competition-doe-gems/data/ |
| Leaderboard | https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/ |
| Reference solution | https://github.com/drivendataorg/gems-prize-reference-solution |
| GeoDAWN survey (USGS) | https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and |
| INGENIOUS project | https://gbcge.org/current-projects/ingenious/ |
| CRS EPSG:32611 | https://epsg.io/32611 |
| Tversky index | https://en.wikipedia.org/wiki/Tversky_index |
| GDR submission 1391 | https://gdr.openei.org/submissions/1391 |
