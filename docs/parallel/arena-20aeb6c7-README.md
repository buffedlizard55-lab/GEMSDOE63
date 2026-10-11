# GEMSDOE63 — DOE GEMS Prize (fault prediction) submission workbench

> **Read this first, every session.** This file holds the standing brief for the project. Re-read it before any work,
> keep the Core Values and the protocol below in view, and update the status block when a number changes.

## 1. Status (2026-10-11, UTC)

| Item | Status |
|---|---|
| Submission GeoTIFF | **Built, format-valid, and ON HOLD.** Uniqueness gate returned DUPLICATE (IR-17), so it is not published for download. File: `held/gems63-cvar-ridge-q020-s2-r3-20261011T212315Z.tif` |
| May I download it? | **No (not cleared).** May I submit it? **No (not endorsed).** |
| Recommendation to spend a weekly slot on it | **NO (verdict: negative).** Its holdout DTI is about 1.2× random dots (see §6). |
| Organiser score for this file | **None.** It has not been submitted. No ORGANIZER-CONFIRMED number exists. |
| Site (executive summary first) | [`index.html`](index.html) (root, because Pages serves `main /`); research page [`research.html`](research.html) |
| Irregularities for review | [`registry/irregularities.md`](registry/irregularities.md) (16 items, HIGH items first) |
| Hypotheses (ranked, with the non-fault mimic for each) | [`registry/hypotheses.md`](registry/hypotheses.md) |
| Run card (JSON) | [`registry/run_card.json`](registry/run_card.json) |

**Bottom line.** The file is format-valid, but the uniqueness gate tripped, so it is on hold and not to be downloaded. Even if cleared, it is not a competitive submission. Our own evidence says it is not a competitive submission:
the catalogue-free emission scores about the same as random dots on the spatially blocked holdout. Do not treat this file
as a winning entry. The next gains need a different signal, not a re-tuned version of this one (§8).

## 2. Standing brief (core instructions, kept verbatim where short)

**Core values** (as given in the brief): **Maximize P(Win)** — weigh tradeoffs and risk and choose the path with the highest chance of winning.
**Own the Outcome** — own results end to end and treat failure as a signal to improve.

**Standing requirements.** Work line by line from official, verifiable sources, with links for manual review. No manual input
(the agent works on its own). Flag irregularities for review. No hallucinations: every number carries an evidence label (§5). Never tune
against the public leaderboard. Keep large data and competition rasters out of git.

**The goal.** Place top of the DrivenData GEMS Prize leaderboard
(https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) by producing a single, validated GeoTIFF
that scores well on the private test set. The official board (read 2026-10-11) puts #1 at 0.4133 and the 0.3195 score at #9 (`registry/official_leaderboard_snapshot_2026-10-11.csv`).
The brief's "0.3774" and "0.3195 is the highest" are not what the official board shows (IR-02).

**Submission format** (official problem page, https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/): one single-band float32 GeoTIFF,
EPSG:32611, 100 m, same grid as the training data (3730 rows × 3292 columns, `registry/grid_facts.json`), values in [0, 1], NaN outside the bounds.
Rules (https://docs.nlr.gov/docs/fy26osti/96647.pdf): three submissions per week; one final submission; generative-AI disclosure in §3.2.

**Scoring.** Distance-weighted Tversky index, α = 0.2 (false positives), β = 0.8 (false negatives), triangular kernel R = 3 px (300 m) (official problem page). Known USGS/INGENIOUS fault pixels are
masked: organiser staff confirmed this on 2026-09-16 (https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516).

****Robust-selection lane (this session).** Choose the emission by CVaR across simulated label worlds, because the true labels may differ
from the catalogue. Worlds (40 per fold): segments shifted by up to ±4 px, 10–30% dropped, decoys added at 0.5–1.0× the withheld truth.
Selection: J = ½·mean DTI + ½·CVaR₀.₂ over the worlds (Rockafellar & Uryasev 2000, the CVaR reference; the journal volume/page is disputed — IR-10).
Shift calibration: Hermant et al. (2025) report USGS-vs-TLS fault-label distances "up to 400 m", for north-central Nevada only
(https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2025/Hermant.pdf). This is a local figure, not a global one (IR-09).

**Parallel-run protocol (summary; the lane is "robust selection").**
1. Lane discipline: if this raster's rank correlation with any registry raster is > 0.90, or > 70% of its dots fall within 3 px of one registry raster's dots, log it as a duplicate and stop.
2. Reuse the template tools (feature stack, holdout, writer). Any shared-tool bug is fixed once and reported.
3. Label every number **HOLDOUT-DTI** (evaluator version, withheld positives, 95% CI) or **ORGANIZER-CONFIRMED** (copied from a submission-page receipt). A projection is never written as a score.
4. Leakage canary: each feature alone on the holdout; AUC > 0.90 means leakage until proven otherwise.
5. End with one run card (§7).
6. Budget: stop after 3 experiments or 2 hours. Promotion to a weekly slot is a separate selector step.

**Generative-AI disclosure** (rules §3.2). This repo, its analysis and its prose were produced with AI-agent assistance. Every number was recomputed by the scripts
here, and the owner must review and sign the disclosure before any final submission.

## 3. How to submit (once a file is cleared)

Nothing is cleared today. For a future file: (1) confirm the uniqueness gate passed and the file is in `docs/downloads/`; (2) on the DrivenData submission page
choose *New submission*, upload the `.tif`; (3) enter the name and a short note from its manifest in `registry/submissions/`; (4) if the site reports
"Predicted values must be in range [0, 1]", the file is wrong, so stop and check the validator output.
The held file's name is `gems63-cvar-ridge-q020-s2-r3-20261011T212315Z` and its note is in its manifest.

## 4. Data, provenance, reproduction

Competition inputs are **not** committed. The sandbox cannot reach DrivenData, so inputs come from public GitHub mirrors
at fixed commits, as pinned by the owner's GEMSDOE32 manifest. Every hash is checked. Pins: [`registry/data_pins.json`](registry/data_pins.json). See IR-04.

```bash
python3 -m venv ~/.venv-gems && ~/.venv-gems/bin/pip install numpy scipy rasterio pyproj scikit-learn scikit-image pytest pyflakes
export GEMS_DATA_DIR=/tmp/gems_data            # outside the repo
~/.venv-gems/bin/python scripts/fetch_data.py   # verifies SHA-256 of labels, sample, features
~/.venv-gems/bin/python scripts/inspect_inputs.py
PYTHONPATH=. ~/.venv-gems/bin/python -m pytest -q tests
PYTHONPATH=. ~/.venv-gems/bin/python scripts/e1_leakage_canary.py      # E1
PYTHONPATH=. ~/.venv-gems/bin/python scripts/e2_cvar_select.py         # E2 (~3 min)
PYTHONPATH=. ~/.venv-gems/bin/python scripts/controls.py               # controls
PYTHONPATH=. ~/.venv-gems/bin/python scripts/build_submission.py       # E3 (writes docs/downloads/; the held file was then moved to held/ by hand)
PYTHONPATH=. ~/.venv-gems/bin/python scripts/uniqueness_check.py docs/downloads/<name>.tif   # needs /tmp/regs (see below)
```
Registry rasters for the uniqueness test come from the 62 GEMSDOE repos in `/tmp/regs/repos.txt` that contain TIFs (1,458 files at HEAD; see IR-13).

Repo layout: `gems63/` (metric, grid, features, worlds, emission, validator), `scripts/` (experiments and build), `tests/`,
`registry/` (ledger, pins, experiments, submissions, run card, irregularities, hypotheses), `docs/` (site + downloads).

## 5. Evidence labels used everywhere

| Label | Meaning | Where |
|---|---|---|
| **HOLDOUT-DTI** | Our exact DTI on the spatially blocked catalogue holdout (evaluator `gems63.dti`, pooled over 5 folds, n withheld positives and 95% CI given). Not an organiser score. | E2, E3 |
| **ORGANIZER-CONFIRMED** | Copied from a submission-page receipt. **None exists for this work.** | — |
| **OFFICIAL-SNAPSHOT** | Public leaderboard read on a stated date. Public, not private. | `registry/official_leaderboard_snapshot_2026-10-11.csv` |
| **USER-REPORTED** | Copied from the brief or from an owner page; no receipt. | `registry/ledger.csv` |
| **PROJECTION** | Reasoned expectation, never a score. | `registry/hypotheses.md` |
| **CONTROL** | Pipeline check (random or oracle), not a competition quantity. | `registry/experiments/controls.json` |

## 6. Results (every number labelled)

**E1 — leakage canary** (`registry/experiments/e1_canary.json`). Each of the 24 catalogue-free features tested alone (single-feature AUC on the
holdout). **PASS:** none exceeds 0.90. The top are `geod_shear` (AUC 0.852) and `geod_2ndinv` (0.849), with per-fold swings of about 0.15–0.77.
Both are close to the threshold and are flagged for review. Withheld positives: 60,988.
*Evidence: canary AUC (not DTI).*

**E2 — CVaR robust selection** (`registry/experiments/e2_cvar.json`, 142.8 s). 40 label worlds per fold, pooled over 5 folds; J = ½·mean + ½·CVaR₀.₂.
Search (script docstring): stage 1 grids family (ridge, logit) × q (top fraction of active pixels kept before NMS; grid 0.5%, 1%, 2%) × s (NMS half-window in px; grid 2, 3, 4) with r = 0; stage 2 greedily moves r, then neighbouring q and s. 24 policies were scored in total. Top rows (all HOLDOUT-DTI):

| Policy | Dots | Mean world DTI | CVaR₀.₂ | J | Nominal HOLDOUT-DTI |
|---|---:|---:|---:|---:|---:|
| **chosen: ridge q=2%, s=2, r=3** | 57,435 | 0.0249 | 0.0234 | 0.0242 | **0.0258** (95% CI 0.0158–0.0356; n = 60,988 positives) |
| ridge q=2%, s=2, r=4 | 56,129 | 0.0249 | 0.0234 | 0.0241 | 0.0258 |
| ridge q=2%, s=2, r=0 | 62,437 | 0.0246 | 0.0232 | 0.0239 | 0.0254 |
| logit q=2%, s=2, r=0 | 29,399 | 0.0199 | 0.0184 | 0.0192 | 0.0341 (fragile under worlds; rejected) |

The chosen point is at the top of the q-grid (q = 2%); a larger q was not tried in this session (§8).
Note: the file's "chosen" block contains only the 95% CI for the chosen policy; other rows have no CI.
**CONTROLS** (`registry/experiments/controls.json`, not competition scores):
- **Random dots, same count:** HOLDOUT-DTI **0.0208** (5 seeds). The chosen policy (0.0258) is only about 1.2× chance.
- **Oracle** (dots on a random 50% of withheld truth): HOLDOUT-DTI **0.796**. The scoring machinery behaves.

**Verdict on the lane:** negative. The ridge (1.2× the random-dot control) carries very little information about the withheld faults.
The logistic model has a higher nominal score (0.0341) but falls to a world mean of 0.0199 under the label worlds, so it is rejected.
Under the protocol, no holdout win means no slot.

**E3 — the held file** (`registry/submissions/gems63-cvar-ridge-q020-s2-r3-20261011T212315Z.json`): 11,237 dots from the full catalogue, with a 3 px flank cut.
Validator (9 checks, `gems63/validate.py`): PASS. Checks cover single band, float32, EPSG:32611, shape/transform identical to `labels.tif`, NaN outside the footprint, values in [0, 1].
SHA-256: `73713d024eb45bc03d2f58977cb8526ef7c75b5df971849c9e90d4f7e5e57564`.

### 6a. Uniqueness (protocol item 1)
Result: `registry/uniqueness_check.json`. Verdict **DUPLICATE**, from the dot-overlap test only:
the candidate's dots are 100% within 3 px of the dots of `5GEMSDOE/docs/downloads/gems-density-probe-20260925T184215Z-9cdae9b4.tif`.
That registry raster has 5,106,385 "dot" cells (almost every footprint cell), so the overlap gate is saturated and is almost certainly an artefact (IR-17).
Pearson check: max 0.095 vs gate 0.90 (no duplicate). Exact SHA-256 matches: none. Compared 1,434 registry rasters; 1 placeholder missing (IR-18).
Protocol consequence: the lane stops for this candidate. The gate is **not** changed to get a pass.

## 7. Run card

See [`registry/run_card.json`](registry/run_card.json) (one card, protocol item 5).

## 8. Limitations and what is needed next

0. **Uniqueness gate tripped (blocking).** The file is on hold. The owner decides how to resolve IR-17 (fix the overlap-test definition for dense rasters, then re-run; this is a protocol decision, not a sandbox decision).
1. **Wrong proxy for the target (main limitation).** The holdout's truth is catalogue-type faults. The organisers say the private truth includes faults missing from
   the public USGS database (the rules' description of the task). We have no measurement of that distribution, so the holdout is a proxy and may not represent it.
2. **No organiser score.** Nothing here is submitted, so the sandbox cannot confirm any number. This needs a human submission and a receipt.
3. **Budget edge.** The chosen policy sits at the top of the q-grid (2%). A larger budget and a better score map are the obvious next experiment; this session did not run it.
4. **Missing signal.** The 1 m DEM (scarp detection, H5, ranked first by PROJECTION only) is blocked: its hosts are not on the sandbox egress allowlist (IR-16).
   The owner's derived scarp raster is not an official product (IR-16).
5. **Registry coverage.** Uniqueness is HEAD-only and covers the 62 repos listed (IR-13).
6. **Mirror provenance.** The inputs are third-party mirrors (IR-04). Confirm the rules allow their use, or download from DrivenData with an account.
7. **Next session, in order:** (a) the owner reviews the AI disclosure (IR-14); (b) get the 1 m DEM and test H5 on the same holdout, with the controls; (c) only then decide whether any raster is promoted.

## 9. Changelog

- 2026-10-11: uniqueness gate tripped (DUPLICATE via dot-overlap, IR-17); file moved to held/; site moved to root index.html.
- 2026-10-11: repo initialised with the holdout, CVaR and validator pipeline; E1–E3 run; controls; uniqueness check; irregularities register; ledger.

---

## 10. Prior status on `main` (PR #1, merged earlier today) — superseded

PR #1 (`arena/e03322eb-gemsdoe63`) put a different file on `main` and marked it **"OK TO DOWNLOAD AND SUBMIT"**. This section records the conflict, because the top-priority question is whether a file may be downloaded and submitted. **This branch's status (§1) governs.** The owner must resolve the items below.

| Item (PR #1) | Claim | Our check |
|---|---|---|
| File | `docs/downloads/gems63-cvar-robust-greedy-pruned-10290dots-20261011T220000Z-f49a21c0-zeros.tif` (SHA-256 `13e04439…`) | Not verified by us (not in our tree before merge). Not endorsed. |
| Holdout | "HOLDOUT-DTI 0.01013 (CI 0.00883–0.01041, 1,822 withheld segments, evaluator `gems63.holdout.v1`)" | Labelled HOLDOUT-DTI by its author. Our random-dot control on **our** holdout is 0.0208. The holdouts differ, so the numbers are **not directly comparable**. This must be checked on one common holdout before any "OK". |
| Outside-footprint values | "0.0 all-finite zeros (nodata=None)" — 12,279,160 finite cells | Official page and sample use NaN outside bounds (rules wording: "null or nan"). Zero-fill is a format risk (IR-06). |
| Uniqueness | "max Spearman 0.003; max 3-px overlap 0.293" | Our overlap test (IR-17) and comparator set differ. Not reconciled. |
| Duplicate code | `src/gems63/` (metric, uniqueness, submission_writer) duplicates our `gems63/` (dti, validate, grid, emit). | Violates "reuse, don't rebuild". Needs consolidation (IR-19). |
| Score claims | Cites 0.2778 (GEMSDOE32 `h33-2-b2`) as the comparator; no organiser receipt (IR-01). | Same as IR-01/IR-15. |

**What to do (owner decision, not made here):** (1) decide whether the PR #1 file stays downloadable; (2) run both holdouts on one common fold set; (3) choose one `gems63` package and one uniqueness gate. Until then, no file in this repo is cleared for submission.
