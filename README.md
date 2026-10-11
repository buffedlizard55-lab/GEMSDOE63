# GEMSDOE63 — robust-selection lane for the DOE GEMS Prize Challenge

**The submission file is at [`docs/downloads/`](docs/downloads/).** Open
**[`docs/index.html`](docs/index.html)** (or the
[GitHub Pages site](https://buffedlizard55-lab.github.io/GEMSDOE63/)) for the download
button, the validator verdict, and an explicit answer to *"is it OK to submit this?"*.

---

## Result of this session — read this before spending a slot

| | |
|---|---|
| **File** | [`docs/downloads/gems63-cvar-offset-band.tif`](docs/downloads/gems63-cvar-offset-band.tif) |
| **sha256** | `b8f15e4d5fee63348c00bb7fb5a92ed529be438fa3b0a70af933ba63fac25244` |
| **Format** | 13/13 validator checks pass — single band, float32, EPSG:32611, 100 m, exact geotransform, 0 NaN, values `{0, 1}` |
| **HOLDOUT-DTI** | 0.2336 pooled over 48 label worlds × 4 folds (mean per world 0.2466, CVaR worst-20 % 0.1248) |
| **Leakage canary** | PASS — mean AUC of distance-to-catalogue 0.6457 against a 0.90 gate |
| **Lane-drift gate** | **LITERALLY TRIPPED.** 100 % of dots within 3 px of `gems8_hedge_v2_01563` (gate 70 %) |
| **Verdict** | **DO NOT SPEND A SLOT.** Format-valid and submittable, but not recommended |

The lane-drift trip is recorded exactly as the brief specifies, and then diagnosed
rather than explained away: `gems8_hedge_v2` emits `p = 1.0` on **all 60,988** catalogue
pixels (measured here, corroborated by `8GEMSDOE/reports/hedge_v2.json`), and this
emission is the 1-px ring around that same catalogue — so every dot is **0 px** from a
hedge dot and the 70 % statistic is 1.000 *by construction*. It cannot discriminate.
The three other literal trips are dense-comparator artifacts (lift 1.01, 1.00, 1.35).
The discriminating statistic is clean: maximum Spearman rank correlation across all 21
registry rasters is **0.079** against a 0.90 gate.

The stronger reason not to submit is independent of the gate: this mechanism is already
occupied, and the sibling that occupies it scored **0.1563** (owner-reported) against the
group's best off-catalogue artifact at **0.2778**. See
[`docs/why-h33.html`](docs/why-h33.html) for the measured explanation — **94.2 % of that
artifact's dots are more than 3 px from any mapped fault**, and it scored highest.
Spend the next slot on H1 in [`docs/hypotheses.html`](docs/hypotheses.html) instead.

---

## What this repository is

A pipeline that produces **one single-band float32 GeoTIFF** for the
[DOE GEMS Prize Challenge](https://www.drivendata.org/competitions/306/competition-doe-gems/),
together with the evidence needed to decide whether that file is worth a submission slot.

The task: predict geological faults in the GeoDAWN region (northwestern Great Basin,
Nevada) that are **not** in the public USGS/INGENIOUS catalogue. Scored by a
distance-weighted Tversky index (alpha 0.2, beta 0.8, 300 m triangular kernel).

The lane is **robust selection**: rather than pick the emission that wins one holdout
draw, every candidate emission is scored across an **ensemble of plausible label worlds**
and chosen by mean DTI penalised by the worst-tail average — conditional value-at-risk
(Rockafellar & Uryasev, *Journal of Risk*, 2000) — then pruned greedily.

## The operating brief

**[`PROMPT.md`](PROMPT.md) is the standing brief and is read at the start of every
session.** The full session prompt is reproduced verbatim in
[`docs/prompt-verbatim.md`](docs/prompt-verbatim.md) so the aim, the constraints and the
protocol never have to be reconstructed from memory.

The binding points, in short:

1. **A unique TIF must be generated every session.** Never a copy of an earlier
   submission. It must be obvious whether it is OK to download and submit it.
2. **The download must be easy and above the fold.**
3. **Label every number** HOLDOUT-DTI or ORGANIZER-CONFIRMED. A projection is never
   written as a score.
4. **Leakage canary**: any feature with holdout AUC > 0.90 is leakage until proven
   otherwise.
5. **Lane-drift gate**: rank-correlation > 0.90, or > 70 % of dots within 3 px of any
   registry raster, means duplicate — log it and stop.
6. **End with a run card.** Negative results are deliverables.

## Repository layout

```
src/gems63/
  metric.py       distance-weighted Tversky index, exact, + a verified binary fast path
  holdout.py      hide-and-recover folds, masking-faithful domain, cluster bootstrap
  worlds.py       the label-world generator (shift / omit / decoy / phi)
  emission.py     candidate emission policies: ribbons, bands, thinning, continuations
  cvar.py         CVaR selection and greedy pruning
  validate.py     the format checks the submission form applies
  uniqueness.py   the lane-drift gate against every registry raster
  raster_io.py    read/write pinned to the organizer's own template
  auc.py          dependency-free AUC for the leakage canary
scripts/
  run_lane.py         the lane, end to end
  fetch_registry.py   re-download and sha256-verify every prior raster
  make_site.py        generate the site from evidence/run_card.json
tests/                62 tests, all passing
docs/                 the GitHub Pages site
registry/             prior-art manifest + verified cache
evidence/             the run card and receipts
```

## Run it

```bash
pip install rasterio numpy scipy pytest
python3 -m pytest tests/ -q               # 62 tests
python3 scripts/fetch_registry.py         # 21 prior rasters, sha256-verified
python3 scripts/run_lane.py --worlds 12 --name my-run
python3 scripts/make_site.py
```

The competition feature stack (419 MB, gitignored) is expected at
`data/cache/gems-geodawn-numerical-features.tif`. See
[`data/README.md`](data/README.md) for its provenance and the caveat that matters.

## The core values this project is built on

**Maximize P(Win).** Every decision weighs the trade-offs, assesses the risk, and picks
the path that maximises the probability the project wins — including the decision to
report a negative result rather than ship a hopeful one.

**Own the outcome.** Results are owned end to end, not slice by slice. When a shared tool
is wrong it is fixed once and reported, never forked privately. When a gate fails, the
artifact is marked *do not submit* rather than quietly offered.

## Claims this repository does not make

- No organizer-confirmed score exists for any file produced here. None is claimed.
- No holdout number is presented as a score, and no projection is written as one.
- No pixel emitted here is asserted to be a fault. The emission is a set of candidates
  with a named non-tectonic confuser recorded in the run card.
- The feature stack used here is a **second-hand copy**, not a download from the
  organizer. See [`data/README.md`](data/README.md) and
  [`docs/irregularities.html`](docs/irregularities.html).
