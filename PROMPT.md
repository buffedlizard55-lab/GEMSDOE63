# Operating brief — read this first, every session

This file is the standing brief for the `GEMSDOE63` line of work. It is read at the
start of every session and is the base we build on, so the aim, the constraints and
the protocol never have to be reconstructed from memory. It is reproduced verbatim
from the session prompt.

---

## Part 1 — The lane (this session's assignment)

> **Robust-selection lane: choose and shape the emission by CVaR across plausible
> label worlds.** The rules force one submission chosen before private scores are
> known (§3.6.2). Organizers say Phase 2 faults can be added, removed or changed, and
> Hermant et al. measured catalogue offsets up to 400 m, so the truth you're scored
> on will differ from the truth you validated on. Build a label-world ensemble on the
> hide-and-recover holdout: shift withheld segments by up to ±4 px, drop 10–30% of
> them, and add decoy segments at the observed density. Score candidate rasters or
> emission policies (spacing, ribbon width, budget) in every world with the exact
> metric. Select by mean DTI penalized by the worst-tail average, i.e. conditional
> value-at-risk (Rockafellar and Uryasev, *Journal of Risk*, 2000), rather than by a
> single holdout score. Then prune greedily, dropping the dot or segment whose removal
> most improves CVaR, to produce a new raster. The result is only as honest as the
> world generator, so calibrate shift and omission rates from measured offsets and
> never tune against the public leaderboard. Output the standard validated GeoTIFF,
> uniqueness-checked against every earlier raster.

## Part 2 — Parallel-run protocol (binding on every session)

1. **LANE.** Stay inside the single method paragraph above. If the raster's
   rank-correlation with any registry raster exceeds **0.90**, or more than **70 %**
   of the dots fall within 3 px of one registry raster's dots, the session has drifted
   into another lane: log it as a duplicate and stop. Check this on the surface before
   placement **and** on the final dots.
2. **REUSE, DON'T REBUILD.** Use the template's cached feature stack,
   `evaluate_holdout.py` and `submission_writer.py`. Holdout = hide-and-recover:
   withhold whole fault segments with a buffer, derive every catalogue-based feature
   only from the visible faults, mask visible faults pixel-exactly, score pooled DTI
   (alpha 0.2, beta 0.8, 300 m triangular kernel). If a shared tool is wrong, fix it
   once in the template and report it; never keep a private fork.
3. **LABEL EVERY NUMBER** as *HOLDOUT-DTI* (evaluator version, number of withheld
   positives, 95 % CI) or *ORGANIZER-CONFIRMED* (copied from a submission-page
   receipt). **A projection is never written as a score.**
4. **LEAKAGE CANARY.** Test each feature alone on the holdout before trusting any
   result. AUC above **0.90** means leakage until proven otherwise.
5. **RUN CARD.** End with one JSON card: hypothesis; mechanism; the named non-fault
   process that could mimic it; holdout DTI + CI; correlation/overlap vs registry;
   raster sha256; validator output (no NaN inside the footprint, values in [0, 1],
   CRS/shape/transform match); submission name + note of at most 140 characters;
   verdict **promote / negative**. **Negative results are deliverables.**
6. **BUDGET.** Stop after **3** experiments or **2** hours. Do not pick submissions:
   promotion to a real slot is a separate selector step, within the weekly cap shown
   on the submission page.

## Part 3 — Standing brief (applies to every session in this line)

* **Goal: place at the top of the leaderboard** for the
  [DOE GEMS Prize](https://www.drivendata.org/competitions/306/competition-doe-gems/).
  Understand the problem from the
  [overview and problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
  and the [about page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/).
* **A unique TIF submission must be generated every session.** Do not copy a previous
  submission unless it is for learning and education. It must be obvious whether it is
  OK to download and submit the generated TIF.
* **There must be an easy-to-download submission TIF**, surfaced at the very top of the
  site so it is obvious on arrival.
* **The submission form must accept the file.** A previous session's download was
  rejected with *"Predicted values must be in range [0, 1]"*. Every file this
  repository writes is validated against the organizer's own template before it is
  offered for download.
* **Hypotheses first.** Before implementing, generate 3–5 candidate geological
  hypotheses we have not tried yet, each naming the specific layer(s) involved, the
  physical signature targeted, why it should catch a fault *missing* from the
  USGS/INGENIOUS catalogue rather than one already in it, and how it differs from
  anything already implemented here. Rank by expected DTI improvement and
  implementation cost. Validate the top candidate on the spatially-blocked holdout
  **before** touching a weekly submission slot.
* **Work line by line from official verified trusted sources; provide links for manual
  review. No manual input. No hallucinations. Flag any irregularities.**
* **Own the outcome.** We own results end to end, treat failure and success as signals,
  and stay accountable for the final artifact — not just our slice of it.
* **Maximize P(Win).** In every decision weigh the trade-offs, assess the risk, and
  choose the path that maximises the probability the project wins.

## Part 4 — What "done" means for a session

A session is done when all of the following exist and are committed:

1. A **validated GeoTIFF** under `docs/downloads/` that a human can download and
   submit, with the validator's PASS recorded in the run card.
2. A **run card** (`evidence/run_card.json`) with every field the protocol asks for.
3. A **uniqueness gate** result against every registry raster.
4. A **site** (`docs/`) whose landing page puts the download and the "is it OK to
   submit this?" verdict above the fold.
5. A **pull request merged to `main`**.
6. An explicit list of **what was not verified and why**.
