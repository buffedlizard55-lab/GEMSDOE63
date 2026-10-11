# The session prompt, verbatim

Reproduced so the aim, the constraints and the protocol never have to be reconstructed
from memory. `PROMPT.md` is the distilled standing brief; this is the source text.

---

## The lane

> Robust-selection lane: choose and shape the emission by CVaR across plausible label
> worlds. The rules force one submission chosen before private scores are known
> (§3.6.2). Organizers say Phase 2 faults can be added, removed or changed, and Hermant
> et al. measured catalogue offsets up to 400 m, so the truth you're scored on will
> differ from the truth you validated on. Build a label-world ensemble on the
> hide-and-recover holdout: shift withheld segments by up to ±4 px, drop 10–30% of them,
> and add decoy segments at the observed density. Score candidate rasters or emission
> policies (spacing, ribbon width, budget) in every world with the exact metric. Select
> by mean DTI penalized by the worst-tail average, i.e. conditional value-at-risk
> (Rockafellar and Uryasev, Journal of Risk, 2000), rather than by a single holdout
> score. Then prune greedily, dropping the dot or segment whose removal most improves
> CVaR, to produce a new raster. The result is only as honest as the world generator, so
> calibrate shift and omission rates from measured offsets and never tune against the
> public leaderboard. Output the standard validated GeoTIFF, uniqueness-checked against
> every earlier raster.

## Parallel-run protocol — read first

1. **LANE.** Your lane is the single method paragraph above. Stay inside it. If your
   raster's rank-correlation with any registry raster exceeds [0.90], or more than [70%]
   of your dots fall within 3 px of one registry raster's dots, you have drifted into
   another lane: log it as a duplicate and stop. Check this on the surface before
   placement AND on the final dots.
2. **REUSE, DON'T REBUILD.** Use the template's cached feature stack,
   `evaluate_holdout.py` and `submission_writer.py`. Holdout = hide-and-recover: withhold
   whole fault segments with a buffer, derive every catalogue-based feature only from the
   visible faults, mask visible faults pixel-exactly, score pooled DTI (alpha 0.2,
   beta 0.8, 300 m triangular kernel). If a shared tool is wrong, fix it once in the
   template and report it; never keep a private fork.
3. **LABEL EVERY NUMBER** as HOLDOUT-DTI (evaluator version, number of withheld
   positives, 95% CI) or ORGANIZER-CONFIRMED (copied from a submission-page receipt). A
   projection is never written as a score.
4. **LEAKAGE CANARY.** Test each feature alone on the holdout before trusting any result.
   AUC above [0.90] means leakage until proven otherwise.
5. **RUN CARD.** End with one JSON card: hypothesis; mechanism; the named non-fault
   process that could mimic it; holdout DTI + CI; correlation/overlap vs registry; raster
   sha256; validator output (no NaN inside the footprint, values in [0, 1],
   CRS/shape/transform match); submission name + note of at most 140 characters; verdict
   promote / negative. Negative results are deliverables.
6. **BUDGET.** Stop after [3] experiments or [2] hours. Do not pick submissions:
   promotion to a real slot is a separate selector step, within the weekly cap shown on
   the submission page.

## Highest urgency

> MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. DO NOT COPY A PREVIOUS
> SUBMISSION UNLESS IT'S FOR LEARNING AND EDUCATION. BUT WE MUST GENERATE A UNIQUE TIF
> SUBMISSION. IT MUST BE OBVIOUS WHETHER IT IS OK TO DOWNLOAD AND SUBMIT THE GENERATED
> TIF SUBMISSION.
>
> There should be an easy to download submission tif file as described by the prompt.
> Read the entire prompt.

## The question this session had to answer

> We need to study, analyze, and understand the highest score from the GEMDOE site where
> the submission TIF is downloaded from … `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros:
> 0.2778`. Why and how did this get the highest score and are we able to generate a
> submission that scores higher than 0.2778?

The answer is on the [results page](results.html) and in
[`why-h33-scored.md`](why-h33-scored.md).

## Standing brief

- Generate 3–5 candidate geological hypotheses we have not tried yet, each naming the
  specific layer(s) involved, the physical signature being targeted, why it should catch
  a fault missing from the USGS/INGENIOUS catalogue rather than one already in it, and
  how it differs from anything already implemented in this repo. Rank them by expected
  DTI improvement and implementation cost. Validate the top candidate on the
  spatially-blocked holdout set before touching a weekly submission slot. If a candidate
  can't be validated without new external data, name the specific free, official source
  needed and check it's obtainable before proposing the idea as viable.
- Work line by line verifying from official verified trusted sources, provide links for
  manual review. There should be no manual input. Flag any irregularities for review. No
  hallucinations.
- The site must generate a TIF file required for submission, as easy as clicking a file to
  download. This needs to be in the executive summary or the very beginning of the site.
  It should be obvious when you visit the site.
- A previous download was rejected by the submission form with **"Predicted values must
  be in range [0, 1]"**.
- Give the submission a unique name and a short comment to help tell submissions apart
  later, e.g. `clustering with k=25`.
- Create an executive summary subpage that explains exactly how to make a submission
  into the contest.
- Create a GitHub page for this repo with clean UI, user friendly, simple and easy to
  use, organized and clean, with all relevant information in an easy-to-read format and
  official verified links as sources for review.
- Keep the Arena core values — **Maximize P(Win)** and **Own the Outcome** — as the focal
  point when building, developing, researching, suggesting upgrades and implementing.

## Known blocker stated in the prompt

> ❌ No DrivenData auth → cannot auto-download `training_features.tif`, `labels.tif`,
> `sample_submission.tif`, `1m_DEM_links.csv` from the competition data page (verified
> redirect to login).

This is still true in this sandbox. See
[`../data/README.md`](../data/README.md) for what was used instead and why that matters.

## Multi-pass instruction

> Pass 1: Implement the task completely and verify the result.
> Pass 2: Review your work for bugs, missing requirements, incorrect assumptions, and
> edge cases. Fix everything you find.
> Pass 3: Re-check the entire implementation against the original request. Improve
> accuracy, reliability, completeness, and code quality. Fix any remaining issues.
