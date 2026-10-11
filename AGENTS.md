# Lane 63 agent protocol (parallel-run protocol, verbatim intent)

1. LANE. Robust selection: CVaR across label worlds on the hide-and-recover
   holdout. Stay inside it. Duplicate check vs registry: Spearman rho > 0.90
   or > 70 % of dots within 3 px of one registry raster = duplicate; log and stop.
2. REUSE. Shared template lives in vendor/template66 (GEMSDOE66 src/gems57 +
   src/gems, byte-pinned; see vendor/template66/TEMPLATE_SHA256.txt). Fix shared
   tools once, report them; never fork privately.
3. LABEL EVERY NUMBER. HOLDOUT-DTI (evaluator version, n withheld positives,
   95 % CI) or ORGANIZER-CONFIRMED (submission-page receipt). Leaderboard values
   cited in the prompt are USER-REPORTED-NOT-ORGANIZER-CONFIRMED.
4. LEAKAGE CANARY. AUC > 0.90 for any single feature on the holdout = leakage
   until proven otherwise (evidence/e1_canaries.json).
5. RUN CARD. One JSON card at evidence/run_card.json + docs/run-card.html.
6. BUDGET. 3 experiments or 2 hours. Do not spend submission slots; promotion
   is a separate selector step.

## Layout
- vendor/template66/  shared template (gems57 = evaluator/writer/uniqueness)
- src/gems63/         lane code: policies, worlds, cvar, canary
- scripts/run_lane63.py  e1 | e2 | e3 | unique
- evidence/           all measurements (JSON)
- submissions/        validated TIF + ZIP + receipt
- docs/               GitHub Pages site (build: scripts/build_site63.py)
- data/bridge/        pinned mirrors of labels/existing_faults/sample_submission
- data/registry_rasters/  730 prediction-like blobs for the uniqueness scan

## Reproduce
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py e1
    PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py e2
    PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py e3
    PYTHONPATH=vendor/template66:src .venv/bin/python scripts/run_lane63.py unique
    .venv/bin/python scripts/build_site63.py
