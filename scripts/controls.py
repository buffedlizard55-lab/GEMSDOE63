#!/usr/bin/env python3
"""Pipeline controls (NOT experiments; they check that the holdout machinery behaves).
C1 random dots with the same per-fold count as the chosen policy (chance baseline)
C2 oracle: dots placed on a random 50% of withheld truth (should be high; checks geometry/scoring)
Outputs registry/experiments/controls.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems63_stack.dti import components_binary, dti_from_components
from gems63_stack.emit import ridge_score_vec, score_to_grid, policy_dots
from gems63_stack.features import build_feature_stack
from gems63_stack.grid import REPO, load_grid, N_FOLDS
from gems63_stack.worlds import FoldSegments, nominal_truth

pol = json.loads((REPO / "registry" / "submission_policy.json").read_text())["policy"]
rng = np.random.default_rng(7)
g = load_grid()
X, names = build_feature_stack(g.footprint)
ridge_grid = score_to_grid(ridge_score_vec(X, names), g.footprint)
del X
rand_tot, orc_tot = np.zeros(3), np.zeros(3)
rand_runs = []
for k in range(N_FOLDS):
    fs = FoldSegments(g, k)
    nom = nominal_truth(fs)
    dots = policy_dots(ridge_grid, g, fs.visible, pol["q"], pol["s"], pol["r"]) & fs.active
    n = int(dots.sum())
    cand = np.argwhere(fs.active)
    rr = []
    for seed in range(5):
        pick = cand[rng.choice(len(cand), size=n, replace=False)]
        c = components_binary(pick, nom)
        rr.append(dti_from_components(c["tp"], c["fp"], c["fn"]))
        rand_tot += np.array([c["tp"], c["fp"], c["fn"]]) / 5
    rand_runs.append(rr)
    orc = nom[rng.random(nom.shape[0]) < 0.5]
    c = components_binary(orc, nom)
    orc_tot += np.array([c["tp"], c["fp"], c["fn"]])
rand_dti = dti_from_components(*rand_tot)
orc_dti = dti_from_components(*orc_tot)
out = dict(policy=pol, random_dots_same_count_pooled_dti_mean5seeds=float(rand_dti),
           oracle_half_truth_pooled_dti=float(orc_dti),
           note="controls for the holdout machinery; not competition scores")
(REPO / "registry" / "experiments").mkdir(parents=True, exist_ok=True)
(REPO / "registry" / "experiments" / "controls.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
