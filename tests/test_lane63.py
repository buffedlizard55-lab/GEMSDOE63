"""Fast regression tests for lane 63 (run with pytest from the repo root)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "template66"))
sys.path.insert(0, str(ROOT / "src"))

from gems57 import holdout as H  # noqa: E402
from gems57 import metric  # noqa: E402
from gems57.grid import load_grid  # noqa: E402
from gems63 import canary, cvar, policies  # noqa: E402
from gems63.worlds_arena8f46dad0 import WorldCell  # noqa: E402


def _cell_stub():
    class C:
        pass
    c = C()
    c.key = "k"
    c.bbox = (slice(0, 40), slice(0, 40))
    c.active = np.ones((40, 40), bool)
    return c


def test_cvar_and_objective_monotone():
    dtis = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert cvar.cvar(dtis, 0.2) == pytest.approx(0.1)
    assert cvar.cvar(dtis, 1.0) == pytest.approx(dtis.mean())
    better = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    assert cvar.objective(better) > cvar.objective(dtis)


def test_world_terms_reproduces_template_evaluator():
    g = load_grid()
    mask, diag = policies.dotted_ridge(g.catalogue, g.footprint,
                                       policies.PolicySpec(name="t", spacing=2.8),
                                       return_diagnostics=True)
    assert diag["dots"] > 30000
    assert int((mask & g.catalogue).sum()) == 0, "dots must never sit on the catalogue"
    ctx = H.build_holdout(g, seeds=(20,), modes=("all",))
    cell = [c for c in ctx.cells if c.mode == "all"][0]
    truth = ctx.hidden_by_cell[cell.key][cell.bbox] & cell.active
    ty, tx = np.nonzero(truth)
    w0 = WorldCell(cell.key, 0, (ty, tx), int(len(ty)), 0, 0, 0, 0, 0)
    wt = cvar.world_terms([cell], {cell.key: [w0]}, mask)
    r = H.score_cell(cell, mask)
    assert wt["tp"][0] == pytest.approx(r["tp"], abs=1e-6)
    assert wt["fp"][0] == pytest.approx(r["fp"], abs=1e-4)  # float32 k_dg in template
    assert wt["fn"][0] == pytest.approx(r["fn"], abs=1e-6)


def test_prune_overlapping_cells_exact():
    ca = _cell_stub()
    cb = _cell_stub()
    cb.key = "k2"
    truth = np.zeros((40, 40), bool)
    truth[20, 5:35] = True
    ty, tx = np.nonzero(truth)
    truth_b = np.zeros((40, 40), bool)
    truth_b[21, 5:35] = True
    tyb, txb = np.nonzero(truth_b)
    dots = np.zeros((40, 40), bool)
    dots[19, ::3] = True
    dots[35, ::3] = True
    res = cvar.greedy_prune([ca, cb],
                            {"k": [WorldCell("k", 0, (ty, tx), len(ty), 0, 0, 0, 0, 0)],
                             "k2": [WorldCell("k2", 0, (tyb, txb), len(tyb), 0, 0, 0, 0, 0)]},
                            dots, lam=0.5, q=0.5, eps=1e-9, max_iter=60,
                            verbose=False, verify_every=3)
    assert res["dots"][35].sum() == 0
    assert res["dots"][19].sum() >= 10
    wt_b = cvar.world_terms([ca, cb], {"k": [WorldCell("k", 0, (ty, tx), len(ty), 0, 0, 0, 0, 0)],
                                       "k2": [WorldCell("k2", 0, (tyb, txb), len(tyb), 0, 0, 0, 0, 0)]},
                            dots)
    wt_a = cvar.world_terms([ca, cb], {"k": [WorldCell("k", 0, (ty, tx), len(ty), 0, 0, 0, 0, 0)],
                                       "k2": [WorldCell("k2", 0, (tyb, txb), len(tyb), 0, 0, 0, 0, 0)]},
                            res["dots"])
    assert cvar.objective(wt_a["dtis"], 0.5, 0.5) >= cvar.objective(wt_b["dtis"], 0.5, 0.5) - 1e-12
    assert abs(cvar.objective(wt_a["dtis"], 0.5, 0.5) - res["objective_end"]) < 1e-9


def test_metric_break_even_bar():
    # break-even credit for one more non-redundant dot: k > alpha * DTI
    assert metric.marginal_inclusion_threshold(0.28) == pytest.approx(0.056)


def test_canary_auc_perfect_and_random():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1], bool)
    assert canary.auc(s, y) == pytest.approx(1.0)
    s2 = np.array([0.9, 0.8, 0.2, 0.1])
    assert canary.auc(s2, y) == pytest.approx(0.0)
