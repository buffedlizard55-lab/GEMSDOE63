"""Tests for the emission policies, the CVaR selector, the validator and the gate."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from gems63_rs import cvar, emission
from gems63_rs.raster_io import EXPECTED_SHAPE, write_submission
from gems63_rs.uniqueness import (
    OVERLAP_GATE,
    RANK_CORR_GATE,
    compare_to_registry,
    overlap_fraction,
    spearman_rank_corr,
    uniqueness_gate,
)
from gems63_rs.validate import validate_submission


# ------------------------------- emission ------------------------------------


def test_ribbon_excludes_the_source_pixels():
    src = np.zeros((40, 40), dtype=bool)
    src[20, 5:35] = True
    r = emission.ribbon(src, 2.0, exclude_source=True)
    assert not (r & src).any()
    assert r.any()


def test_ribbon_width_grows_monotonically():
    src = np.zeros((60, 60), dtype=bool)
    src[30, 10:50] = True
    sizes = [emission.ribbon(src, w).sum() for w in (1, 2, 3, 4)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_distance_bands_partition_the_ribbon():
    src = np.zeros((60, 60), dtype=bool)
    src[30, 10:50] = True
    bands = [emission.distance_band(src, b - 1, b) for b in range(1, 5)]
    for a, b in zip(bands, bands[1:]):
        assert not (a & b).any()
    union = bands[0]
    for b in bands[1:]:
        union = union | b
    assert (union == emission.ribbon(src, 4.0, exclude_source=True)).all()


def test_nms_thin_enforces_separation_and_never_adds():
    rng = np.random.default_rng(0)
    mask = rng.random((80, 80)) < 0.3
    pri = rng.random((80, 80))
    thin = emission.nms_thin(mask, pri, 3.0)
    assert (thin <= mask).all()
    assert thin.sum() < mask.sum()
    ys, xs = np.nonzero(thin)
    d2 = ((ys[:, None] - ys[None, :]) ** 2 + (xs[:, None] - xs[None, :]) ** 2).astype(float)
    np.fill_diagonal(d2, np.inf)
    assert np.sqrt(d2.min()) >= 2.0        # NMS keeps points at least ~sep apart


def test_budget_cap_keeps_the_highest_priority():
    mask = np.ones((10, 10), dtype=bool)
    pri = np.arange(100, dtype=float).reshape(10, 10)
    out = emission.budget_cap(mask, pri, 5)
    assert out.sum() == 5
    assert set(pri[out].tolist()) == {95.0, 96.0, 97.0, 98.0, 99.0}


def test_dti_optimal_threshold_follows_the_marginal_bar():
    """A higher operating DTI must emit strictly less (the bar is alpha*DTI).

    pi has to be *sparse* for this to bite: with 29 neighbours in the kernel disc, a
    dense pi saturates E[k] near 1 everywhere and no bar in (0, 1) can exclude
    anything.  At pi ~ U(0, 0.02) the field straddles realistic bar values.
    """
    rng = np.random.default_rng(1)
    pi = rng.random((60, 60)) * 0.02
    lo = emission.dti_optimal(pi, 0.01)
    hi = emission.dti_optimal(pi, 0.60)
    assert hi.sum() < lo.sum()
    assert emission.dti_optimal(pi, 0.01).sum() > 0


def test_continuations_extend_past_tips_only():
    cat = np.zeros((60, 60), dtype=bool)
    cat[30, 10:40] = True
    cont = emission.continuations(cat, 8)
    assert not (cont & cat).any()
    assert cont.any()
    assert cont[:, 40:].sum() > 0          # continued beyond the mapped end


# --------------------------------- CVaR --------------------------------------


def test_cvar_tail_is_the_mean_of_the_worst_fraction():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert cvar.cvar_worst_tail(s, 0.2) == pytest.approx(np.mean([0.1, 0.2]))
    assert cvar.var_worst_tail(s, 0.2) == pytest.approx(0.2)


def test_cvar_rockafellar_uryasev_equals_the_empirical_tail():
    """The two routes to CVaR must agree (Rockafellar & Uryasev 2000)."""
    rng = np.random.default_rng(4)
    s = rng.random(500)
    for beta, tail in ((0.8, 0.2), (0.9, 0.1), (0.95, 0.05)):
        ru = cvar.cvar_rockafellar_uryasev(s, beta=beta)
        emp = 1.0 - cvar.cvar_worst_tail(s, tail)
        assert ru == pytest.approx(emp, abs=1e-9), (beta, ru, emp)


def test_objective_interpolates_mean_and_tail():
    assert cvar.objective(0.4, 0.2, lam=0.0) == pytest.approx(0.4)
    assert cvar.objective(0.4, 0.2, lam=1.0) == pytest.approx(0.2)
    assert cvar.objective(0.4, 0.2, lam=0.5) == pytest.approx(0.3)


def test_selector_prefers_the_stable_candidate():
    """A candidate with a fatter bad tail must lose at lam = 1 even if it wins on mean."""
    worlds = [type("W", (), {"name": f"w{i}", "fold": "a"})() for i in range(10)]
    folds = [type("F", (), {"name": "a"})()]
    stable = np.full(10, 0.30)
    lucky = np.concatenate([np.full(8, 0.20), np.full(2, 0.90)])

    def scorer(mask):
        vals = stable if mask[0, 0] == 1 else lucky
        return {w.name: float(v) for w, v in zip(worlds, vals)}

    m1 = np.zeros((2, 2), bool); m1[0, 0] = True
    m0 = np.zeros((2, 2), bool)
    s1 = cvar.score_candidate("stable", {}, m1, worlds, folds, lam=1.0, scorer=scorer)
    s0 = cvar.score_candidate("lucky", {}, m0, worlds, folds, lam=1.0, scorer=scorer)
    assert s1.mean_dti < s0.mean_dti          # lucky wins on the mean
    assert s1.objective > s0.objective        # stable wins on the tail


def test_greedy_prune_removes_a_harmful_group():
    """A group that only costs false positives must be pruned."""
    worlds = [type("W", (), {"name": f"w{i}", "fold": "a"})() for i in range(6)]
    folds = [type("F", (), {"name": "a"})()]
    truth = np.zeros((30, 30), bool); truth[10:20, 15] = True
    good = np.zeros((30, 30), bool); good[10:20, 15] = True
    junk = np.zeros((30, 30), bool); junk[25:29, 25:29] = True
    emit = good | junk

    from gems63_rs.metric import binary_dti_components

    def scorer(mask):
        d = binary_dti_components(mask, truth).dti
        return {w.name: d for w in worlds}

    out, log = cvar.greedy_prune(emit, {"good": good, "junk": junk}, worlds, folds,
                                 scorer=scorer)
    assert (out == good).all()
    assert any(r["action"] == "drop:junk" for r in log)
    assert not any(r["action"] == "drop:good" for r in log)


# ------------------------------- validator -----------------------------------


def test_write_then_validate_roundtrip_passes(tmp_path):
    arr = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    arr[100:110, 100:110] = 1.0
    p = tmp_path / "ok.tif"
    write_submission(arr, p)
    rep = validate_submission(p)
    assert rep.ok, rep.failures
    assert rep.checks["CRS is EPSG:32611"]
    assert rep.checks["shape matches template (3730 x 3292)"]
    assert rep.checks["geotransform matches template exactly"]
    assert rep.checks["dtype is float32"]
    assert rep.checks["min value >= 0"] and rep.checks["max value <= 1"]


def test_write_rejects_values_above_one(tmp_path):
    """This is the exact failure the submission form reported."""
    arr = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    arr[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        write_submission(arr, tmp_path / "bad.tif")


def test_write_rejects_nan_and_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        write_submission(np.full(EXPECTED_SHAPE, np.nan, np.float32), tmp_path / "n.tif")
    with pytest.raises(ValueError, match="shape"):
        write_submission(np.zeros((10, 10), np.float32), tmp_path / "s.tif")


def test_validator_flags_a_nan_inside_the_file(tmp_path):
    arr = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    p = tmp_path / "nan.tif"
    arr[500, 500] = np.nan            # a NaN inside the survey footprint
    with rasterio.open("data/official/sample_submission.tif") as tpl:
        prof = tpl.profile.copy()
    prof.update(compress=None, tiled=False)
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(arr, 1)
    rep = validate_submission(p, require_all_finite=True)
    assert not rep.ok
    assert any("no NaN anywhere" in f for f in rep.failures)


# ------------------------------ uniqueness -----------------------------------


def test_identical_raster_is_flagged_as_duplicate():
    rng = np.random.default_rng(0)
    a = rng.random((200, 200)) < 0.02
    comps = compare_to_registry(a, {"same": {"array": a.copy(), "sha256": "x",
                                             "owner_reported_score": 0.1}})
    assert comps[0].rank_corr == pytest.approx(1.0, abs=1e-9)
    assert comps[0].overlap_fraction == pytest.approx(1.0)
    gate = uniqueness_gate(comps)
    assert not gate["clear"]
    assert gate["duplicates"] == ["same"]


def test_disjoint_raster_passes_the_gate():
    a = np.zeros((300, 300), bool); a[20:40, 20] = True
    b = np.zeros((300, 300), bool); b[200:220, 250] = True
    comps = compare_to_registry(a, {"other": {"array": b, "sha256": "y",
                                              "owner_reported_score": None}})
    gate = uniqueness_gate(comps)
    assert gate["clear"]
    assert comps[0].overlap_fraction < OVERLAP_GATE
    assert comps[0].rank_corr < RANK_CORR_GATE


def test_overlap_fraction_measures_within_3px():
    a = np.zeros((60, 60), bool); a[30, 30] = True
    b = np.zeros((60, 60), bool); b[30, 33] = True      # exactly 3 px away
    c = np.zeros((60, 60), bool); c[30, 34] = True      # 4 px away
    assert overlap_fraction(a, b) == pytest.approx(1.0)
    assert overlap_fraction(a, c) == pytest.approx(0.0)


def test_gate_thresholds_are_the_brief_values():
    assert RANK_CORR_GATE == 0.90
    assert OVERLAP_GATE == 0.70
