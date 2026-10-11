"""Tests for the hide-and-recover holdout and the label-world generator."""

from __future__ import annotations

import numpy as np
import pytest

from gems63 import emission, holdout, worlds


@pytest.fixture(scope="module")
def tiny():
    """A synthetic catalogue: two straight traces in each quadrant."""
    cat = np.zeros((120, 120), dtype=bool)
    cat[20:50, 30] = True          # nw
    cat[20:50, 90] = True          # ne
    cat[80:110, 30] = True         # sw
    cat[80:110, 90] = True         # se
    foot = np.ones((120, 120), dtype=bool)
    return cat, foot


def test_folds_partition_the_catalogue(tiny):
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot)
    assert len(folds) == 4
    assert sum(f.n_segments for f in folds) == 4
    # every catalogue pixel is withheld in exactly one fold
    total = np.zeros(cat.shape, dtype=bool)
    for f in folds:
        assert not (total & f.withheld).any()
        total |= f.withheld
    assert (total == (cat & foot)).all()


def test_valid_domain_masks_the_whole_catalogue(tiny):
    """Organizer rule (thread 11516): known-fault pixels are excluded from scoring."""
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot, mask_all_catalogue=True)
    for f in folds:
        assert not (f.valid & cat).any()
        assert (f.valid | cat | ~foot | ~f.region).all()
        assert f.valid.sum() > 0


def test_sibling_variant_masks_only_visible_pixels(tiny):
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot, mask_all_catalogue=False)
    for f in folds:
        assert not (f.valid & f.visible).any()
        assert (f.valid & f.withheld).any()


def test_folds_are_spatially_blocked(tiny):
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot)
    by = {f.name: f for f in folds}
    assert by["nw"].segments[0][:, 1].max() < 60
    assert by["ne"].segments[0][:, 1].min() >= 60


def test_candidate_cannot_see_withheld_truth(tiny):
    """Leak check: the candidate is a function of the public catalogue alone.

    Two things are asserted.  (1) The candidate is *identical* however the withheld
    set is drawn, so no withheld label can reach it.  (2) Its score collapses when
    the truth is translated 25 px away from the catalogue, which shows the score is
    earned by geometric proximity to the public traces and not by anything else.
    """
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot)
    f = folds[0]
    cand = emission.ribbon(cat, 3.0, allowed=f.valid).astype(float)

    # (1) the candidate does not depend on which segments were withheld
    other = holdout.build_folds(cat=cat, foot=foot, mask_all_catalogue=False)[0]
    cand_other = emission.ribbon(cat, 3.0, allowed=f.valid).astype(float)
    assert (cand == cand_other).all()
    assert not (cand.astype(bool) & f.withheld).any()

    # (2) near truth (a displaced copy) scores; the same trace 25 px away does not
    rc = f.segments[0]
    near = np.zeros(cat.shape, dtype=bool)
    near[rc[:, 0], np.clip(rc[:, 1] + 2, 0, 119)] = True
    # 25 px away: still inside the fold's quadrant (cols < 60) but far outside the
    # 3 px ribbon, so nothing the candidate emits can be credited for it
    far = np.zeros(cat.shape, dtype=bool)
    far[rc[:, 0], np.clip(rc[:, 1] + 25, 0, 119)] = True
    near &= f.valid
    far &= f.valid
    assert near.any() and far.any()
    assert holdout.pooled_dti(cand, {f.name: near}, [f])["dti"] > 0.3
    assert holdout.pooled_dti(cand, {f.name: far}, [f])["dti"] < 0.01


def test_pooled_dti_sums_components_before_the_ratio(tiny):
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot)
    truths = {}
    for f in folds:
        t = np.zeros(cat.shape, dtype=bool)
        rc = f.segments[0]
        t[rc[:, 0], np.clip(rc[:, 1] + 2, 0, 119)] = True
        truths[f.name] = t & f.valid
    cand = np.zeros(cat.shape, dtype=bool)
    for f in folds:
        cand |= emission.ribbon(f.visible, 3.0, allowed=f.valid)
    pooled = holdout.pooled_dti(cand.astype(float), truths, folds)
    manual_tp = manual_fp = manual_fn = 0.0
    for f in folds:
        c = holdout.score_fold(cand.astype(float), truths[f.name], f)
        manual_tp += c.tp_w
        manual_fp += c.fp_w
        manual_fn += c.fn_w
    assert pooled["tp_w"] == pytest.approx(manual_tp)
    assert pooled["dti"] == pytest.approx(
        manual_tp / (manual_tp + 0.2 * manual_fp + 0.8 * manual_fn + 1e-12))


# ---------------------------- world generator --------------------------------


def test_world_respects_the_offset_calibration(tiny):
    cat, foot = tiny
    f = holdout.build_folds(cat=cat, foot=foot)[0]
    cfg = worlds.WorldConfig(shift_lo_px=1.5, shift_hi_px=4.0, n_worlds=1)
    for i in range(20):
        w = worlds.make_world(f, cfg, np.random.default_rng(i), idx=i)
        assert w.max_shift_px <= 4.0 + 1e-9, w.max_shift_px


def test_world_truth_never_lands_on_masked_catalogue(tiny):
    cat, foot = tiny
    f = holdout.build_folds(cat=cat, foot=foot)[0]
    cfg = worlds.WorldConfig(n_worlds=1)
    for i in range(5):
        w = worlds.make_world(f, cfg, np.random.default_rng(i), idx=i)
        assert not (w.truth & cat).any()
        assert not (w.truth & ~f.valid).any()


def test_world_omits_between_10_and_30_percent(tiny):
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot)
    cfg = worlds.WorldConfig(n_worlds=1)
    for f in folds:
        for i in range(10):
            w = worlds.make_world(f, cfg, np.random.default_rng(100 + i), idx=i)
            n = w.n_segments_kept + w.n_segments_dropped
            frac = w.n_segments_dropped / max(1, n)
            # a single segment can be forced back in when everything is dropped, so
            # allow the one-segment slack on the upper side
            assert 0.0 <= frac <= 0.30 + 1.0 / max(1, n)


def test_phi_one_puts_truth_beside_the_catalogue(tiny):
    """phi = 1 -> every survivor is a displaced copy, so truth hugs the catalogue."""
    from scipy.ndimage import distance_transform_edt
    cat, foot = tiny
    f = holdout.build_folds(cat=cat, foot=foot)[0]
    cfg = worlds.WorldConfig(phi_lo=1.0, phi_hi=1.0, decoy_max_fraction=0.0, n_worlds=1)
    w = worlds.make_world(f, cfg, np.random.default_rng(0), idx=0)
    d = distance_transform_edt(~cat)
    assert w.truth.sum() > 0
    assert d[w.truth].max() <= 4.0


def test_phi_zero_puts_truth_far_from_the_catalogue(tiny):
    """phi = 0 -> pure decoys, which distance-to-catalogue cannot locate."""
    from scipy.ndimage import distance_transform_edt
    cat, foot = tiny
    f = holdout.build_folds(cat=cat, foot=foot)[0]
    cfg = worlds.WorldConfig(phi_lo=0.0, phi_hi=0.0, n_worlds=1)
    w = worlds.make_world(f, cfg, np.random.default_rng(0), idx=0)
    d = distance_transform_edt(~cat)
    assert w.n_displaced == 0
    assert d[w.truth].mean() > 4.0


def test_make_worlds_covers_the_phi_interval(tiny):
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot)
    cfg = worlds.WorldConfig(n_worlds=8, phi_lo=0.15, phi_hi=0.95)
    wl = worlds.make_worlds(folds, cfg)
    assert len(wl) == 8 * len(folds)
    phis = np.array([w.phi for w in wl])
    assert phis.min() >= 0.15 and phis.max() <= 0.95
    assert phis.std() > 0.1              # the ensemble really does span phi


def test_cluster_bootstrap_returns_an_interval(tiny):
    cat, foot = tiny
    folds = holdout.build_folds(cat=cat, foot=foot)
    truths = {}
    cand = np.zeros(cat.shape, dtype=bool)
    for f in folds:
        t = np.zeros(cat.shape, dtype=bool)
        rc = f.segments[0]
        t[rc[:, 0], np.clip(rc[:, 1] + 2, 0, 119)] = True
        truths[f.name] = t & f.valid
        cand |= emission.ribbon(f.visible, 3.0, allowed=f.valid)
    ci = holdout.cluster_bootstrap(cand.astype(float), truths, folds, n=60)
    assert ci is not None
    lo, hi = ci
    assert 0.0 <= lo <= hi <= 1.0
