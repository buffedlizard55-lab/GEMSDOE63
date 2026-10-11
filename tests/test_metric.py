"""Tests for gems63_rs.metric — anchored on the competition's own published numbers."""

from __future__ import annotations

import numpy as np
import pytest

from gems63_rs.metric import (
    ALPHA,
    BETA,
    RADIUS_PX,
    dti_bruteforce,
    dti_components,
    dti_from_components,
    expected_kernel_weight,
    kernel,
    marginal_bar,
    offsets_within,
)


def test_kernel_is_triangular_with_300m_support():
    assert kernel(0.0) == 1.0
    assert kernel(1.5) == 0.5
    assert kernel(3.0) == 0.0
    assert kernel(4.0) == 0.0
    assert RADIUS_PX == 3.0


def test_published_tversky_example():
    """Page 967 worked example: TP=3.00, FP=1.89, FN=2.00 -> displayed as 0.60.

    The page prints "0.60"; evaluating its own formula exactly gives
    3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 3.00 / 4.978 = 0.6026516..., which is
    0.60 to the two decimals the page shows.  Both facts are asserted here so the
    rounding is not mistaken for a discrepancy.
    """
    got = dti_from_components(3.00, 1.89, 2.00)
    assert got == pytest.approx(3.00 / (3.00 + 0.2 * 1.89 + 0.8 * 2.00), rel=1e-12)
    assert round(got, 2) == 0.60


def test_alpha_beta_are_the_published_values():
    assert ALPHA == 0.2 and BETA == 0.8


def test_offsets_disc_has_29_pixels_for_radius_3():
    assert len(offsets_within(3.0)) == 29


def test_fn_is_n_truth_minus_tp_identity():
    """FN_w = |G| - TP_w, from which DTI = T/(0.2(T+S)+0.8N) follows."""
    rng = np.random.default_rng(7)
    truth = rng.random((40, 40)) < 0.05
    pred = (rng.random((40, 40)) < 0.20).astype(float)
    c = dti_components(pred, truth)
    assert c.fn_w == pytest.approx(c.n_truth - c.tp_w, abs=1e-9)
    alt = c.tp_w / (0.2 * (c.tp_w + c.fp_w) + 0.8 * c.n_truth)
    assert c.dti == pytest.approx(alt, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_bruteforce_transcription(seed):
    """The fast implementation equals the literal O(|G|*|P|) transcription."""
    rng = np.random.default_rng(seed)
    truth = rng.random((18, 18)) < 0.08
    # float predictions, not just binary, to exercise the weighted max-filter
    pred = np.where(rng.random((18, 18)) < 0.25, rng.random((18, 18)), 0.0)
    fast = dti_components(pred, truth)
    slow = dti_bruteforce(pred, truth)
    assert fast.tp_w == pytest.approx(slow.tp_w, abs=1e-9)
    assert fast.fp_w == pytest.approx(slow.fp_w, abs=1e-9)
    assert fast.fn_w == pytest.approx(slow.fn_w, abs=1e-9)
    assert fast.dti == pytest.approx(slow.dti, abs=1e-9)


def test_valid_mask_removes_pixels_from_every_sum():
    """Organizer masking rule (thread 11516): masked pixels count nowhere."""
    pred = np.zeros((12, 12))
    pred[3, 3] = 1.0          # emission inside the masked zone
    pred[9, 9] = 1.0          # emission in the scored domain, far from truth
    truth = np.zeros((12, 12), dtype=bool)
    truth[3, 4] = True        # truth inside the masked zone
    truth[6, 6] = True        # truth in the scored domain
    valid = np.ones((12, 12), dtype=bool)
    valid[2:5, 2:5] = False

    c = dti_components(pred, truth, valid=valid)
    # masked truth removed -> only (6,6); masked emission removed -> only (9,9)
    assert c.n_truth == 1
    assert c.n_emitted == 1
    assert c.tp_w == pytest.approx(0.0)
    assert c.fp_w == pytest.approx(1.0)      # d((9,9),(6,6)) > 3 -> kmax = 0
    assert c.fn_w == pytest.approx(1.0)


def test_dot_exactly_on_a_straight_line_earns_three():
    """The published example's arithmetic: one dot centred on a line earns TP_w = 3."""
    truth = np.zeros((9, 9), dtype=bool)
    truth[:, 4] = True                     # vertical line, as in the published figure
    pred = np.zeros((9, 9))
    pred[4, 4] = 1.0                       # single dot on the line
    c = dti_components(pred, truth)
    # pixels within 3 px of the dot: d = 0,1,1,2,2,3,3 -> k = 1, 2/3, 2/3, 1/3, 1/3, 0, 0
    assert c.tp_w == pytest.approx(1 + 2 / 3 + 2 / 3 + 1 / 3 + 1 / 3, abs=1e-9)
    assert c.fp_w == pytest.approx(0.0, abs=1e-12)   # dot sits on truth: no penalty


def test_marginal_bar_is_alpha_times_dti():
    assert marginal_bar(0.2778) == pytest.approx(0.2 * 0.2778)


def test_emission_above_the_marginal_bar_raises_the_score():
    """k > alpha*DTI is a genuine improvement criterion, verified numerically."""
    rng = np.random.default_rng(11)
    truth = rng.random((60, 60)) < 0.04
    base = (rng.random((60, 60)) < 0.10).astype(float)
    c0 = dti_components(base, truth)
    bar = marginal_bar(c0.dti)

    # add a dot at k = 1 (on truth): must improve
    on_truth = np.argwhere(truth & (base == 0))[0]
    up = base.copy()
    up[on_truth[0], on_truth[1]] = 1.0
    assert dti_components(up, truth).dti > c0.dti
    assert 1.0 > bar

    # add a dot far from any truth: k = 0 < bar, must hurt
    far = np.argwhere((~truth) & (base == 0))
    from scipy.ndimage import distance_transform_edt
    dg = distance_transform_edt(~truth)
    far = far[dg[far[:, 0], far[:, 1]] > RADIUS_PX][0]
    down = base.copy()
    down[far[0], far[1]] = 1.0
    assert dti_components(down, truth).dti < c0.dti


@pytest.mark.parametrize("level", [0.05, 0.2, 0.4])
def test_expected_kernel_weight_matches_simulation(level):
    """E[k] from the closed form equals Monte-Carlo under the independence model.

    With truth = a set of pixels, TP_w / |truth| is exactly the mean over those
    pixels of max_g k(d) -- the same quantity ``expected_kernel_weight`` returns.
    The check is restricted to the *interior*, because a pixel within 3 px of the
    array edge has fewer than 29 neighbours and a genuinely lower E[k].
    """
    rng = np.random.default_rng(3)
    n = 24
    pi = np.full((n, n), level)
    analytic = expected_kernel_weight(pi)
    inner = np.zeros((n, n), dtype=bool)
    inner[3:-3, 3:-3] = True
    a = analytic[inner]
    assert a.min() == pytest.approx(a.max(), abs=1e-9)   # interior is flat

    sims = []
    for _ in range(6000):
        draw = (rng.random((n, n)) < level).astype(float)
        sims.append(dti_components(draw, inner).tp_w / inner.sum())
    mc = float(np.mean(sims))
    assert abs(float(a.mean()) - mc) < 0.01, (a.mean(), mc)


def test_expected_kernel_weight_matches_hand_calculation():
    """Independent check: pi = 0.2 uniform, interior, must give 0.65552.

    Hand-enumerated: P(max = v_j) = S_{j-1} - S_j with S_j = 0.8^(cumulative count)
      v=1      : 1     * (1 - 0.8^1)  = 0.20000
      v=2/3    : 0.6667* (0.8^1 - 0.8^5)  = 0.31488
      v=0.5286 : 0.5286* (0.8^5 - 0.8^9)  = 0.10226
      v=1/3    : 0.3333* (0.8^9 - 0.8^13) = 0.02641
      v=0.2546 : 0.2546* (0.8^13- 0.8^21) = 0.01166
      v=0.0572 : 0.0572* (0.8^21- 0.8^25) = 0.00031
    total 0.65552
    """
    analytic = expected_kernel_weight(np.full((16, 16), 0.2))
    got = float(analytic[3:-3, 3:-3].mean())
    assert got == pytest.approx(0.65552, abs=2e-4)


def test_perfect_prediction_scores_one():
    truth = np.zeros((20, 20), dtype=bool)
    truth[5:15, 10] = True
    pred = truth.astype(float)
    assert dti_components(pred, truth).dti == pytest.approx(1.0, abs=1e-9)


def test_empty_prediction_scores_zero():
    truth = np.zeros((20, 20), dtype=bool)
    truth[5:15, 10] = True
    assert dti_components(np.zeros((20, 20)), truth).dti == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_binary_fast_path_equals_general_path(seed):
    """binary_dti_components must equal dti_components exactly for p in {0, 1}."""
    from gems63_rs.metric import binary_dti_components
    rng = np.random.default_rng(seed)
    truth = rng.random((64, 64)) < 0.05
    emit = rng.random((64, 64)) < 0.15
    valid = rng.random((64, 64)) < 0.85
    a = dti_components(emit.astype(float), truth, valid=valid)
    b = binary_dti_components(emit, truth, valid=valid)
    assert a.tp_w == pytest.approx(b.tp_w, rel=1e-12)
    assert a.fp_w == pytest.approx(b.fp_w, rel=1e-12)
    assert a.fn_w == pytest.approx(b.fn_w, rel=1e-12)
    assert a.dti == pytest.approx(b.dti, rel=1e-12)


def test_binary_fast_path_honours_precomputed_edt():
    from scipy.ndimage import distance_transform_edt
    from gems63_rs.metric import binary_dti_components
    rng = np.random.default_rng(9)
    emit = rng.random((50, 50)) < 0.1
    valid = np.ones((50, 50), bool)
    pre = distance_transform_edt(~(emit & valid))
    for _ in range(3):
        truth = rng.random((50, 50)) < 0.05
        a = binary_dti_components(emit, truth, valid=valid)
        b = binary_dti_components(emit, truth, valid=valid, edt_emit=pre)
        assert a.dti == pytest.approx(b.dti, rel=1e-12)


@pytest.mark.parametrize("scores,labels,expected", [
    ([1, 2, 3, 4], [False, False, True, True], 1.0),
    ([4, 3, 2, 1], [False, False, True, True], 0.0),
    ([1, 1, 1, 1], [False, False, True, True], 0.5),
    # positives score 1,3,5 and negatives 2,4,6 -> 3 of 9 pairs ordered correctly
    ([1, 2, 3, 4, 5, 6], [True, False, True, False, True, False], 3 / 9),
])
def test_auc_matches_known_values(scores, labels, expected):
    from gems63_rs.auc import auc_from_scores
    assert auc_from_scores(np.array(scores, float), np.array(labels, bool)) == \
        pytest.approx(expected)
