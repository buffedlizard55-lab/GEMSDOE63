"""Verify the fast exact DTI against the literal brute-force transcription of the official formula."""
import numpy as np

from gems63.dti import dti_binary_pixels, dti_bruteforce, cvar_lower


def _rand_case(rng, h=40, w=40, n_dots=30, n_truth=25):
    dots = np.zeros((h, w), bool)
    truth = np.zeros((h, w), bool)
    dots.flat[rng.choice(h * w, n_dots, replace=False)] = True
    truth.flat[rng.choice(h * w, n_truth, replace=False)] = True
    return dots, truth


def test_fast_matches_bruteforce_random_grids():
    rng = np.random.default_rng(0)
    for _ in range(25):
        dots, truth = _rand_case(rng)
        fast = dti_binary_pixels(np.argwhere(dots), np.argwhere(truth))
        slow = dti_bruteforce(dots.astype(float), truth)
        assert abs(fast["tp"] - slow["tp"]) < 1e-9
        assert abs(fast["fp"] - slow["fp"]) < 1e-9
        assert abs(fast["fn"] - slow["fn"]) < 1e-9
        assert abs(fast["dti"] - slow["dti"]) < 1e-9


def test_worked_example_single_dot_on_truth():
    # One truth pixel with one dot exactly on it: TP=1, FP=0, FN=0 -> DTI = 1
    d = dti_binary_pixels(np.array([[5, 5]]), np.array([[5, 5]]))
    assert abs(d["dti"] - 1.0) < 1e-9  # EPS in denominator


def test_edge_cases_empty():
    assert dti_binary_pixels(np.zeros((0, 2), int), np.array([[1, 1]]))["dti"] == 0.0
    d = dti_binary_pixels(np.array([[0, 0]]), np.zeros((0, 2), int))
    assert d["n_truth"] == 0 and d["dti"] == 0.0


def test_kernel_radius_is_three_pixels():
    # dot at distance exactly 3 px gets zero credit; at 1.5 px gets 0.5
    assert dti_binary_pixels(np.array([[0, 3]]), np.array([[0, 0]]))["tp"] == 0.0
    assert abs(dti_binary_pixels(np.array([[0, 1]]), np.array([[0, 0]]))["tp"] - (2 / 3)) < 1e-12


def test_cvar_lower_tail():
    vals = np.arange(10, dtype=float)  # worst 20% of 10 values = {0,1}
    assert cvar_lower(vals, 0.2) == 0.5
