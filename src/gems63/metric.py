"""Official Distance-Weighted Tversky Index (DTI) for the DOE GEMS Prize Challenge.

PROVENANCE: adapted verbatim (bit-exact) from the shared template
buffedlizard55-lab/GEMSDOE32 src/gems32/metric.py (session 32, 2026-10-04), which
states: "Verified from official competition problem description (DrivenData Page
967): https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
and official organizer forum clarification on known-fault masking (Thread 11516):
https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516"

Mathematical formulation:
- Triangular kernel: k(d) = max(1 - d / R, 0), with R = 300 m = 3 pixels at 100 m.
- TP_w = sum_{g in G} max_{x : d(x,g) <= R} p(x) * k(d(x,g))
- FP_w = sum_{x : p(x) > 0} p(x) * [1 - max_{g in G} k(d(x,g))]
- FN_w = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))] = |G| - TP_w
- DTI(alpha=0.2, beta=0.8) = TP_w / (TP_w + 0.2 * FP_w + 0.8 * FN_w + eps)
"""

from __future__ import annotations

import numpy as np

ALPHA: float = 0.2
BETA: float = 0.8
RADIUS_PX: float = 3.0
EPS: float = 1e-12


def kernel(d, radius: float = RADIUS_PX):
    """k(d) = max(1 - d / R, 0). d and radius in pixels (1 px = 100 m)."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


def _offsets(radius: float = RADIUS_PX):
    r = int(np.ceil(radius))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = float(kernel(np.hypot(dy, dx), radius))
            if k > 0.0:
                out.append((dy, dx, k))
    return out


_OFFS = _offsets()


def dti_from_counts(tp: float, fp: float, n_truth: int, alpha: float = ALPHA, beta: float = BETA) -> float:
    """Pooled DTI from pooled weighted counts (exactly as the shared template)."""
    fn = float(n_truth) - tp
    return float(tp / (tp + alpha * fp + beta * fn + EPS))
