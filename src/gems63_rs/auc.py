"""Dependency-free ROC-AUC, used only by the leakage canary.

The parallel-run brief requires every feature to be tested alone on the holdout
before any result is trusted, with AUC > 0.90 treated as leakage until proven
otherwise.  That needs one number, so it is computed here from the Mann-Whitney U
identity rather than by pulling in scikit-learn:

    AUC = ( sum of ranks of the positives - n_pos*(n_pos+1)/2 ) / (n_pos * n_neg)

with ties handled by average ranks, which is exactly the usual
P(score(positive) > score(negative)) + 0.5 P(tie) estimator.
"""

from __future__ import annotations

import numpy as np


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC of `scores` for discriminating `labels` (True = positive)."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    # average ranks within ties
    i = 0
    r = 1.0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + (r + (j - i))) / 2.0
        r += (j - i) + 1
        i = j + 1
    rank_sum = float(ranks[y].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
