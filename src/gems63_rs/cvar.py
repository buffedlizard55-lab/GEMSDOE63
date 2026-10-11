"""CVaR selection and greedy pruning over the label-world ensemble.

The rules force a single submission to be chosen before the private scores are known
(competition structure, page 967), and Phase 2 re-scores that same submission against
an expert-expanded label set.  A candidate that wins on one holdout draw can lose on
another, so this lane selects on the *distribution* of scores across plausible label
worlds rather than on a point estimate.

Conditional value-at-risk
-------------------------
Following Rockafellar & Uryasev (*Journal of Risk*, 2000), with loss
L = 1 - DTI in [0, 1] and confidence level beta,

    VaR_beta(L)  = the beta-quantile of L
    CVaR_beta(L) = E[ L | L >= VaR_beta(L) ]

which for an empirical sample is the mean of the worst (1 - beta) fraction.  The
same authors show CVaR_beta(L) = min_zeta F_beta(zeta) with

    F_beta(zeta) = zeta + 1/(1 - beta) * E[ (L - zeta)_+ ],

and ``cvar_rockafellar_uryasev`` implements that minimisation; ``tests/test_cvar.py``
asserts the two routes agree, so the tail number reported in the run card is not
taken on trust.

Selection objective
-------------------
``objective = (1 - lam) * mean(DTI) + lam * CVaR_worst(DTI)``, i.e. the mean
penalised by the worst-tail average.  lam = 0 is the naive holdout mean; lam = 1
ignores the mean entirely and maximises the tail.  The default lam = 0.5 is stated in
the run card, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def var_worst_tail(scores: np.ndarray, tail: float) -> float:
    """Value-at-risk of the *lower* tail of DTI: the `tail`-quantile from below.

    Uses the inverse-CDF (``method="inverted_cdf"``) convention, so the returned
    value is exactly the largest member of the tail set that
    :func:`cvar_worst_tail` averages.  The default linear-interpolation quantile
    would report a value that is not in the sample at all and would not correspond
    to the set the CVaR is taken over.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return 0.0
    return float(np.quantile(s, tail, method="inverted_cdf"))


def cvar_worst_tail(scores: np.ndarray, tail: float = 0.2) -> float:
    """Mean of the worst `tail` fraction of DTI values.

    For an empirical sample of size M this is the mean of the smallest
    max(1, ceil(tail*M)) order statistics.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    m = s.size
    if m == 0:
        return 0.0
    k = max(1, int(np.ceil(tail * m)))
    return float(s[:k].mean())


def cvar_rockafellar_uryasev(scores: np.ndarray, beta: float = 0.8) -> float:
    """CVaR_beta of the loss L = 1 - DTI, via the Rockafellar-Uryasev minimisation.

    ``beta`` is the confidence level on the loss scale, so ``beta = 0.8`` averages
    the worst 20 % -- the same set ``cvar_worst_tail(scores, 0.2)`` picks.  Returns
    the CVaR of the *loss*; subtract from 1 for the DTI-scale tail average.
    """
    loss = 1.0 - np.asarray(scores, dtype=np.float64)
    # F_beta is convex and piecewise linear with kinks at the sample points, so the
    # minimum is attained at one of them; an exact scan is both cheap and exact.
    best = np.inf
    for z in np.unique(loss):
        f = z + float(np.maximum(loss - z, 0.0).mean()) / (1.0 - beta)
        best = min(best, f)
    return float(best)


@dataclass
class CandidateScore:
    name: str
    params: dict
    mean_dti: float
    tail_dti: float
    var_dti: float
    worst_world: str
    best_world: str
    objective: float
    per_fold_mean: dict
    n_emitted: int
    per_world: list

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("per_world")
        return d


def objective(mean_dti: float, tail_dti: float, lam: float = 0.5) -> float:
    """Mean DTI penalised by the worst-tail average."""
    return (1.0 - lam) * mean_dti + lam * tail_dti


def score_candidate(name: str, params: dict, emission, worlds, folds,
                    *, tail: float = 0.2, lam: float = 0.5,
                    scorer=None) -> CandidateScore:
    """Score one candidate emission in every world, pooled DTI per world.

    ``scorer(emission) -> {world.name: dti}`` is injectable so the same selector can
    be driven by the real grid or by a synthetic test problem.
    """
    if scorer is None:
        from .holdout import pooled_dti
        by_fold = {}
        for w in worlds:
            by_fold.setdefault(w.fold, []).append(w)

        def scorer(mask):
            out = {}
            for f in folds:
                for w in by_fold.get(f.name, []):
                    out[w.name] = pooled_dti(mask, {f.name: w.truth}, [f])["dti"]
            return out

    per_world = scorer(emission)
    names = list(per_world)
    vals = np.array([per_world[n] for n in names], dtype=np.float64)
    if vals.size == 0:
        raise ValueError("no worlds were scored")
    mean = float(vals.mean())
    tailmean = cvar_worst_tail(vals, tail)
    var = var_worst_tail(vals, tail)
    per_fold_mean = {}
    for f in folds:
        fv = [per_world[w.name] for w in worlds if w.fold == f.name]
        if fv:
            per_fold_mean[f.name] = float(np.mean(fv))
    n_emitted = int(np.asarray(emission > 0).sum())
    return CandidateScore(
        name=name, params=params, mean_dti=mean, tail_dti=tailmean, var_dti=var,
        worst_world=names[int(np.argmin(vals))], best_world=names[int(np.argmax(vals))],
        objective=objective(mean, tailmean, lam), per_fold_mean=per_fold_mean,
        n_emitted=n_emitted, per_world=[float(v) for v in vals],
    )


def greedy_prune(emission: np.ndarray, groups: dict, worlds, folds,
                 *, tail: float = 0.2, lam: float = 0.5,
                 scorer=None, verbose: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Greedily drop the group whose removal most improves the CVaR objective.

    The brief asks to "prune greedily, dropping the dot or segment whose removal most
    improves CVaR".  Doing that one *pixel* at a time on a several-hundred-thousand
    pixel emission is O(N) full-metric evaluations and is not tractable, so the unit
    of pruning here is the named **group** (a distance band, a fold, a trace family):
    the smallest grouping that still answers the question the pruning is for, which is
    *how wide and how far out should the emission reach*.  Each step re-scores every
    world, so a group survives only if keeping it improves the tail-penalised mean.
    """
    current = np.asarray(emission).copy()
    remaining = dict(groups)
    log: list[dict] = []
    base = score_candidate("current", {}, current, worlds, folds,
                           tail=tail, lam=lam, scorer=scorer)
    log.append({"step": 0, "action": "init", "objective": base.objective,
                "mean_dti": base.mean_dti, "tail_dti": base.tail_dti,
                "n_emitted": base.n_emitted})

    improved = True
    while improved and remaining:
        improved = False
        best_name, best_score = None, None
        for gname, gmask in remaining.items():
            trial = current & ~gmask
            if not trial.any():
                continue
            s = score_candidate(f"drop:{gname}", {}, trial, worlds, folds,
                                tail=tail, lam=lam, scorer=scorer)
            if best_score is None or s.objective > best_score.objective:
                best_name, best_score = gname, s
        if best_score is not None and best_score.objective > base.objective + 1e-12:
            current = current & ~remaining.pop(best_name)
            base = best_score
            improved = True
            log.append({"step": len(log), "action": f"drop:{best_name}",
                        "objective": base.objective, "mean_dti": base.mean_dti,
                        "tail_dti": base.tail_dti, "n_emitted": base.n_emitted})
            if verbose:
                print(f"  pruned {best_name}: objective {base.objective:.5f}")
    log.append({"step": len(log), "action": "final", "objective": base.objective,
                "mean_dti": base.mean_dti, "tail_dti": base.tail_dti,
                "n_emitted": base.n_emitted})
    return current, log
