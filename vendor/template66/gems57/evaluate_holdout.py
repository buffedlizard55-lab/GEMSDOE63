"""Shared hide-and-recover evaluator for the H57 spatial-quadrant instrument.

Delegates official-formula arithmetic to metric.max_cover / dti_from_components.
The inherited holdout.score API was replaced elsewhere while this file still
called it; v2 repairs that shared interface rather than adding a private metric. Spatial block terms are bookkeeping for a
conditional bootstrap, not a new scoring rule. No translated synthetic truth.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from . import metric

VERSION = 'gems57-pooled-hide-v2'


def implementation_hashes():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ('evaluate_holdout.py', 'holdout.py', 'metric.py', 'spatial.py')}


def evaluate(prediction, fold, valid, block_side=200, *, origin=(0, 0),
             global_shape=None):
    """Exact fold score plus additive terms on globally aligned spatial blocks.

    ``origin`` and ``global_shape`` locate a cropped fold in the common pixel
    grid. Pass them when combining terms from multiple crop windows; otherwise
    local block coordinates would not denote the same physical clusters.
    """
    if block_side <= 0:
        raise ValueError('block_side must be positive')
    p = np.asarray(prediction)
    if p.ndim != 2:
        raise ValueError('prediction must be a 2D array')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('predictions must be finite in [0,1] before masking')
    valid = np.asarray(valid, bool)
    region = np.asarray(fold['region'], bool)
    known = np.asarray(fold.get('masked_known', fold['visible']), bool)
    truth_all = np.asarray(fold['truth'], bool)
    if p.ndim != 2 or any(a.shape != p.shape for a in (valid, region, known, truth_all)):
        raise ValueError('fold/prediction grid shape mismatch')
    active = valid & region & ~known
    p = np.where(active, p, 0).astype(np.float64)
    truth = active & truth_all
    if not truth.any():
        raise ValueError('a holdout fold must contain positives')
    result = metric.dti_exact(p, truth, valid=active)
    covers, q, _ = metric.max_cover(p, truth)
    tpw = float(covers.sum())
    fpw = float((p * (1.0 - q)).sum())
    fnw = float(truth.sum()) - tpw
    h, w = truth.shape
    oy, ox = map(int, origin)
    gh, gw = map(int, global_shape or (h + oy, w + ox))
    if oy < 0 or ox < 0 or oy + h > gh or ox + w > gw:
        raise ValueError('crop origin/global_shape do not contain the evaluation arrays')
    ncols = (gw + block_side - 1) // block_side
    nrows = (gh + block_side - 1) // block_side
    terms = np.zeros((ncols * nrows, 4), np.float64)
    y, x = np.nonzero(truth)
    gy, gx = y + oy, x + ox
    ids = (gy // block_side) * ncols + (gx // block_side)
    for j, v in ((0, covers), (2, 1.0 - covers), (3, np.ones(len(y)))):
        terms[:, j] = np.bincount(ids, weights=v, minlength=len(terms))
    y, x = np.nonzero(p > 0)
    gy, gx = y + oy, x + ox
    ids = (gy // block_side) * ncols + (gx // block_side)
    terms[:, 1] = np.bincount(ids, weights=p[y, x].astype(float) * (1.0 - q[y, x]), minlength=len(terms))
    totals = terms.sum(axis=0)
    np.testing.assert_allclose(
        totals, [result['tp'], result['fp'], result['fn'], result['n_truth']],
        rtol=1e-11, atol=1e-7)
    result.update(tpw=result['tp'], fpw=result['fp'], fnw=result['fn'],
                  n_truth=int(truth.sum()), n_emitted=int((p > 0).sum()),
                  emitted=int((p > 0).sum()),
                  evidence_class='HOLDOUT-DTI', evaluator_version=VERSION,
                  spatial_bootstrap_cluster_m=block_side * metric.PIXEL_M)
    return result, terms


def from_terms(terms):
    a = np.asarray(terms, dtype=float)
    tp, fp, fn = a[..., 0], a[..., 1], a[..., 2]
    den = tp + metric.ALPHA * fp + metric.BETA * fn + metric.EPS
    return np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)


def pooled_summary(terms_by_arm, draws=1000, seed=520810, candidate='disagreement'):
    """Pool terms first; paired percentile CI resamples physical 20 km blocks.

    Each arm has identical block coordinates/order and evaluation masks. Contributions
    from the same physical block across folds are merged BEFORE resampling, so overlap
    in a component tail is not mistaken for an independent new spatial cluster.
    """
    names = list(terms_by_arm)
    if not names or candidate not in terms_by_arm:
        raise ValueError('candidate and at least one arm required')
    if draws < 20:
        raise ValueError('too few bootstrap draws')
    arrays = {n: np.asarray(v, float) for n, v in terms_by_arm.items()}
    shape = next(iter(arrays.values())).shape
    if len(shape) != 2 or shape[1] != 4 or any(a.shape != shape for a in arrays.values()):
        raise ValueError('aligned per-spatial-block arrays of four terms required')
    if any(not np.isfinite(a).all() or (a < -1e-8).any() for a in arrays.values()):
        raise ValueError('invalid metric terms')
    truth_counts = [a[:, 3] for a in arrays.values()]
    reference_counts = truth_counts[0]
    if not np.allclose(reference_counts, np.rint(reference_counts), rtol=0.0, atol=1e-8):
        raise ValueError('spatial-block truth counts must be integers')
    if any(not np.array_equal(counts, reference_counts) for counts in truth_counts[1:]):
        raise ValueError('all candidate/control arms must use the same integer truth count per spatial block')
    # Include negative-only clusters carrying FP weight for any comparator.
    active = np.any(np.stack([a.sum(axis=1) > 0 for a in arrays.values()]), axis=0)
    arrays = {n: a[active] for n, a in arrays.items()}
    nb = int(active.sum())
    if nb < 2:
        raise ValueError('need at least two nonempty spatial clusters')
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, nb, size=(draws, nb))
    boot = {n: from_terms(a[picks].sum(axis=1)) for n, a in arrays.items()}
    summaries = {}
    for n, a in arrays.items():
        t = a.sum(axis=0)
        summaries[n] = dict(evidence_class='HOLDOUT-DTI', evaluator_version=VERSION,
            dti=float(from_terms(t)), ci95=[float(x) for x in np.quantile(boot[n], [.025, .975])],
            withheld_positive_pixels=int(round(t[3])), tpw=float(t[0]), fpw=float(t[1]), fnw=float(t[2]))
    controls = [n for n in names if n != candidate]
    best = max(controls, key=lambda n: summaries[n]['dti']) if controls else None
    differences = {}
    for n in controls:
        delta = boot[candidate] - boot[n]
        differences[n] = dict(evidence_class='HOLDOUT-DTI', evaluator_version=VERSION,
            delta=summaries[candidate]['dti'] - summaries[n]['dti'],
            ci95=[float(x) for x in np.quantile(delta, [.025, .975])],
            withheld_positive_pixels=summaries[candidate]['withheld_positive_pixels'])
    return dict(evidence_class='HOLDOUT-DTI', evaluator_version=VERSION,
        alpha=metric.ALPHA, beta=metric.BETA, triangular_radius_m=metric.R_M,
        pooled=True, scores=summaries, best_comparable_control=best,
        paired_differences=differences, bootstrap=dict(method='paired physical spatial-cluster percentile',
            clusters=nb, draws=draws, seed=seed, confidence=.95,
            caveat='Conditional on fitted folds, catalogue labels and fixed budgets; not a leaderboard interval.'),
        implementation_sha256=implementation_hashes())
