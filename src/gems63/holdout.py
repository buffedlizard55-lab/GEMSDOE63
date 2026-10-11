"""Hide-and-recover holdout: withhold whole catalogue segments, score pooled DTI.

Protocol (from the parallel-run brief, §2)
------------------------------------------
* **Folds.** The grid is split into four spatial quadrants (2x2).  Whole 8-connected
  catalogue segments are assigned to the fold containing their centroid, so a fold
  never sees part of a segment that another fold scores.
* **Withholding.** The segments of a fold are the held-out positives.  Everything
  else is visible.
* **Masking.** The scored domain is `footprint & ~visible catalogue pixels`,
  pixel-exact.  This is the organizer's rule verbatim -- chrisk-dd, thread 11516,
  2026-09-16: "Pixels corresponding to known USGS/INGENIOUS faults are masked /
  excluded from evaluation, so they do not count towards penalty terms."  No
  buffered dilation is applied, because the pixels *adjacent* to a mapped trace are
  precisely the region this lane predicts and precisely the region the organizer
  leaves in the scored set.
* **Metric.** The exact pooled DTI of :mod:`gems63.metric`: alpha 0.2, beta 0.8,
  300 m triangular kernel.  Pooling sums the components across folds before forming
  the ratio, which is what "pooled DTI" means.
* **Uncertainty.** Cluster bootstrap over whole withheld segments -- the segment,
  not the pixel, is the independent unit.

Leakage discipline
------------------
The candidate emission is built from the **visible** catalogue only.  The withheld
segments reach the pipeline solely through :mod:`gems63.worlds`, which turns them
into perturbed *truth*.  Nothing derived from a withheld segment is allowed into a
candidate; ``tests/test_holdout.py`` asserts that a candidate built from the visible
catalogue scores 0 when the withheld truth is translated far away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .metric import binary_dti_components, dti_components
from .raster_io import catalogue, footprint

FOLD_NAMES = ("nw", "ne", "sw", "se")
STRUCT8 = np.ones((3, 3), dtype=int)


@dataclass
class Fold:
    """One hide-and-recover fold."""

    name: str
    region: np.ndarray            # bool: pixels belonging to this quadrant
    visible: np.ndarray           # bool: catalogue pixels NOT withheld in this fold
    withheld: np.ndarray          # bool: pixels of the withheld segments
    valid: np.ndarray             # bool: the scored domain
    segments: list[np.ndarray] = field(default_factory=list)   # (N,2) int coords

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def quadrant_regions(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = shape
    south = np.arange(h)[:, None] >= h // 2
    east = np.arange(w)[None, :] >= w // 2
    return {
        "nw": (~south) & (~east),
        "ne": (~south) & east,
        "sw": south & (~east),
        "se": south & east,
    }


def segment_coords(cat: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """8-connected catalogue segments as (N, 2) (row, col) coordinate arrays.

    Uses ``find_objects`` bounding boxes rather than a full-array comparison per
    label: the naive loop is O(n_labels * n_pixels) and took 119 s on the real
    3730x3292 catalogue, which made every experiment untenable.
    """
    struct = STRUCT8 if connectivity == 8 else None
    lab, n = ndi.label(cat, structure=struct)
    if n == 0:
        return []
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl]
        rr, cc = np.nonzero(sub == i)
        if rr.size:
            out.append(np.column_stack([rr + sl[0].start, cc + sl[1].start]))
    return out


def build_folds(cat: np.ndarray | None = None,
                foot: np.ndarray | None = None,
                *,
                min_segment_px: int = 1,
                mask_all_catalogue: bool = True) -> list[Fold]:
    """Split the catalogue into four spatially-blocked hide-and-recover folds."""
    cat = catalogue() if cat is None else np.asarray(cat, bool)
    foot = footprint() if foot is None else np.asarray(foot, bool)
    cat = cat & foot

    segs = [s for s in segment_coords(cat) if len(s) >= min_segment_px]
    regions = quadrant_regions(cat.shape)

    folds: dict[str, Fold] = {}
    for name in FOLD_NAMES:
        folds[name] = Fold(name=name, region=regions[name],
                           visible=np.zeros(cat.shape, bool),
                           withheld=np.zeros(cat.shape, bool),
                           valid=np.zeros(cat.shape, bool), segments=[])

    for rc in segs:
        cy, cx = rc[:, 0].mean(), rc[:, 1].mean()
        name = ("s" if cy >= cat.shape[0] // 2 else "n") + \
               ("e" if cx >= cat.shape[1] // 2 else "w")
        folds[name].segments.append(rc)

    for name, f in folds.items():
        withheld = np.zeros(cat.shape, bool)
        for rc in f.segments:
            withheld[rc[:, 0], rc[:, 1]] = True
        f.withheld = withheld
        f.visible = cat & ~withheld
        if mask_all_catalogue:
            # Organizer rule verbatim (thread 11516): known USGS/INGENIOUS fault
            # pixels are masked out of evaluation.  The whole catalogue is public,
            # so the whole catalogue is masked -- including the pixels this fold
            # withholds, whose *displaced* copies are what becomes truth.
            visible_masked = cat
        else:
            # Sibling-protocol variant: only the still-mapped pixels are masked.
            visible_masked = f.visible
        f.valid = foot & ~visible_masked & f.region
    return [folds[n] for n in FOLD_NAMES]


def score_fold(prediction: np.ndarray, truth: np.ndarray, fold: Fold,
               edt_emit: np.ndarray | None = None):
    """Exact DTI of `prediction` against `truth` inside `fold.valid`.

    Binary predictions (the optimum for this metric) take the two-distance-transform
    path; a precomputed emission distance transform may be passed so the same
    candidate can be scored against many truths cheaply.  The earlier version always
    used the general float path -- a 29-shift max-filter over the full 3730x3292 grid
    -- which made a 120-replicate bootstrap take minutes per replicate.
    """
    p = np.asarray(prediction)
    if p.dtype == bool or np.array_equal(np.unique(p), np.array([0.0, 1.0])):
        return binary_dti_components(p.astype(bool), truth, valid=fold.valid,
                                     edt_emit=edt_emit)
    return dti_components(p, truth, valid=fold.valid)


def pooled_dti(prediction: np.ndarray, truths: dict[str, np.ndarray],
               folds: list[Fold]) -> dict:
    """Pooled DTI: components summed across folds, then the ratio formed once."""
    tp = fp = fn = 0.0
    n_truth = n_emitted = 0
    for f in folds:
        if f.name not in truths:
            continue
        c = score_fold(prediction, truths[f.name], f)
        tp += c.tp_w
        fp += c.fp_w
        fn += c.fn_w
        n_truth += c.n_truth
        n_emitted += c.n_emitted
    dti = tp / (tp + 0.2 * fp + 0.8 * fn + 1e-12)
    return dict(tp_w=tp, fp_w=fp, fn_w=fn, n_truth=n_truth,
                n_emitted=n_emitted, dti=float(dti))


def cluster_bootstrap(prediction: np.ndarray,
                      truths: dict[str, np.ndarray],
                      folds: list[Fold],
                      *, n: int = 200, seed: int = 20261011) -> tuple[float, float] | None:
    """Percentile 95 % CI of the pooled DTI from a whole-segment cluster bootstrap.

    The resampled unit is a connected component **of the truth**, not of the
    catalogue.  That distinction is not cosmetic in this lane: the truth is a
    laterally *displaced* copy of a withheld segment, so its pixels never coincide
    with the withheld segment's own pixels, and an earlier version of this routine
    matched on the catalogue pixels, found no overlap, and returned ``None`` for
    every input.  A truth segment is also the right geological unit -- it is one
    fault, and faults are what is independently present or absent.
    """
    rng = np.random.default_rng(seed)
    per_fold = []
    for f in folds:
        if f.name not in truths:
            continue
        t = truths[f.name] & f.valid
        if not t.any():
            continue
        lab, nseg = ndi.label(t, structure=STRUCT8)
        if nseg == 0:
            continue
        per_fold.append((f, lab, nseg))
    if not per_fold:
        return None

    # The emission side is fixed across bootstrap replicates, so its distance
    # transform is computed once per fold rather than once per replicate.
    from scipy.ndimage import distance_transform_edt
    pb = np.asarray(prediction)
    binary = pb.dtype == bool or np.array_equal(np.unique(pb), np.array([0.0, 1.0]))
    pre = {}
    if binary:
        for f, _lab, _n in per_fold:
            pre[f.name] = distance_transform_edt(~(pb.astype(bool) & f.valid))

    vals = []
    for _ in range(n):
        tp = fp = fn = 0.0
        for f, lab, nseg in per_fold:
            take = np.unique(rng.integers(1, nseg + 1, size=nseg))
            t = np.isin(lab, take)
            c = score_fold(prediction, t, f, edt_emit=pre.get(f.name))
            tp += c.tp_w
            fp += c.fp_w
            fn += c.fn_w
        vals.append(tp / (tp + 0.2 * fp + 0.8 * fn + 1e-12))
    if len(vals) < 10:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))
