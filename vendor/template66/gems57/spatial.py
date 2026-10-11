"""R2 spatial whole-component folds and fail-closed negative-error diagnostics."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi, stats


def disk(radius):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def component_folds(catalogue, eligible):
    """Whole original 8-connected components; majority eligible quadrant wins."""
    h, w = catalogue.shape
    yy, xx = np.indices((h, w), sparse=True)
    quadrant = ((yy >= h // 2).astype(np.int8) * 2 + (xx >= w // 2).astype(np.int8))
    comp, n = ndi.label(catalogue, np.ones((3, 3), bool))
    votes = np.zeros((n + 1, 4), np.int64)
    sel = catalogue & eligible
    np.add.at(votes, (comp[sel], quadrant[sel]), 1)
    owner = votes.argmax(axis=1).astype(np.int8)
    assigned = owner[comp]
    assigned[comp == 0] = -1
    return comp, assigned, quadrant


def legacy_component_tail_folds(catalogue, eligible, buffer_px=80):
    """REPRODUCTION ONLY: rejected CTD5-v1 geometry-informed evaluation domains.

    The tail halo depends on withheld labels and can leak location to a regional
    placement rule. Retained only to reproduce the negative historical receipt;
    NEVER use this split as strict promotion evidence. Use folds() below.
    """
    comp, assigned, quadrant = component_folds(catalogue, eligible)
    for f in range(4):
        held_all = catalogue & (assigned == f)
        truth = held_all & eligible
        # Include full available component tails, and all kernel-support pixels around them.
        region = eligible & ((quadrant == f) | ndi.binary_dilation(held_all, structure=disk(3)))
        train = eligible & (ndi.distance_transform_edt(~region) > buffer_px) & ~held_all
        visible = catalogue & ~held_all
        train_components = np.unique(comp[train & catalogue])
        held_components = np.unique(comp[truth])
        shared = np.intersect1d(train_components[train_components > 0], held_components[held_components > 0])
        if shared.size:
            raise AssertionError("a connected catalogue component leaked into training")
        if (train & region).any() or (train & held_all).any():
            raise AssertionError("spatial train/evaluation overlap")
        minimum = float(ndi.distance_transform_edt(~region)[train].min()) if train.any() else None
        yield dict(fold=f, region=region, quadrant=quadrant == f, train=train,
                   truth=truth, visible=visible, held_all=held_all, components=comp,
                   receipt=dict(fold=f, truth_px=int(truth.sum()), evaluation_px=int(region.sum()),
                                training_domain_px=int(train.sum()), held_components=int((held_components > 0).sum()),
                                shared_train_truth_components=int(shared.size),
                                buffer_px=buffer_px, nearest_training_to_region_px=minimum,
                                held_pixels_without_feature_support=int((held_all & ~eligible).sum())))



def folds(catalogue, eligible, buffer_px=80):
    """Fixed, label-blind evaluation quadrants; entire intersecting segments hidden.

    A component crossing several quadrants is excluded from training in EACH of
    those folds, while its pixels are scored only in their fixed spatial quadrant.
    Evaluation/placement geometry depends on grid and eligible footprint only,
    never on the hidden trace. The buffer covers both the quadrant and all held
    component tails. Boundary clipping is conservative, not a truth-shaped halo.
    """
    comp, _, quadrant = component_folds(catalogue, eligible)
    for f in range(4):
        region = eligible & (quadrant == f)
        ids = np.unique(comp[catalogue & region])
        ids = ids[ids > 0]
        held_all = (np.isin(comp, ids) & catalogue) if len(ids) else np.zeros_like(catalogue, bool)
        truth = held_all & region
        exclusion = region | held_all
        train = eligible & (ndi.distance_transform_edt(~exclusion) > buffer_px)
        visible = catalogue & ~held_all
        training_components = np.unique(comp[train & catalogue])
        if np.intersect1d(training_components[training_components > 0], ids).size:
            raise AssertionError('a whole held component reached training')
        if (train & region).any() or (train & held_all).any():
            raise AssertionError('train/evaluation overlap')
        minimum = float(ndi.distance_transform_edt(~region)[train].min()) if train.any() else None
        yield dict(fold=f, region=region, quadrant=quadrant == f, train=train,
            truth=truth, visible=visible, held_all=held_all, components=comp,
            receipt=dict(fold=f, split_version='label-blind-quadrants-v2',
                evaluation_region_label_blind=True, truth_px=int(truth.sum()),
                evaluation_px=int(region.sum()), training_domain_px=int(train.sum()),
                held_components=len(ids), shared_train_truth_components=0, buffer_px=buffer_px,
                nearest_training_to_region_px=minimum,
                held_pixels_without_feature_support=int((held_all & ~eligible).sum()),
                held_positive_pixels_outside_this_evaluation_region=int((held_all & eligible & ~region).sum())))

def correlations(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or np.ptp(a) <= 1e-14 or np.ptp(b) <= 1e-14:
        return dict(n=int(len(a)), pearson=None, spearman=None, reason="too few observations or constant errors")
    return dict(n=int(len(a)), pearson=float(stats.pearsonr(a, b).statistic),
                spearman=float(stats.spearmanr(a, b).statistic))


def negative_block_errors(pa, pb, negative_mask, fold, thresholds, side=50, minimum=32):
    """OOF continuous and threshold errors on actual held-out catalogue-zero proxies.

    Does not accept unfilled zero OOF grids as measurements. Caller supplies a
    finite coverage mask and non-training region. Same negatives for both views.
    """
    h, w = negative_mask.shape
    rows = []
    for y in range(0, h, side):
        for x in range(0, w, side):
            sl = np.s_[y:min(y + side, h), x:min(x + side, w)]
            good = negative_mask[sl] & np.isfinite(pa[sl]) & np.isfinite(pb[sl])
            if int(good.sum()) < minimum:
                continue
            a, b = pa[sl][good].astype(float), pb[sl][good].astype(float)
            rows.append(dict(fold=int(fold), block_row=y // side, block_col=x // side, n_negatives=int(good.sum()),
                             mse_A=float(np.mean(a * a)), mse_B=float(np.mean(b * b)),
                             fpr_A=float(np.mean(a >= thresholds[0])), fpr_B=float(np.mean(b >= thresholds[1]))))
    return rows


def independence(rows, threshold=0.6, min_blocks=20):
    out = dict(n_blocks=len(rows), n_negative_predictions=sum(r["n_negatives"] for r in rows),
               negative_class="held-out catalogue-zero proxies, not verified absence",
               threshold=threshold, minimum_blocks=min_blocks, tests={})
    for name, a, b in (("negative_mean_squared_error", "mse_A", "mse_B"), ("negative_false_positive_rate", "fpr_A", "fpr_B")):
        out["tests"][name] = correlations([r[a] for r in rows], [r[b] for r in rows])
    values = [abs(c[k]) for c in out["tests"].values() for k in ("pearson", "spearman") if c[k] is not None]
    undefined = any(c[k] is None for c in out["tests"].values() for k in ("pearson", "spearman"))
    strong = bool(values and max(values) >= threshold)
    out.update(max_abs_correlation=max(values) if values else None,
               measured=not undefined and len(rows) >= min_blocks,
               allow_exchange=not undefined and len(rows) >= min_blocks and not strong,
               reason="strongly correlated OOF negative errors: abandon co-training" if strong else
                      "insufficient or degenerate negative error blocks: disable co-training" if undefined or len(rows) < min_blocks else
                      "weak proxy negative-error correlation; not proof of conditional feature independence")
    out["blocks"] = rows
    return out


def whole_pseudo_segments(donor, receiver, train, forbidden, donor_threshold, receiver_lo, receiver_hi,
                          side=50, min_pixels=5, cap=2000):
    """Select whole confident-disagreement components, never crossing train/block boundaries.

    Components cut by a training boundary, a labelled sample, a known-fault
    buffer, or a 50x50 block edge are rejected rather than clipped into fragments.
    Receiver must abstain in a middle quantile interval, not be confidently negative.
    """
    candidate = np.isfinite(donor) & np.isfinite(receiver) & (donor >= donor_threshold) & (receiver >= receiver_lo) & (receiver <= receiver_hi)
    comp, n = ndi.label(candidate, np.ones((3, 3), bool))
    objects = ndi.find_objects(comp)
    kept, receipts, total = [], [], 0
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        local = comp[sl] == i
        yy, xx = np.nonzero(local)
        yy, xx = yy + sl[0].start, xx + sl[1].start
        if len(yy) < min_pixels or len(yy) + total > cap:
            continue
        if not train[yy, xx].all() or forbidden[yy, xx].any():
            continue
        by, bx = yy // side, xx // side
        if np.unique(by).size != 1 or np.unique(bx).size != 1:
            continue
        idx = yy * donor.shape[1] + xx
        kept.append(idx)
        total += len(idx)
        receipts.append(dict(component=int(i), pixels=int(len(idx)), block_row=int(by[0]), block_col=int(bx[0]),
                             mean_donor=float(donor[yy, xx].mean()), mean_receiver=float(receiver[yy, xx].mean())))
    return np.concatenate(kept) if kept else np.empty(0, np.int64), receipts
