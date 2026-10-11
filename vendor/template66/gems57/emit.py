"""Dot allocation: greedy max-coverage under a clipped-convolution surrogate.

Why this allocation rule
------------------------
For a binary dot field (see :mod:`gems57.metric` for the derivation, including
why the denominator does **not** collapse to ``alpha*n + beta*|G|``)::

    D   = alpha*(T + n - M) + beta*|G|
    DTI = T / D

where ``T`` is the covered truth credit and ``M`` the total self-credit of the
emitted dots.  The allocation surrogate assigns a dot at ``x`` **fixed** self-credit
``k(x) = E[k](x) = sum_g p(g) k(d(x,g))`` -- the convolution of the per-cell
truth probability with the kernel -- which does not depend on the other dots,
This clipped convolution is an approximation to unknown-truth max self-credit,
not its exact expectation; its marginal coverage ``dT(x)`` falls as neighbouring truth cells get
covered.  The exact acceptance test ``dT > alpha*DTI*(dT + 1 - k)`` therefore
rearranges to a per-candidate bar::

    dT > c * (1 - k(x)),     c = alpha*DTI / (1 - alpha*DTI)

which is what the loop below applies.  Candidates are processed in rounds in
decreasing ``E[k]`` order with the bar frozen inside a round; a candidate is only
re-evaluated when a nearby acceptance could have changed its ``dT``.  ``dT`` is
non-increasing and the bar rises while DTI rises, so the loop terminates when a
round accepts nothing.

Nothing about the *shape* of the emission is assumed here -- the spatial
structure comes entirely from ``p``, which :mod:`gems57.anatomy` fits to the
hide-and-recover holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .metric import ALPHA, BETA, EPS, OFF_DX, OFF_DY, OFF_K


def expected_credit(p: np.ndarray) -> np.ndarray:
    """``E[k](x) = sum_g p(g) k(d(x,g))`` -- convolution of ``p`` with the kernel.

    For a first dot this sums per-cell expected coverage. The greedy surrogate
    uses its clipped value for self-credit, although expected maximum kernel
    self-credit is not generally the expectation of this sum. Actual holdout
    scores are computed by the exact shared evaluator, not by this surrogate.
    """
    p = np.asarray(p, np.float32)
    out = np.zeros(p.shape, np.float32)
    H, W = p.shape
    for j, i, k in zip(OFF_DY, OFF_DX, OFF_K):
        ys0, ys1 = max(0, j), min(H, H + j)
        xs0, xs1 = max(0, i), min(W, W + i)
        out[ys0:ys1, xs0:xs1] += p[max(0, -j):min(H, H - j),
                                   max(0, -i):min(W, W - i)] * np.float32(k)
    return out


@dataclass
class Allocation:
    emitted: np.ndarray
    n_dots: int
    expected_covered_credit: float = 0.0   # running T under the surrogate p
    expected_self_credit: float = 0.0      # running M under the surrogate p
    rounds: int = 0
    surrogate_dti: float = 0.0
    trace: list = field(default_factory=list)


def greedy_allocate(p: np.ndarray, allowed: np.ndarray, k_truth: float, *,
                    floor: float = 0.02, max_dots: int = 200_000,
                    candidate_cap: int = 400_000, per_round: int = 20_000,
                    max_rounds: int = 400, trace_every: int = 0) -> Allocation:
    """Select dots while each one provably raises the surrogate DTI.

    Parameters
    ----------
    p
        Per-cell probability of being a hidden truth cell, in [0, 1].
    allowed
        Boolean mask of cells that may carry a dot (footprint, catalogue removed).
    k_truth
        ``|G|``, the number of hidden truth cells, which sets the bar.
    floor
        Cells with ``E[k] < floor`` are never considered.  Pure speed-up: with
        ``alpha = 0.2`` the bar is ``c*(1-k) >= 0`` and any candidate accepted at
        a realistic score has ``E[k]`` well above this.
    """
    p = np.asarray(p, np.float32)
    H, W = p.shape
    ek = expected_credit(p)
    cand = allowed & (ek >= floor)
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return Allocation(np.zeros(p.shape, bool), 0)
    order = np.argsort(-ek[ys, xs], kind="stable")
    if order.size > candidate_cap:
        order = order[:candidate_cap]
    ys, xs = ys[order], xs[order]
    ekc = ek[ys, xs].astype(np.float64)
    # ``E[k]`` is a *sum* of p*k over the neighbourhood and can exceed 1, but the
    # metric's per-dot self-credit is ``max_g k(d(x,g))``, which cannot.  Using
    # the unclipped sum makes ``1 - k`` negative, which flips the bar negative and
    # lets zero-credit dots through -- that bug produced 7,987 duplicate
    # acceptances on a synthetic field (``IR-57-KCLIP-01``).
    kself = np.minimum(ekc, 1.0)
    ncand = ys.size

    cover = np.zeros(p.shape, np.float32)
    emitted = np.zeros(p.shape, bool)
    upper = ekc.copy()            # non-increasing upper bound on each candidate's dT
    dirty = np.ones(ncand, bool)  # needs (re-)evaluation
    T = M = 0.0
    n_dots = 0
    rounds = 0
    trace: list[dict] = []

    while n_dots < max_dots and rounds < max_rounds:
        rounds += 1
        def _c() -> float | None:
            D_ = ALPHA * (T + n_dots - M) + BETA * k_truth + EPS
            a_ = ALPHA * (T / D_)
            return None if a_ >= 1.0 else a_ / (1.0 - a_)

        c = _c()
        if c is None:
            break
        # the bar is recomputed after every acceptance: DTI moves inside a round,
        # and freezing it lets marginal acceptances slip through that would
        # actually lower DTI (the surrogate DTI was non-monotone before this fix)
        eligible = dirty & (upper > c * (1.0 - kself))
        if not eligible.any():
            break
        idxs = np.flatnonzero(eligible)
        if idxs.size > per_round:
            idxs = idxs[:per_round]
        accepted = 0
        for i in idxs:
            if n_dots >= max_dots:
                break
            if c is None:
                break
            y, x = int(ys[i]), int(xs[i])
            if emitted[y, x]:
                dirty[i] = False
                continue
            ny = y + OFF_DY
            nx = x + OFF_DX
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            nyv, nxv, kw = ny[ok], nx[ok], OFF_K[ok]
            pv = p[nyv, nxv].astype(np.float64)
            cv = cover[nyv, nxv].astype(np.float64)
            dT = float((pv * np.maximum(kw - cv, 0.0)).sum())
            upper[i] = dT
            dirty[i] = False
            if dT <= c * (1.0 - kself[i]):
                continue
            emitted[y, x] = True
            n_dots += 1
            T += dT
            M += float(kself[i])
            np.maximum.at(cover, (nyv, nxv), kw.astype(np.float32))
            accepted += 1
            c = _c()
            # candidates within 6 px can have a changed dT (3 px kernel, twice)
            m = (ys >= y - 6) & (ys <= y + 6) & (xs >= x - 6) & (xs <= x + 6)
            dirty[m] = True
            dirty[i] = False
            if trace_every and n_dots % trace_every == 0:
                D2 = ALPHA * (T + n_dots - M) + BETA * k_truth + EPS
                trace.append({"n_dots": n_dots, "T": T, "M": M,
                              "dti_est": T / D2, "last_dT": dT})
        if accepted == 0:
            break

    D = ALPHA * (T + n_dots - M) + BETA * k_truth + EPS
    return Allocation(emitted=emitted, n_dots=n_dots,
                      expected_covered_credit=T, expected_self_credit=M,
                      rounds=rounds, surrogate_dti=float(T / D), trace=trace)


def dilate_zone(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a mask by ``px`` pixels (square structuring element)."""
    return ndi.binary_dilation(np.asarray(mask, bool), iterations=int(px))


def allocate_by_marginal_bar(p: np.ndarray, allowed: np.ndarray, k_truth: float, *,
                             floor: float = 0.0, max_dots: int = 200_000,
                             candidate_cap: int = 600_000,
                             chunk: int = 50_000) -> Allocation:
    """Select dots in descending ``E[k]`` order while each one raises the DTI.

    Same decision rule as :func:`greedy_allocate` -- the exact two-sided test

        (T + dT) / D1  >  T / D0,     D = alpha*(T + n - M) + beta*K

    -- but evaluated once per candidate in a single descending pass instead of
    by repeated round scans.  Under the (stated) approximation that a candidate's
    marginal truth credit ``dT`` is non-increasing as better-ranked candidates
    are accepted first, the single pass is the same greedy solution; it costs
    ``O(n_cand * |kernel|)`` instead of ``O(n_dots * n_cand)``, which is what
    makes a whole-footprint allocation (millions of candidates) tractable.

    Candidates are visited in decreasing ``E[k] = p (*) k`` order and the loop
    stops at the first candidate whose marginal credit no longer pays the bar,
    which is where the greedy optimum sits.
    """
    p = np.asarray(p, np.float32)
    H, W = p.shape
    ek = expected_credit(p)
    cand = allowed & (ek >= floor)
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return Allocation(np.zeros(p.shape, bool), 0)
    order = np.argsort(-ek[ys, xs], kind="stable")
    if order.size > candidate_cap:
        order = order[:candidate_cap]
    ys, xs = ys[order], xs[order]
    kself = np.minimum(ek[ys, xs].astype(np.float64), 1.0)

    cover = np.zeros(p.shape, np.float32)
    emitted = np.zeros(p.shape, bool)
    T = M = 0.0
    n_dots = 0
    K = float(k_truth)
    stop_at = None
    for start in range(0, ys.size, chunk):
        if n_dots >= max_dots:
            break
        cy, cx, ksl = ys[start:start + chunk], xs[start:start + chunk], kself[start:start + chunk]
        for i in range(cy.size):
            if n_dots >= max_dots:
                break
            y, x = int(cy[i]), int(cx[i])
            ny = y + OFF_DY
            nx = x + OFF_DX
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            nyv, nxv, kw = ny[ok], nx[ok], OFF_K[ok]
            pv = p[nyv, nxv].astype(np.float64)
            cv = cover[nyv, nxv].astype(np.float64)
            dT = float((pv * np.maximum(kw - cv, 0.0)).sum())
            if n_dots == 0:
                if dT <= 0.0:
                    continue
            else:
                D0 = ALPHA * (T + n_dots - M) + BETA * K + EPS
                D1 = ALPHA * (T + dT + (n_dots + 1) - (M + kself[i])) + BETA * K + EPS
                if not ((T + dT) / D1 > T / D0):
                    stop_at = start + i
                    break
            emitted[y, x] = True
            n_dots += 1
            T += dT
            M += float(ksl[i])
            np.maximum.at(cover, (nyv, nxv), kw.astype(np.float32))
        if stop_at is not None:
            break
    D = ALPHA * (T + n_dots - M) + BETA * K + EPS
    return Allocation(emitted=emitted, n_dots=n_dots, expected_covered_credit=T,
                      expected_self_credit=M, rounds=1, surrogate_dti=float(T / D))


def allocate_patient(p: np.ndarray, allowed: np.ndarray, k_truth: float, *,
                    floor: float = 0.0, max_dots: int = 200_000,
                    candidate_cap: int = 4_000_000,
                    patience: int = 100_000) -> Allocation:
    """Greedy allocation that skips failing candidates instead of stopping.

    :func:`allocate_by_marginal_bar` visits candidates once in descending
    ``E[k] = p (*) k`` order and **breaks** at the first candidate that fails
    the two-sided DTI test.  That single pass is only the greedy solution while
    a candidate's marginal credit ``dT`` is non-increasing in rank -- and it is
    not, because ``dT`` depends on *local* kernel saturation, not on rank.  On
    a surface with large flat plateaus of near-equal ``E[k]`` (a tight cluster
    of high-probability cells) the next candidate in row-major order can sit on
    top of a dot already placed, return ``dT = 0``, and end the pass thousands
    of dots early.  Measured on the session-4 surface: it stopped at 3,405 dots
    where the same bar supports roughly ten times as many.

    This variant keeps the identical accept/reject test but only stops after
    ``patience`` consecutive rejections, since a rejected candidate says
    nothing about one ranked below it.  Every decision the single pass made is
    reproduced, so the result is never worse; the extra cost is scanning the
    tail of the ranking.

    ``n_skipped`` and ``n_scanned`` are returned on ``trace`` for diagnostics.
    """
    p = np.asarray(p, np.float32)
    H, W = p.shape
    ek = expected_credit(p)
    cand = allowed & (ek >= floor)
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return Allocation(np.zeros(p.shape, bool), 0)
    order = np.argsort(-ek[ys, xs], kind="stable")
    if order.size > candidate_cap:
        order = order[:candidate_cap]
    ys, xs = ys[order], xs[order]
    kself = np.minimum(ek[ys, xs].astype(np.float64), 1.0)

    cover = np.zeros(p.shape, np.float32)
    emitted = np.zeros(p.shape, bool)
    T = M = 0.0
    n_dots = 0
    n_skipped = 0
    n_fail = 0
    K = float(k_truth)
    for i in range(ys.size):
        if n_dots >= max_dots or n_fail >= patience:
            break
        y, x = int(ys[i]), int(xs[i])
        ny = y + OFF_DY
        nx = x + OFF_DX
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nyv, nxv, kw = ny[ok], nx[ok], OFF_K[ok]
        pv = p[nyv, nxv].astype(np.float64)
        cv = cover[nyv, nxv].astype(np.float64)
        dT = float((pv * np.maximum(kw - cv, 0.0)).sum())
        if n_dots == 0:
            if dT <= 0.0:
                n_skipped += 1
                continue
        else:
            D0 = ALPHA * (T + n_dots - M) + BETA * K + EPS
            D1 = ALPHA * (T + dT + (n_dots + 1) - (M + kself[i])) + BETA * K + EPS
            if not ((T + dT) / D1 > T / D0):
                n_fail += 1
                n_skipped += 1
                continue
        emitted[y, x] = True
        n_dots += 1
        n_fail = 0
        T += dT
        M += float(kself[i])
        np.maximum.at(cover, (nyv, nxv), kw.astype(np.float32))
    D = ALPHA * (T + n_dots - M) + BETA * K + EPS
    return Allocation(emitted=emitted, n_dots=n_dots, expected_covered_credit=T,
                      expected_self_credit=M, rounds=1, surrogate_dti=float(T / D),
                      trace=[("n_skipped", n_skipped), ("n_scanned", int(i) + 1)])


def surrogate_terms(dots: np.ndarray, p: np.ndarray, k_truth: float) -> dict:
    """Closed-form surrogate DTI of a candidate dot set under the fitted posterior.

    ``T``  = covered credit ``sum_g p(g) * max_dot k(d(g,dot))``  (kernel dilation of
    the dot set, weighted by the posterior -- the same expectation
    :func:`greedy_allocate` accumulates, but computed in one vectorised pass).

    ``M``  = total self-credit ``sum_dot max_{dot'} k(d(dot,dot'))``, i.e. how much
    of the emitted mass the metric can excuse as "near a prediction".

    Both are *surrogates*: the true maxima are over the unknown hidden truth, so
    ``T``/``M`` here are the same clipped-convolution approximation the greedy
    allocator uses.  The reported HOLDOUT-DTI is always recomputed by the shared
    evaluator; this function exists only to choose a dot budget in seconds
    instead of hours.
    """
    p = np.asarray(p, np.float32)
    dots_b = np.asarray(dots, bool)
    if p.shape != dots_b.shape:
        raise ValueError("dots and posterior must share the grid")
    H, W = p.shape
    cover = np.zeros(p.shape, np.float32)
    self_credit = np.zeros(p.shape, np.float32)
    for j, i, k in zip(OFF_DY, OFF_DX, OFF_K):
        ys0, ys1 = max(0, j), min(H, H + j)
        xs0, xs1 = max(0, i), min(W, W + i)
        shifted = dots_b[max(0, -j):min(H, H - j), max(0, -i):min(W, W - i)]
        kk = np.float32(k)
        np.maximum(cover[ys0:ys1, xs0:xs1], shifted * kk, out=cover[ys0:ys1, xs0:xs1])
        np.maximum(self_credit[ys0:ys1, xs0:xs1], shifted * kk,
                   out=self_credit[ys0:ys1, xs0:xs1])
    T = float((p * cover).sum())
    M = float(self_credit[dots_b].sum())
    n = int(dots_b.sum())
    D = ALPHA * (T + n - M) + BETA * float(k_truth) + EPS
    return {"n_dots": n, "T": T, "M": M,
            "surrogate_dti": float(T / D) if D > 0 else 0.0}


def emit_decimated(p: np.ndarray, allowed: np.ndarray, *, tile_px: int = 3,
                   floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Rank every ``tile_px`` x ``tile_px`` tile by its best ``E[k]`` and return
    (dot mask for the full ranking, per-tile expected credit).

    Why a decimated ranking rather than a second greedy loop
    -------------------------------------------------------
    :func:`greedy_allocate` is exact for the shared decision rule but costs
    ``O(n_dots * n_candidates)`` because every acceptance re-scans the dirty
    neighbourhood: measured on this repository's own surfaces it admits roughly
    ten thousand dots per hour, while the registry evidence from the highest
    owner-reported files shows the live optimum sits in the tens of thousands.
    The decimated ranking keeps the *decision variable* (``E[k]``, the same
    convolution the greedy loop uses) and replaces the dirty re-scan with a
    spacing constraint: at most one dot per ``tile_px`` x ``tile_px`` tile,
    chosen at the tile's ``argmax E[k]``.  Because the kernel radius is 3 px, a
    3 px tile grid is exactly the spacing at which a dot stops duplicating its
    neighbour's credit, which is the same geometry the registry's best files
    reach by thinning a thick surface.

    The returned dot mask is the *ranking*: taking the ``n`` highest-credit
    tiles gives the nested budget family used by
    :func:`gems57.emit.surrogate_terms`, so a budget curve costs one pass.
    """
    p = np.asarray(p, np.float32)
    allowed = np.asarray(allowed, bool)
    if p.shape != allowed.shape:
        raise ValueError("posterior and allowed mask must share the grid")
    if tile_px < 1:
        raise ValueError("tile_px must be >= 1")
    ek = expected_credit(p)
    ek = np.where(allowed, ek, 0.0)
    H, W = p.shape
    ph = (-H) % tile_px
    pw = (-W) % tile_px
    padded = np.pad(ek, ((0, ph), (0, pw)), mode="constant")
    th, tw = padded.shape[0] // tile_px, padded.shape[1] // tile_px
    tiles = padded.reshape(th, tile_px, tw, tile_px).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(th * tw, tile_px * tile_px)
    tile_max = tiles.max(axis=1)
    tile_arg = tiles.argmax(axis=1)
    stay = tile_max > np.float32(floor)
    tile_ids = np.flatnonzero(stay)
    order = tile_ids[np.argsort(-tile_max[tile_ids], kind="stable")]
    ty = (order // tw) * tile_px
    tx = ((order % tw) * tile_px)
    ay = ty + (tile_arg[order] // tile_px)
    ax = tx + (tile_arg[order] % tile_px)
    keep = (ay < H) & (ax < W)
    ay, ax = ay[keep], ax[keep]
    stride = np.arange(1, ay.size + 1)
    ranked = np.zeros(p.shape, np.int32)
    ranked[ay, ax] = stride
    return ranked, tile_max[order][keep]


def emit_prefix(ranked: np.ndarray, budget: int) -> np.ndarray:
    """The first ``budget`` dots of a :func:`emit_decimated` ranking."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    return (ranked > 0) & (ranked <= int(budget))
