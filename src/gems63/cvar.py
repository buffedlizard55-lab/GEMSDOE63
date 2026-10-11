"""CVaR scoring, selection and greedy pruning over label worlds (lane 63).

All quantities are computed with the exact official-kernel primitives from the
shared template (``gems57.metric``): triangular kernel, alpha = 0.2, beta = 0.8,
3 px radius.  For each holdout cell and each label world we track the four
metric terms (TP_w, FP_w, FN_w, |G|) and pool them per world exactly as the
shared evaluator pools folds.

Selection objective (Rockafellar & Uryasev, Journal of Risk 2 (2000) 21-42,
applied to the loss tail of the per-world DTI distribution):

    score(candidate) = mean_w DTI_w  -  lam * (mean_w DTI_w - CVaR_q DTI_w)

where CVaR_q is the average of the worst q-fraction of worlds (low DTI = bad
tail).  lam = 0.5 by default; a sensitivity sweep is recorded in evidence.

Pruning: iteratively remove the dot whose removal most improves that
objective, recomputing the exact per-world terms after every removal
(top-2 credit bookkeeping per truth pixel, refreshed by a 29-offset rescan of
the surviving dots).  The loop stops when no removal improves the objective
by more than ``eps`` or the iteration cap is reached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from gems57.metric import ALPHA, BETA, EPS, kernel, OFF_DY, OFF_DX, OFF_K


def cvar(dtis: np.ndarray, q: float = 0.2) -> float:
    """Average of the worst ``q`` fraction of per-world DTI values."""
    d = np.asarray(dtis, np.float64)
    if d.size == 0:
        return 0.0
    k = max(1, int(np.ceil(q * d.size)))
    return float(np.sort(d)[:k].mean())


def objective(dtis: np.ndarray, lam: float = 0.5, q: float = 0.2) -> float:
    m = float(np.mean(dtis)) if len(dtis) else 0.0
    return m - lam * (m - cvar(dtis, q))


def _emit_crop_structures(cell, dots_full: np.ndarray):
    """Crop-local emission id array; returns (ids_crop, local_yx, n_emit)."""
    crop = dots_full[cell.bbox] & cell.active
    ys, xs = np.nonzero(crop)
    ids = np.full(crop.shape, -1, np.int64)
    ids[ys, xs] = np.arange(len(ys))
    return ids, (ys, xs), int(len(ys))


def world_terms(cells, worlds_by_cell, dots_full: np.ndarray) -> dict:
    """Per-world pooled metric terms for one candidate dot field.

    Returns dict with per-world dti array and pooled sums.  The emission's
    distance transform is computed once per cell (world-independent).
    """
    dots_full = np.asarray(dots_full, bool)
    n_worlds = len(next(iter(worlds_by_cell.values())))
    TP = np.zeros(n_worlds)
    FP = np.zeros(n_worlds)
    FN = np.zeros(n_worlds)
    NT = np.zeros(n_worlds)
    n_emit_total = 0
    for cell in cells:
        ids, (ys, xs), n_emit = _emit_crop_structures(cell, dots_full)
        n_emit_total += n_emit
        if n_emit:
            dp = ndi.distance_transform_edt(~(ids >= 0))
        for w, wc in enumerate(worlds_by_cell[cell.key]):
            ty, tx = wc.truth_yx
            NT[w] += wc.n_truth
            if wc.n_truth == 0:
                continue
            if n_emit:
                TP[w] += float(kernel(dp[ty, tx]).sum())
                dg = ndi.distance_transform_edt(~_truth_crop(cell, wc))
                kself = kernel(dg[ys, xs])
                FP[w] += float(n_emit - kself.sum())
            else:
                FP[w] += 0.0
            FN[w] += wc.n_truth - (float(kernel(dp[ty, tx]).sum()) if n_emit else 0.0)
    dtis = TP / (TP + ALPHA * FP + BETA * FN + EPS)
    return dict(dtis=dtis, tp=TP, fp=FP, fn=FN, n_truth=NT,
                n_emitted=n_emit_total, mean=float(dtis.mean()),
                cvar20=cvar(dtis, 0.2), objective=objective(dtis))


def _truth_crop(cell, wc) -> np.ndarray:
    t = np.zeros(cell.active.shape, bool)
    ty, tx = wc.truth_yx
    if len(ty):
        t[ty, tx] = True
    return t


# ---------------------------------------------------------------------------
# greedy CVaR pruning
# ---------------------------------------------------------------------------

@dataclass
class PruneState:
    cell: object
    worlds: list
    ids: np.ndarray                 # crop-local dot ids -> slot in cell dot list
    dot_yx: tuple
    n_emit: int
    alive: np.ndarray               # bool per local dot
    best1: list = field(default_factory=list)   # per world: float array (n_t,)
    best2: list = field(default_factory=list)
    arg1: list = field(default_factory=list)    # per world: int array (n_t,) local dot id
    arg2: list = field(default_factory=list)    # per world: int array (n_t,) local dot id
    kself: list = field(default_factory=list)   # per world: float array (n_local,)
    tp: list = field(default_factory=list)
    fp: list = field(default_factory=list)
    fn: list = field(default_factory=list)
    nt: list = field(default_factory=list)

    def top2_for_world(self, w: int, only: np.ndarray | None = None):
        """(Re)compute top-2 credits for world ``w`` (all truth px or ``only``).

        The offset scan reads the CURRENT ids array (dead dots carry -1), so a
        rescan always yields top-2 among the surviving dots only.  Invariant
        maintained by ``greedy_prune``: every truth pixel whose arg1 or arg2
        referenced a removed dot is rescanned immediately after the removal,
        so arg1/arg2 never point at dead dots.
        """
        wc = self.worlds[w]
        ty, tx = wc.truth_yx
        hgt, wdt = self.ids.shape
        if only is None:
            idx = np.arange(len(ty))
        else:
            idx = np.asarray(only)
        if idx.size == 0:
            return
        b1 = np.full(idx.size, -1.0)
        b2 = np.full(idx.size, -1.0)
        a1 = np.full(idx.size, -1, np.int64)
        a2 = np.full(idx.size, -1, np.int64)
        yy, xx = ty[idx], tx[idx]
        for dy, dx, kk in zip(OFF_DY, OFF_DX, OFF_K):
            ny, nx = yy + dy, xx + dx
            ok = (ny >= 0) & (ny < hgt) & (nx >= 0) & (nx < wdt)
            got = np.full(idx.size, -1, np.int64)
            vals = np.zeros(idx.size)
            if ok.any():
                idv = self.ids[ny[ok], nx[ok]]
                got[ok] = idv
                vals[ok] = np.where(idv >= 0, kk, 0.0)
            live = got >= 0
            better1 = live & (vals > b1)
            # demote old best to second
            b2[better1] = np.maximum(b2[better1], b1[better1])
            a2[better1] = np.where(b1[better1] > 0, a1[better1], -1)
            b1[better1] = vals[better1]
            a1[better1] = got[better1]
            better2 = live & ~better1 & (vals > b2)
            b2[better2] = vals[better2]
            a2[better2] = got[better2]
        self.best1[w][idx] = np.maximum(b1, 0.0)
        self.best2[w][idx] = np.maximum(b2, 0.0)
        self.arg1[w][idx] = a1
        self.arg2[w][idx] = a2


def prune_init(cell, worlds, dots_full: np.ndarray) -> PruneState:
    ids_crop, (ys, xs), n_emit = _emit_crop_structures(cell, dots_full)
    st = PruneState(cell=cell, worlds=worlds, ids=ids_crop, dot_yx=(ys, xs),
                    n_emit=n_emit, alive=np.ones(n_emit, bool))
    for w, wc in enumerate(worlds):
        ty, tx = wc.truth_yx
        st.best1.append(np.zeros(wc.n_truth))
        st.best2.append(np.zeros(wc.n_truth))
        st.arg1.append(np.full(wc.n_truth, -1, np.int64))
        st.arg2.append(np.full(wc.n_truth, -1, np.int64))
        st.top2_for_world(w)
        kself = np.zeros(n_emit)
        if wc.n_truth:
            dg = ndi.distance_transform_edt(~_truth_crop(cell, wc))
            if n_emit:
                kself = kernel(dg[ys, xs])
        st.kself.append(kself)
        tp = float(st.best1[w].sum())
        st.tp.append(tp)
        st.nt.append(float(wc.n_truth))
        st.fn.append(wc.n_truth - tp)
        st.fp.append(float(n_emit - kself.sum()))
    return st


def prune_global_objective(states: list[PruneState], lam: float, q: float):
    n_w = len(states[0].worlds)
    TP = np.array([sum(st.tp[w] for st in states) for w in range(n_w)])
    FP = np.array([sum(st.fp[w] for st in states) for w in range(n_w)])
    FN = np.array([sum(st.fn[w] for st in states) for w in range(n_w)])
    dtis = TP / (TP + ALPHA * FP + BETA * FN + EPS)
    return objective(dtis, lam, q), dtis


def greedy_prune(cells, worlds_by_cell, dots_full: np.ndarray, *,
                 lam: float = 0.5, q: float = 0.2, eps: float = 1e-6,
                 max_iter: int = 4000, verbose: bool = True,
                 verify_every: int = 500) -> dict:
    """Exact greedy CVaR pruning.  Returns final dots + trajectory.

    Every ``verify_every`` removals the internal bookkeeping is audited against
    a from-scratch EDT recomputation (``world_terms`` on the current dots);
    any drift beyond float tolerance fails closed.
    """
    dots_full = np.asarray(dots_full, bool).copy()
    states = [prune_init(cell, worlds_by_cell[cell.key], dots_full) for cell in cells]
    obj, dtis = prune_global_objective(states, lam, q)

    def _current_mask():
        m = dots_full.copy()
        for st in states:
            ys, xs = st.dot_yx
            dead = ~st.alive
            m[st.cell.bbox[0].start + ys[dead],
              st.cell.bbox[1].start + xs[dead]] = False
        return m

    def _audit(tag: str):
        wt = world_terms(cells, worlds_by_cell, _current_mask())
        drift = float(np.abs(wt["dtis"] - dtis_internal[0]).max()) if len(dtis_internal) else 0.0
        if drift > 5e-5:
            import json as _json
            n_w = len(states[0].worlds)
            TPi = np.array([sum(st.tp[w] for st in states) for w in range(n_w)])
            FPi = np.array([sum(st.fp[w] for st in states) for w in range(n_w)])
            FNi = np.array([sum(st.fn[w] for st in states) for w in range(n_w)])
            diag = dict(tag=tag, drift=drift,
                        tp_internal=TPi.tolist(), tp_recomputed=wt["tp"].tolist(),
                        fp_internal=FPi.tolist(), fp_recomputed=wt["fp"].tolist(),
                        fn_internal=FNi.tolist(), fn_recomputed=wt["fn"].tolist())
            import pathlib as _pl
            _pl.Path("evidence/prune_drift_diagnostic.json").write_text(
                _json.dumps(diag, indent=1))
            raise AssertionError(f"prune bookkeeping drift {drift:.3e} at {tag}; "
                                 f"diagnostic in evidence/prune_drift_diagnostic.json")
        if verbose:
            print(f"[prune] audit@{tag}: internal vs recomputed max drift {drift:.2e} OK", flush=True)

    dtis_internal = [dtis]
    _audit("init")
    trajectory = [dict(removed=0, objective=obj, mean=float(dtis.mean()),
                       cvar=cvar(dtis, q))]
    if verbose:
        print(f"[prune] start n={int(dots_full.sum())} obj={obj:.6f} "
              f"mean={dtis.mean():.6f} cvar={cvar(dtis, q):.6f}", flush=True)

    # ---- global pixel identity ------------------------------------------------
    # The same physical dot can be a member of several cell states (draw20 and
    # draw21 cells of one quadrant overlap).  A removal is a GLOBAL pixel
    # deletion and must be applied to every member state; otherwise one state
    # keeps crediting a dot that the shared raster has lost.
    state_gids = []
    pixel_to_gid: dict = {}
    for st in states:
        ys, xs = st.dot_yx
        gy = st.cell.bbox[0].start + ys
        gx = st.cell.bbox[1].start + xs
        keys = gy.astype(np.int64) * 100000 + gx.astype(np.int64)
        gids = np.empty(len(keys), np.int64)
        for i, k in enumerate(keys):
            k = int(k)
            gid = pixel_to_gid.get(k)
            if gid is None:
                gid = len(pixel_to_gid)
                pixel_to_gid[k] = gid
            gids[i] = gid
        state_gids.append(gids)
    n_g = len(pixel_to_gid)
    members: list[list] = [[] for _ in range(n_g)]
    for si, gids in enumerate(state_gids):
        for local, gid in enumerate(gids):
            members[int(gid)].append((si, local))
    alive_g = np.ones(n_g, bool)

    n_removed = 0
    for it in range(1, max_iter + 1):
        # pool the current per-world terms
        n_w = len(states[0].worlds)
        TP = np.array([sum(st.tp[w] for st in states) for w in range(n_w)])
        FP = np.array([sum(st.fp[w] for st in states) for w in range(n_w)])
        FN = np.array([sum(st.fn[w] for st in states) for w in range(n_w)])
        # per-global-pixel removal effects, accumulated across member states
        L = np.zeros((n_w, n_g))
        KFP = np.zeros((n_w, n_g))
        for si, st in enumerate(states):
            gids = state_gids[si]
            for w in range(n_w):
                if st.nt[w] > 0:
                    loss = np.bincount(st.arg1[w][st.arg1[w] >= 0],
                                       weights=(st.best1[w] - st.best2[w])[st.arg1[w] >= 0],
                                       minlength=st.n_emit)
                    L[w] += np.bincount(gids, weights=loss, minlength=n_g)
                KFP[w] += np.bincount(gids, weights=1.0 - st.kself[w], minlength=n_g)
        # exact per-pixel hypothetical DTI after removal, vectorised over worlds
        Tp = TP[:, None] - L
        Fp = FP[:, None] - KFP
        Fn = FN[:, None] + L
        dti_p = Tp / (Tp + ALPHA * Fp + BETA * Fn + EPS)
        m = dti_p.mean(axis=0)
        # CVaR over worlds per dot: mean of worst ceil(q*W) worlds
        k_tail = max(1, int(np.ceil(q * n_w)))
        tail = np.partition(dti_p, k_tail - 1, axis=0)[:k_tail].mean(axis=0)
        objv = m - lam * (m - tail)
        objv = np.where(alive_g, objv, -np.inf)
        best = int(np.argmax(objv))
        if not np.isfinite(objv[best]) or objv[best] - obj <= eps:
            break
        if not alive_g[best]:
            break
        alive_g[best] = False
        # apply the global deletion to every member state
        for si, local in members[best]:
            st = states[si]
            if not st.alive[local]:
                continue
            st.alive[local] = False
            y, x = st.dot_yx[0][local], st.dot_yx[1][local]
            st.ids[y, x] = -1
            for w in range(n_w):
                if st.nt[w]:
                    affected = np.nonzero((st.arg1[w] == local) |
                                          (st.arg2[w] == local))[0]
                else:
                    affected = np.zeros(0, np.int64)
                old = st.best1[w][affected].copy() if affected.size else np.zeros(0)
                if affected.size:
                    st.top2_for_world(w, only=affected)
                st.tp[w] += float(st.best1[w][affected].sum() - old.sum()) if affected.size else 0.0
                st.fn[w] = st.nt[w] - st.tp[w]
                st.fp[w] -= 1.0 - float(st.kself[w][local])
        n_removed += 1
        obj, dtis = prune_global_objective(states, lam, q)
        dtis_internal[0] = dtis
        if verify_every and n_removed % verify_every == 0:
            _audit(f"removed={n_removed}")
        if verbose and (it % 50 == 0 or it <= 5):
            print(f"[prune] iter={it} removed={n_removed} obj={obj:.6f} "
                  f"mean={dtis.mean():.6f} cvar={cvar(dtis, q):.6f}", flush=True)
        trajectory.append(dict(removed=n_removed, objective=obj,
                               mean=float(dtis.mean()), cvar=cvar(dtis, q)))
    _audit("final")
    final = dots_full.copy()
    for si, st in enumerate(states):
        ys, xs = st.dot_yx
        dead = ys[~st.alive], xs[~st.alive]
        bbox = st.cell.bbox
        gy = bbox[0].start + dead[0]
        gx = bbox[1].start + dead[1]
        final[gy, gx] = False
    return dict(dots=final, n_removed=n_removed, objective_start=trajectory[0]["objective"],
                objective_end=obj, mean_end=float(dtis.mean()), cvar_end=cvar(dtis, q),
                trajectory=trajectory, lam=lam, q=q, eps=eps)
