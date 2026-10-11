"""Label-world DTI instrument: exact metric in every world + CVaR selection + exact greedy pruning.

For each holdout cell we precompute, per label world, the max-kernel credit map
kmax(x) = max_{g in G_w} k(d(x, g)) sampled at every universe dot. Then:

- FP of any binary emission S in world w = sum_{x in S} (1 - kmax_w(x))   [exact, linear]
- TP of a binary emission S in world w = sum_{g in G_w} k(d(g, S)) via an EDT  [exact]

DTI is pooled per cell-world exactly as on the leaderboard
(DTI = TP_w / (TP_w + 0.2*FP_w + 0.8*FN_w)).

Robust objective (Rockafellar-Uryasev mean-CVaR blend on the gain):
  J(S) = (1-lam) * mean_w DTI_w + lam * (mean of the worst q tail of DTI)
with lam = 0.5, q = 0.20  (worst-20% mean = lower CVaR of the gain at level q).

Greedy pruning is EXACT: for the incumbent emission we maintain, per cell-world
and per truth pixel, the best credit, its provider dot and the second-best
credit. Removing a batch B changes TP exactly through these top-two structures;
every proposed batch is additionally re-verified with a full provider rescan
before commitment, so all committed step scores are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .metric import ALPHA, BETA, EPS, _OFFS, dti_from_counts, kernel

N_WORLDS = 20
CVAR_LAM = 0.5
CVAR_Q = 0.20


@dataclass
class CellWorldData:
    cell: object
    worlds: list                       # list[World]
    cidx: np.ndarray                   # global universe indices of dots inside bbox & active
    emap: np.ndarray                   # (bbox H x bbox W) int32: global universe index or -1
    kvals: np.ndarray                  # (n_worlds, len(cidx)) float32: kmax_w at each universe dot
    baseline_kvals: np.ndarray         # (len(cidx),) float32 for the unperturbed hidden truth
    baseline_truth: tuple              # (rows, cols) unperturbed hidden truth in active


@dataclass
class Universe:
    pos_flat: np.ndarray               # (N,) int64 flat full-grid positions of universe dots
    N: int
    grid_shape: tuple


def build_universe(candidate_masks: list, grid_shape: tuple) -> Universe:
    """Union of all candidate emission supports."""
    u = np.zeros(grid_shape, dtype=bool)
    for m in candidate_masks:
        u |= m
    pos = np.flatnonzero(u.ravel()).astype(np.int64)
    return Universe(pos_flat=pos, N=int(pos.size), grid_shape=grid_shape)


def _kmax_grid(truth_bool: np.ndarray) -> np.ndarray:
    """kmax(x) = max_g k(d(x,g)) built by the 25-offset shift trick (exact inside bbox)."""
    H, W = truth_bool.shape
    out = np.zeros((H, W), dtype=np.float32)
    for dy, dx, k in _OFFS:
        ys = slice(max(0, dy), H + min(0, dy))
        xs = slice(max(0, dx), W + min(0, dx))
        yd = slice(max(0, -dy), H + min(0, -dy))
        xd = slice(max(0, -dx), W + min(0, -dx))
        if k >= 1.0:
            out[yd, xd] = np.maximum(out[yd, xd], truth_bool[ys, xs].astype(np.float32))
        else:
            np.maximum(
                out[yd, xd],
                truth_bool[ys, xs].astype(np.float32) * np.float32(k),
                out=out[yd, xd],
            )
    return out


def prepare_cell_worlds(cell, worlds, universe: Universe, grid_shape) -> CellWorldData:
    _, Wtot = grid_shape
    sl = cell.bbox
    r0, c0 = sl[0].start, sl[1].start
    Hb, Wb = sl[0].stop - r0, sl[1].stop - c0
    emap = np.full((Hb, Wb), -1, dtype=np.int32)
    rows = (universe.pos_flat // Wtot).astype(np.int64)
    cols = (universe.pos_flat % Wtot).astype(np.int64)
    inb = (rows >= r0) & (rows < sl[0].stop) & (cols >= c0) & (cols < sl[1].stop)
    gidx = np.flatnonzero(inb)
    br = rows[inb] - r0
    bc = cols[inb] - c0
    active_ok = cell.active_sub[br, bc]
    gidx = gidx[active_ok]
    br = br[active_ok]
    bc = bc[active_ok]
    emap[br, bc] = gidx.astype(np.int32)

    # unperturbed hidden truth (the template's catalogue-hidden world) for the baseline
    bt = np.zeros((Hb, Wb), dtype=bool)
    for rr, cc in cell.hidden_comp_coords:
        keep = cell.active_sub[rr, cc]
        bt[rr[keep], cc[keep]] = True
    base_k = _kmax_grid(bt)
    baseline_kvals = base_k[br, bc].astype(np.float32)
    brr, bcc = np.nonzero(bt)

    kvals = np.zeros((len(worlds), gidx.size), dtype=np.float32)
    for wi, w in enumerate(worlds):
        tg = np.zeros((Hb, Wb), dtype=bool)
        rr, cc = w.truth_coords
        tg[rr, cc] = True
        kg = _kmax_grid(tg)
        kvals[wi] = kg[br, bc]

    return CellWorldData(
        cell=cell,
        worlds=worlds,
        cidx=gidx.astype(np.int32),
        emap=emap,
        kvals=kvals,
        baseline_kvals=baseline_kvals,
        baseline_truth=(brr.astype(np.int32), bcc.astype(np.int32)),
    )


def _tp_via_edt(sub_mask: np.ndarray, truth_coords: tuple) -> float:
    rr, cc = truth_coords
    if rr.size == 0 or not sub_mask.any():
        return 0.0
    dp = distance_transform_edt(~sub_mask)
    return float(kernel(dp[rr, cc]).sum())


def evaluate_emission(mask: np.ndarray, cwds: list) -> dict:
    """Full instrument for a binary emission grid mask: world DTIs, J, CI, baseline."""
    all_dtis = []
    all_tp = []
    all_fp = []
    all_n = []
    domain_shares = []
    cell_means = []
    baselines = []
    n_emitted_int_domains = 0
    for cwd in cwds:
        cell = cwd.cell
        sl = cell.bbox
        sub = mask[sl] & cell.active_sub
        n_emitted_int_domains += int(sub.sum())
        # FP per world: sample emission membership through emap
        present = np.zeros(int(cwd.emap.max()) + 2, dtype=bool)
        gi = cwd.emap[sub]
        present[gi[gi >= 0]] = True
        Sdot = present[cwd.cidx]
        fp = ((1.0 - cwd.kvals) * Sdot[None, :]).sum(axis=1)
        if sub.any():
            dp = distance_transform_edt(~sub)
        else:
            dp = None
        dtis = np.zeros(len(cwd.worlds), dtype=np.float64)
        tps = np.zeros(len(cwd.worlds), dtype=np.float64)
        ns = np.zeros(len(cwd.worlds), dtype=np.float64)
        for wi, w in enumerate(cwd.worlds):
            rr, cc = w.truth_coords
            n_t = int(rr.size)
            ns[wi] = n_t
            if n_t == 0:
                dtis[wi] = 0.0
                continue
            tp = float(kernel(dp[rr, cc]).sum()) if dp is not None else 0.0
            tps[wi] = tp
            dtis[wi] = dti_from_counts(tp, float(fp[wi]), n_t)
        all_tp.append(tps)
        all_fp.append(fp.astype(np.float64))
        all_n.append(ns)
        domain_shares.append(float(cwd.cell.n_domain_px))
        rr, cc = cwd.baseline_truth
        n_b = int(rr.size)
        if n_b > 0 and dp is not None:
            tp_b = float(kernel(dp[rr, cc]).sum())
            fp_b = float(((1.0 - cwd.baseline_kvals) * Sdot).sum())
            baselines.append(dti_from_counts(tp_b, fp_b, n_b))
        else:
            baselines.append(0.0)
        all_dtis.append(dtis)
        cell_means.append(float(dtis.mean()) if dtis.size else 0.0)
    v = np.concatenate(all_dtis) if all_dtis else np.zeros(1)
    k = max(1, int(np.ceil(CVAR_Q * v.size)))
    worst = np.sort(v)[:k]
    j = (1 - CVAR_LAM) * float(v.mean()) + CVAR_LAM * float(worst.mean())
    n = v.size
    ci = 1.96 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    out = {
        "J": j,
        "mean_dti": float(v.mean()),
        "cvar_worst20": float(worst.mean()),
        "std": float(v.std(ddof=1)) if n > 1 else 0.0,
        "ci95": ci,
        "n_cell_worlds": n,
        "emitted_in_union_domains": n_emitted_int_domains,
        "cell_mean_dtis": cell_means,
        "baseline_catalogue_hidden_mean": float(np.mean(baselines)),
        "baseline_per_cell": baselines,
        "world_dtis": v,
        "world_tp": np.concatenate(all_tp) if all_tp else np.zeros(1),
        "world_fp": np.concatenate(all_fp) if all_fp else np.zeros(1),
        "world_n": np.concatenate(all_n) if all_n else np.ones(1),
        "domain_share": domain_shares,
    }
    return out


def j_from_dti(v: np.ndarray) -> tuple:
    """Rockafellar-Uryasev mean-CVaR blend on the DTI gain vector."""
    k = max(1, int(np.ceil(CVAR_Q * v.size)))
    worst = np.sort(v)[:k]
    return (1 - CVAR_LAM) * float(v.mean()) + CVAR_LAM * float(worst.mean()), float(v.mean()), float(worst.mean())


def prevalence_scale_factors(world_n: np.ndarray, cwds: list, G_total: float) -> np.ndarray:
    """Per-cell calibrated truth population G_cell = G_total * domain share; each
    world's credits scale by s = G_cell / n_world so a world's hit RATE is kept
    but the truth POPULATION matches the receipt-calibrated hidden size
    (12,691 px; GEMSDOE32 evidence/instrument_calibration.json, calibrated from
    organizer-confirmed live receipts of the blind-lattice and d1.5/d2.8 chain).
    """
    n_cells = len(cwds)
    n_w = len(cwds[0].worlds) if cwds else 0
    dom = np.array([c.cell.n_domain_px for c in cwds], dtype=np.float64)
    shares = dom / dom.sum()
    s = np.zeros_like(world_n)
    for ci in range(n_cells):
        G_cell = G_total * shares[ci]
        sl = slice(ci * n_w, (ci + 1) * n_w)
        nw = world_n[sl]
        s[sl] = np.where(nw > 0, G_cell / np.maximum(nw, 1.0), 0.0)
    return s


def dti_prevalence_calibrated(world_tp, world_fp, world_n, s) -> np.ndarray:
    """Exact-metric DTI with truth population rescaled by s (hit rate kept)."""
    tp = world_tp * s
    n_cal = world_n * s
    fn = n_cal - tp
    out = tp / (tp + ALPHA * world_fp + BETA * fn + EPS)
    return np.where(world_n > 0, out, 0.0)


# ------------- provider structures for exact incremental pruning ----------------

@dataclass
class ProviderState:
    prov: np.ndarray       # (n_truth,) int32 global universe index of best provider, -1 if none
    best: np.ndarray       # (n_truth,) float64
    second: np.ndarray     # (n_truth,) float64
    n_truth: int


def provider_scan(cwd: CellWorldData, S_global: np.ndarray, truth_coords: tuple) -> ProviderState:
    """Top-two credit structure per truth pixel for emission S within this cell.

    S_global indexes the global universe. Only exact top-two credits are kept.
    """
    rr, cc = truth_coords
    n = int(rr.size)
    prov = np.full(n, -1, dtype=np.int64)
    best = np.zeros(n, dtype=np.float64)
    second = np.zeros(n, dtype=np.float64)
    emap = cwd.emap
    Hb, Wb = emap.shape
    for dy, dx, k in _OFFS:
        nr = rr + dy
        nc = cc + dx
        ok = (nr >= 0) & (nr < Hb) & (nc >= 0) & (nc < Wb)
        if not ok.any():
            continue
        idx = emap[nr[ok], nc[ok]]
        has = idx >= 0
        if not has.any():
            continue
        pos = np.flatnonzero(ok)[has]
        gi = idx[has].astype(np.int64)
        present = S_global[gi]
        if not present.any():
            continue
        pos = pos[present]
        gi = gi[present]
        b = best[pos]
        s = second[pos]
        is_best = k > b
        new_second = np.where(is_best, np.maximum(s, b), np.maximum(s, k))
        new_best = np.where(is_best, k, b)
        best[pos] = new_best
        second[pos] = new_second
        prov[pos] = np.where(is_best, gi, prov[pos])
    return ProviderState(prov=prov, best=best, second=second, n_truth=n)
