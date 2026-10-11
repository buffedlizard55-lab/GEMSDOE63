"""Lane 63 emission policies on the *visible* catalogue.

Every policy emits binary dots derived ONLY from the visible (non-withheld)
catalogue geometry, so a hide-and-recover fold can never leak withheld truth
into the emission (parallel-run protocol rule 2).

Policies
--------
* ``dotted``        -- arc-length-spaced dots along ordered catalogue paths
                       (spacing ``d`` px), optional lateral ribbon lanes of
                       width ``r`` lanes, optional tip extensions ``t`` px,
                       optional stepover bridges across collinear endpoint
                       gaps <= ``gap`` px (the h33-analog mechanism).
* ``budget``        -- keep only the longest components until the requested
                       dot-pixel budget is spent (rank = component length).
* ``rad_ratio``     -- radiometric K/Th-ratio maxima candidate (hypothesis
                       H63-B), only used when band provenance is confirmed.

Nothing here reads the holdout truth.  Dots on visible catalogue pixels are
removed at the end (the writer also rejects them; belt and braces).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

STRUCT3 = np.ones((3, 3), bool)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    spacing: float = 2.8
    ribbon: int = 0          # number of lateral lanes per side (0 or 1)
    tips: int = 0            # px of tip extension beyond component ends
    stepover_gap: float = 0.0   # 0 disables bridging; else max gap in px
    budget_frac: float = 1.0    # fraction of component pixel mass kept
    kind: str = "dotted"
    seed: int = 63

    def short(self) -> str:
        bits = [self.kind, f"d{self.spacing:g}"]
        if self.ribbon:
            bits.append(f"r{self.ribbon}")
        if self.tips:
            bits.append(f"t{self.tips}")
        if self.stepover_gap > 0:
            bits.append(f"so{self.stepover_gap:g}")
        if self.budget_frac < 1.0:
            bits.append(f"b{int(round(self.budget_frac * 100)):03d}")
        return "-".join(bits)


def _ordered_paths(comp: np.ndarray, n_comp: int):
    """Nearest-neighbour arc ordering per component (thin 1-px traces)."""
    H, W = comp.shape
    flat = comp.ravel()
    order = np.argsort(flat, kind="stable")
    ids_sorted = flat[order]
    bounds = np.searchsorted(ids_sorted, np.arange(1, n_comp + 2))
    paths = []
    for i in range(n_comp):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if b <= a:
            continue
        idx = order[a:b]
        y, x = idx // W, idx % W
        cid = i + 1
        if b - a == 1:
            paths.append((cid, np.array([(y[0], x[0])], dtype=np.int64), 1))
            continue
        # start at the most north-west pixel; walk to nearest unvisited.
        lex = np.lexsort((x, y))
        y, x = y[lex], x[lex]
        n = len(y)
        pos = np.stack([y, x], 1).astype(np.int64)
        used = np.zeros(n, bool)
        path = [0]
        used[0] = True
        cur = pos[0]
        for _ in range(n - 1):
            d2 = ((pos - cur) ** 2).sum(1) + used * 10 ** 9
            j = int(d2.argmin())
            path.append(j)
            used[j] = True
            cur = pos[j]
        paths.append((cid, pos[np.array(path)], n))
    return paths


def dotted_ridge(visible: np.ndarray, footprint: np.ndarray, spec: PolicySpec,
                 return_diagnostics: bool = False) -> np.ndarray:
    """Ring-lane dotted ridge: arc-length-spaced dots on the 1-px offset ring.

    Dots are placed on the BOTH-SIDES perpendicular ring one pixel off the
    trace (the damage-zone lane), never on catalogue pixels.  This is the
    geometry that reproduces the observed dot counts of the d2.8 family
    (~2 x catalogue_length / d); the on-trace placement would be rejected by
    the submission writer because dots may not sit on the mapped catalogue.
    ``ribbon`` adds a second ring at 2 px offset on both sides.
    """
    vis = np.asarray(visible, bool) & np.asarray(footprint, bool)
    comp, n_comp = ndi.label(vis, STRUCT3)
    paths = _ordered_paths(comp, n_comp)

    # Budget: rank components by pixel length, keep top fraction of mass.
    total_mass = sum(p[2] for p in paths)
    target = spec.budget_frac * total_mass
    paths_sorted = sorted(paths, key=lambda p: -p[2])
    kept, acc = [], 0.0
    for cid, pos, size in paths_sorted:
        if acc >= target:
            break
        kept.append((cid, pos, size))
        acc += size
    paths = kept

    out = np.zeros(vis.shape, bool)
    diag_bridges = 0

    def _stamp(y, x):
        if 0 <= y < out.shape[0] and 0 <= x < out.shape[1] and footprint[y, x]:
            out[y, x] = True

    def _perp(pos, i):
        n = len(pos)
        j0 = max(0, i - 1)
        j1 = min(n - 1, i + 1)
        dy, dx = float(pos[j1][0] - pos[j0][0]), float(pos[j1][1] - pos[j0][1])
        norm = (dy * dy + dx * dx) ** 0.5 or 1.0
        return dx / norm, -dy / norm   # perpendicular unit vector (row, col)

    lane_widths = [1] + ([2] if spec.ribbon else [])
    for cid, pos, size in paths:
        n = len(pos)
        for lw in lane_widths:
            acc_len = 0.0
            for i in range(n):
                if i > 0:
                    acc_len += float(np.hypot(*(pos[i] - pos[i - 1])))
                if i == 0 or acc_len >= spec.spacing - 1e-9:
                    py, px = _perp(pos, i)
                    for side in (-1, 1):
                        _stamp(int(round(pos[i][0] + side * lw * py)),
                               int(round(pos[i][1] + side * lw * px)))
                    acc_len = 0.0
        # tip extensions along the extrapolated end direction (both sides ring)
        if spec.tips and n >= 2:
            for i0, i1 in ((0, 1), (n - 1, n - 2)):
                dy, dx = float(pos[i0][0] - pos[i1][0]), float(pos[i0][1] - pos[i1][1])
                norm = (dy * dy + dx * dx) ** 0.5 or 1.0
                uy, ux = dy / norm, dx / norm
                py, px = ux, -uy
                for k in range(1, spec.tips + 1):
                    for side in (-1, 1):
                        _stamp(int(round(pos[i0][0] + k * uy + side * py)),
                               int(round(pos[i0][1] + k * ux + side * px)))

    # stepover bridges between collinear endpoint pairs
    if spec.stepover_gap > 0.0:
        ends = []
        for cid, pos, size in paths:
            if len(pos) >= 3:
                ends.append((cid, pos[0], pos[1]))
                ends.append((cid, pos[-1], pos[-2]))
        for i in range(len(ends)):
            cid_a, pa, qa = ends[i]
            for j in range(i + 1, len(ends)):
                cid_b, pb, qb = ends[j]
                if cid_a == cid_b:
                    continue
                gap = pb - pa
                g = float(np.hypot(*gap))
                if not (1.0 < g <= spec.stepover_gap):
                    continue
                va = (pa - qa).astype(float)
                vb = (pb - qb).astype(float)
                na = np.linalg.norm(va) or 1.0
                nb = np.linalg.norm(vb) or 1.0
                cos_a = float(np.dot(va / na, gap / g))
                cos_b = float(np.dot(vb / nb, -gap / g))
                if cos_a < 0.82 or cos_b < 0.82:   # ~35 deg collinearity gate
                    continue
                n_steps = max(1, int(round(g / 1.4)))
                for k in range(1, n_steps):
                    f = k / n_steps
                    _stamp(int(round(pa[0] + f * gap[0])),
                           int(round(pa[1] + f * gap[1])))
                diag_bridges += 1

    out &= ~vis & footprint
    if return_diagnostics:
        return out, dict(n_components_kept=len(paths), bridges=diag_bridges,
                         dots=int(out.sum()))
    return out


def rad_ratio_maxima(rad_path: str, footprint: np.ndarray, catalogue: np.ndarray,
                     min_off_cat_px: int = 1, n_dots: int = 30000,
                     k_idx: int | None = None, th_idx: int | None = None):
    """Hypothesis H63-B: dots on local maxima of the K/Th ratio z-score.

    Returns ``(dots, receipt)``; ``receipt['usable']`` is False when the band
    descriptions do not unambiguously identify K and Th channels (no guessing).
    """
    import rasterio
    receipt = dict(usable=False, bands=None, reason="")
    with rasterio.open(rad_path) as ds:
        desc = [str(d).lower() if d else "" for d in ds.descriptions]
        bands = ds.read()
    receipt["bands"] = desc
    if k_idx is None or th_idx is None:
        k_idx = next((i for i, d in enumerate(desc) if "k" in d.split() or d.startswith("k") or "potassium" in d), None)
        th_idx = next((i for i, d in enumerate(desc) if "th" in d.split() or d.startswith("th") or "thorium" in d), None)
    if k_idx is None or th_idx is None:
        receipt["reason"] = "band descriptions do not identify K and Th channels"
        return np.zeros(footprint.shape, bool), receipt
    k = bands[k_idx].astype(np.float64)
    th = bands[th_idx].astype(np.float64)
    valid = footprint & (th > 3.0) & (k > 0.0)   # avoid div0 / noise floor
    ratio = np.zeros_like(k)
    ratio[valid] = np.log1p(k[valid]) / np.log1p(th[valid])
    mu = float(ratio[valid].mean())
    sd = float(ratio[valid].std()) or 1.0
    z = np.where(valid, (ratio - mu) / sd, -9.0)
    zmax = ndi.maximum_filter(z, size=7)
    peaks = valid & (z == zmax) & (z >= 1.5)
    keep = ndi.distance_transform_edt(~catalogue) >= min_off_cat_px
    peaks &= keep & footprint
    ys, xs = np.nonzero(peaks)
    if len(ys) > n_dots:
        order = np.argsort(-z[peaks])[:n_dots]
        sel = np.zeros(len(ys), bool)
        sel[order] = True
        out = np.zeros(peaks.shape, bool)
        out[ys[sel], xs[sel]] = True
    else:
        out = peaks
    receipt.update(usable=True, k_band=int(k_idx), th_band=int(th_idx),
                   peaks=int(out.sum()), z_threshold=1.5, filter="7px max-filter")
    return out, receipt
