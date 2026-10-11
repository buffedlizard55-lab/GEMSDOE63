"""Label-world ensemble built on the hide-and-recover holdout (robust-selection lane).

For fold k, the withheld catalogue segments are the 'observed' truth. Plausible label worlds
perturb that truth the way the organisers' future label revision could:
  * SHIFT  - every kept segment is rigidly translated by integer (dy, dx) in [-4, 4] px
             (max-norm 400 m). Calibration source: Hermant et al. (2025), Stanford GRC workshop
             paper, reports USGS-vs-TLS fault-label distances "up to 400m" in north-central
             Nevada: https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2025/Hermant.pdf
  * DROP   - a fraction U(0.10, 0.30) of withheld segments is removed from the truth
             (omission / not confirmed as faults).
  * DECOY  - straight decoy segments are added with total length U(0.5, 1.0) x the withheld
             truth length, placed at random outside 3 px of any catalogue pixel, with the
             observed segment-length distribution. (Decoy density is an ASSUMPTION: the
             organisers' new-fault density is not published.)
Truth in each world is restricted to the scored domain (footprint and NOT visible catalogue).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .grid import Grid, withheld_mask, visible_mask, clearance_zone

SHIFT_PX = 4
DROP_RANGE = (0.10, 0.30)
DECOY_RANGE = (0.5, 1.0)


class FoldSegments:
    """Pixel coordinates of each withheld segment of fold k, plus scored-domain masks."""

    def __init__(self, g: Grid, k: int):
        wh = withheld_mask(g, k)
        self.k = k
        self.visible = visible_mask(g, k)
        self.active = g.footprint & ~self.visible          # scored domain for fold k
        self.clear = clearance_zone(g)                       # decoys avoid 3 px of ANY catalogue
        seg = g.segments * wh
        ids = np.unique(seg[seg > 0])
        self.ids = ids
        objs = ndimage.find_objects(seg)
        self.coords = []
        for i in ids:
            sl = objs[i - 1]
            sub = seg[sl] == i
            rr, cc = np.nonzero(sub)
            self.coords.append(np.stack([rr + sl[0].start, cc + sl[1].start], axis=1))
        self.lengths = np.array([c.shape[0] for c in self.coords], dtype=int)
        self.n_truth_px = int(self.lengths.sum())
        self.shape = g.shape


def _rasterize_line(r0, c0, theta, length, shape):
    t = np.arange(0.0, length, 0.5)
    rr = np.round(r0 + t * np.sin(theta)).astype(int)
    cc = np.round(c0 + t * np.cos(theta)).astype(int)
    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return rr[ok], cc[ok]


def make_world(fs: FoldSegments, rng: np.random.Generator) -> np.ndarray:
    """Return (n, 2) int pixel coordinates of the scored truth in one perturbed label world."""
    n = len(fs.coords)
    drop_frac = rng.uniform(*DROP_RANGE)
    n_drop = int(round(drop_frac * n))
    dropped = set(rng.choice(n, size=n_drop, replace=False).tolist()) if n_drop else set()
    H, W = fs.shape
    pts = []
    for i, c in enumerate(fs.coords):
        if i in dropped:
            continue
        dy, dx = rng.integers(-SHIFT_PX, SHIFT_PX + 1, size=2)
        s = c + np.array([dy, dx])
        ok = (s[:, 0] >= 0) & (s[:, 0] < H) & (s[:, 1] >= 0) & (s[:, 1] < W)
        pts.append(s[ok])
    target = int(round(rng.uniform(*DECOY_RANGE) * fs.n_truth_px))
    added = 0
    tries = 0
    lens = fs.lengths
    while added < target and tries < 100_000:
        tries += 1
        L = int(rng.choice(lens))
        r0 = rng.integers(0, H)
        c0 = rng.integers(0, W)
        if not fs.active[r0, c0] or fs.clear[r0, c0]:
            continue
        rr, cc = _rasterize_line(r0, c0, rng.uniform(0, np.pi), L, fs.shape)
        if rr.size == 0:
            continue
        keep = fs.active[rr, cc] & ~fs.clear[rr, cc]
        rr, cc = rr[keep], cc[keep]
        if rr.size == 0:
            continue
        pts.append(np.stack([rr, cc], axis=1))
        added += rr.size
    if not pts:
        return np.zeros((0, 2), dtype=int)
    allp = np.unique(np.concatenate(pts, axis=0), axis=0)
    ok = fs.active[allp[:, 0], allp[:, 1]]
    return allp[ok]


def nominal_truth(fs: FoldSegments) -> np.ndarray:
    """Unperturbed observed truth for fold k (withheld segments inside the scored domain)."""
    allp = np.concatenate(fs.coords, axis=0) if fs.coords else np.zeros((0, 2), int)
    return allp[fs.active[allp[:, 0], allp[:, 1]]] if allp.size else allp
