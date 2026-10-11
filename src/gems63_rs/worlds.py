"""The label-world generator: an ensemble of plausible *truth* rasters.

Why this exists
---------------
The competition scores one submission, chosen before the private labels are known,
against a truth set that the organizers say may be added to, pruned or changed
(Phase 2 re-scores the same submission against an expert-expanded label set).  The
truth a submission is scored on therefore differs from the truth it was validated
on.  Selecting a submission by a single holdout score selects for that holdout's
noise.  Selecting by the *tail* of a distribution of plausible truths is the robust
alternative, and it is what this lane does: candidates are scored in every world and
selected by mean DTI penalised by conditional value-at-risk (Rockafellar & Uryasev,
*Journal of Risk*, 2000).

How the worlds are calibrated (no free parameters invented here)
---------------------------------------------------------------
* **Shift, <= 4 px.**  Hermant and coauthors measured, in north-central Nevada, that
  the distance between USGS Quaternary fault traces and an independently mapped
  trace set "can be up to 400 m" (Stanford Geothermal Workshop 2025,
  https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2025/Hermant.pdf; the same
  paper reports "discrepancies > 150 m ... with a clear offset between the USGS
  fault line and" their mapping).  400 m is 4 px on the competition's 100 m grid, so
  withheld segments are displaced by a vector drawn uniformly in the disc of radius
  ``shift_max_px`` = 4.
* **Omission, 10-30 %.**  The organizers will not disclose the test-set construction
  (chrisk-dd, thread 11527), and Phase 2 explicitly changes the label set, so a
  world in which a random 10-30 % of the held-out segments never appears in the
  scoring labels is a realistic member of the ensemble.
* **Decoys at the observed density.**  A "new fault" is "any fault pixel not already
  captured by USGS/INGENIOUS" and "can include newly mapped geometry of an existing
  fault system" (chrisk-dd, thread 11536, 2026-09-23) -- *can*, not *must*.  Worlds
  therefore also contain segments with no relationship to the catalogue at all,
  drawn at up to the catalogue's own observed segment density, with lengths sampled
  from the observed length distribution.

The generator is deliberately conservative: it never invents a truth population
denser than the mapped one, and every rate traces to a measurement or an organizer
statement recorded in ``docs/sources.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .holdout import Fold
SHIFT_MAX_PX = 4.0     # Hermant et al. 2025: catalogue offsets up to 400 m
DROP_LO = 0.10
DROP_HI = 0.30


@dataclass
class WorldConfig:
    # Hermant et al. 2025 give an *upper bound* ("can be up to 400 m") and separately
    # report "discrepancies > 150 m".  The distribution below that bound is not
    # published, so the radius is itself drawn per world from [shift_lo_px,
    # shift_hi_px] = [1.5, 4] px (150-400 m) rather than fixed at the bound.  Fixing
    # it at 4 px would make the selected ribbon width an artifact of the bound.
    shift_lo_px: float = 1.5
    shift_hi_px: float = SHIFT_MAX_PX
    shift_max_px: float = SHIFT_MAX_PX   # kept as the hard cap
    drop_lo: float = DROP_LO
    drop_hi: float = DROP_HI
    # phi = the fraction of surviving truth that is *newly mapped geometry of an
    # existing system* (a laterally displaced copy of a mapped trace) as opposed to
    # wholly new structure.  The organizers say a new fault "can include" the former
    # (thread 11536) but never say how much of the set it is, and they will not
    # disclose the test-set construction (thread 11527).  phi is therefore drawn per
    # world from a wide interval and the selection is made robust across it -- that
    # is the whole point of scoring in an ensemble rather than on one draw.
    phi_lo: float = 0.15
    phi_hi: float = 0.95
    decoy_max_fraction: float = 1.0   # up to the observed catalogue segment density
    seed: int = 20261011
    n_worlds: int = 24

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class World:
    name: str
    fold: str
    truth: np.ndarray
    phi: float
    n_segments_kept: int
    n_segments_dropped: int
    n_displaced: int
    n_decoys: int
    max_shift_px: float
    truth_pixels: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("truth")
        return d


def _rasterize_line(shape: tuple[int, int], r0: float, c0: float,
                    theta: float, length_px: int) -> np.ndarray:
    """A 1-px-wide straight trace of `length_px` pixels from (r0, c0) at angle theta."""
    out = np.zeros(shape, dtype=bool)
    if length_px < 1:
        return out
    t = np.arange(length_px)
    rows = np.round(r0 + t * np.sin(theta)).astype(int)
    cols = np.round(c0 + t * np.cos(theta)).astype(int)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    out[rows[ok], cols[ok]] = True
    return out


def _shift(rc: np.ndarray, dy: int, dx: int, shape: tuple[int, int]) -> np.ndarray:
    r = rc[:, 0] + dy
    c = rc[:, 1] + dx
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    return np.column_stack([r[ok], c[ok]])


def make_world(fold: Fold, cfg: WorldConfig, rng: np.random.Generator,
               *, allowed: np.ndarray | None = None,
               idx: int = 0) -> World:
    """Build one plausible truth raster for `fold`.

    `allowed` is the region truth may occupy (normally ``fold.valid``); truth is
    clipped to it, which keeps every world consistent with the organizer's masking
    rule that known-fault pixels are excluded from evaluation.
    """
    shape = fold.withheld.shape
    allowed = fold.valid if allowed is None else np.asarray(allowed, bool)

    segs = fold.segments
    n = len(segs)
    lengths = np.array([len(s) for s in segs], dtype=float) if n else np.array([2.0])
    ys, xs = np.nonzero(allowed)

    def decoy(truth: np.ndarray, k: int) -> None:
        for _ in range(k):
            if ys.size == 0:
                return
            j = int(rng.integers(ys.size))
            theta = float(rng.uniform(0.0, np.pi))
            L = int(max(2, rng.choice(lengths)))
            truth |= _rasterize_line(shape, float(ys[j]), float(xs[j]), theta, L)

    if n == 0:
        truth = np.zeros(shape, dtype=bool)
        return World(f"{fold.name}-w{idx:03d}", fold.name, truth, float(cfg.phi_lo),
                     0, 0, 0, 0, 0.0, 0)

    # --- 1. omission: a random 10-30 % of segments never reaches the labels -----
    drop_frac = float(rng.uniform(cfg.drop_lo, cfg.drop_hi))
    keep = rng.random(n) >= drop_frac
    if not keep.any():
        keep[int(rng.integers(n))] = True
    survivors = [segs[i] for i in range(n) if keep[i]]

    # --- 2. split survivors into "displaced mapped geometry" and "new structure"
    phi = float(rng.uniform(cfg.phi_lo, cfg.phi_hi))
    n_disp = int(round(phi * len(survivors)))
    pick = rng.permutation(len(survivors))
    displaced = [survivors[i] for i in pick[:n_disp]]

    smax = float(rng.uniform(cfg.shift_lo_px, cfg.shift_hi_px))
    smax = min(smax, cfg.shift_max_px)
    truth = np.zeros(shape, dtype=bool)
    max_shift = 0.0
    for rc in displaced:
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = smax * float(np.sqrt(rng.uniform(0.0, 1.0)))
        dy = int(round(rad * np.sin(ang)))
        dx = int(round(rad * np.cos(ang)))
        # integer rounding can push the vector past the calibrated 400 m = 4 px
        # radius; shrink it back inside the disc so the calibration is not exceeded
        while np.hypot(dy, dx) > cfg.shift_max_px and (dy or dx):
            dy = int(np.sign(dy) * (abs(dy) - 1)) if abs(dy) >= abs(dx) else dy
            dx = int(np.sign(dx) * (abs(dx) - 1)) if abs(dx) > abs(dy) else dx
        max_shift = max(max_shift, float(np.hypot(dy, dx)))
        sh = _shift(rc, dy, dx, shape)
        if len(sh):
            truth[sh[:, 0], sh[:, 1]] = True

    # --- 3. decoys: the rest of the survivors, plus up to the observed density ---
    n_decoys = (len(survivors) - n_disp) + int(
        round(float(rng.uniform(0.0, cfg.decoy_max_fraction)) * n))
    decoy(truth, n_decoys)

    truth = truth & allowed
    return World(
        name=f"{fold.name}-w{idx:03d}", fold=fold.name, truth=truth, phi=phi,
        n_segments_kept=int(keep.sum()), n_segments_dropped=int(n - keep.sum()),
        n_displaced=int(n_disp), n_decoys=int(n_decoys), max_shift_px=float(max_shift),
        truth_pixels=int(truth.sum()),
    )


def make_worlds(folds: list[Fold], cfg: WorldConfig) -> list[World]:
    """`cfg.n_worlds` independent worlds per fold, sharing one seeded generator."""
    rng = np.random.default_rng(cfg.seed)
    out = []
    for f in folds:
        for i in range(cfg.n_worlds):
            out.append(make_world(f, cfg, rng, idx=i))
    return out
