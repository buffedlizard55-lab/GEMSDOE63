"""Label-world ensemble generator - the core of the GEMSDOE63 robust-selection lane.

The rules force one submission chosen before private scores are known (competition
rules section 3.6.2). Organizers state Phase 2 faults can be added, removed or
changed, and Hermant et al. measured catalogue offsets up to 400 m (4 px at 100 m),
so the private truth will differ from the catalogue we validate on.

Calibration (declared before any run):
- SHIFT: each withheld fault component is translated by (dy, dx) iid uniform over
  {-4..+4} px, matching the up-to-400 m catalogue offsets measured by
  Hermant et al. (aftershock/relocated catalogue mislocations), cited in the
  session prompt.
- OMISSION: each world drops a fraction rho in U[0.10, 0.30] of withheld
  components (prompt-specified plausible omission band).
- DECOYS (ADDITIONS): organizers state round-2 faults can be ADDED, removed or
  changed, and the lane specifies "add decoy segments at the observed density".
  So each world adds n_decoy = floor(phi * n_fold_components_observed) decoy
  components with phi ~ U[0.10, 0.30] (the same calibrated drift band as the
  omission rate), sampled from the fold's own catalogue component shapes and
  placed at uniform random positions inside the fold's domain, i.e. decoys occur
  AT THE OBSERVED component density, not tied to the catalogue's own geography.
  (Session-1 defect fix: the first generator tied decoys only to dropped mass,
  which rewarded proven-worse catalogue-flank hugging; corrected before any
  selection was trusted.)

Everything is documented and reproducible (fixed seeds). The generator is the
honesty bound of the result: it represents "catalogue-drift" label worlds, not
new-tectonics worlds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    cell_key: str
    world_id: int
    drop_frac: float
    decoy_frac: float
    n_dropped: int
    n_decoys: int
    shifts: list
    truth_coords: tuple  # (rows, cols) int32 within cell bbox, already clipped to active


def generate_worlds(cell, n_worlds: int = 20, seed: int = 63_000, shift_max: int = 4,
                    drop_lo: float = 0.10, drop_hi: float = 0.30) -> list:
    """Generate `n_worlds` plausible label worlds for one holdout cell."""
    rng = np.random.default_rng(seed + cell.fold * 1_000 + cell.seed * 17)
    H, W = cell.active_sub.shape
    comps = cell.hidden_comp_coords
    n_hidden = len(comps)
    shapes = cell.fold_comp_shapes
    worlds = []

    for w in range(n_worlds):
        rho = float(rng.uniform(drop_lo, drop_hi))
        n_drop = int(round(rho * n_hidden))
        order = rng.permutation(n_hidden)
        dropped = set(order[:n_drop].tolist())
        truth = np.zeros((H, W), dtype=bool)
        shifts = []
        n_survivors = 0
        for i, (rr, cc) in enumerate(comps):
            if i in dropped:
                continue
            dy = int(rng.integers(-shift_max, shift_max + 1))
            dx = int(rng.integers(-shift_max, shift_max + 1))
            nr = rr + dy
            nc = cc + dx
            ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
            nr, nc = nr[ok], nc[ok]
            keep = cell.active_sub[nr, nc]
            nr, nc = nr[keep], nc[keep]
            if nr.size:
                truth[nr, nc] = True
                n_survivors += 1
            shifts.append((i, dy, dx))

        # decoys (additions) at the observed fold component density, drift band 10-30%
        phi = float(rng.uniform(drop_lo, drop_hi))
        n_decoy = int(np.floor(phi * cell.n_fold_components)) if shapes else 0
        placed = 0
        for _ in range(n_decoy):
            si = int(rng.integers(0, len(shapes)))
            sr, sc, _sz = shapes[si]
            for _try in range(40):
                r0 = int(rng.integers(0, H))
                c0 = int(rng.integers(0, W))
                nr = sr.astype(np.int64) + r0
                nc = sc.astype(np.int64) + c0
                ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
                if ok.mean() < 0.9:
                    continue
                nr, nc = nr[ok], nc[ok]
                frac_act = float(cell.active_sub[nr, nc].mean())
                if frac_act < 0.9:
                    continue
                keep = cell.active_sub[nr, nc]
                truth[nr[keep], nc[keep]] = True
                placed += 1
                break

        rr_t, cc_t = np.nonzero(truth)
        worlds.append(
            World(
                cell_key=cell.key,
                world_id=w,
                drop_frac=rho,
                decoy_frac=phi,
                n_dropped=n_drop,
                n_decoys=placed,
                shifts=shifts,
                truth_coords=(rr_t.astype(np.int32), cc_t.astype(np.int32)),
            )
        )
    return worlds
