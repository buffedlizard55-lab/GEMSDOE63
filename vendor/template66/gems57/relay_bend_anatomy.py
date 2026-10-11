"""Multi-scale host-bend asymmetry, two-host relay superposition, and slip-sense transitions.

Implements the next predeclared fault-zone anatomy hypotheses from REMAINING_WORK.md §3:
  * H57-H (BEND_FEATURES): Multi-scale host-bend damage asymmetry (sigma=3 px vs
    sigma=9 px visible-trace structure tensors) and detrended-elevation scarp
    relative strike (cached band 12 `det_elev`).
  * H57-I2 (RELAY_FEATURES): Two-host damage-zone superposition and en echelon
    stepover relay geometry between the nearest two DISTINCT visible 8-connected
    fault components (C1 != C2), computed via exact bitplane Euclidean distance
    transforms.
  * H57-J (TRANSITION_FEATURES + SENSE_FEATURES): Along-host / inter-host
    slip-sense transition heterogeneity between strike-slip (RL/LL) and normal (N)
    visible INGENIOUS traces within a 1.5 km (15 px) damage neighborhood.

All catalogue-derived quantities are computed strictly from the `visible` fault
mask of the fold, never from withheld (`hidden`) traces. No textbook Riedel or
stress-lobe angle is hard-coded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

from .anatomy import FEATURES, SENSE_FEATURES, fold_geometry
from .network import STRUCT3, local_strike
from .strand_orientation import magnetic_geometry

BEND_FEATURES = (
    "host_bend_turn",       # (1 - cos(2*(theta_3 - theta_9))) * coh_3 * coh_9 in [0, 2]
    "host_bend_asym",       # side * sin(2*(theta_3 - theta_9)) * coh_3 * coh_9 in [-1, 1]
    "scarp_rel_cos2",       # coh_scarp * cos(2*(theta_scarp - theta_host)) in [-1, 1]
)

RELAY_FEATURES = (
    "d2",                   # Euclidean distance (px) to nearest distinct visible component C2 != C1
    "relay_ratio",          # d1 / max(d1 + d2, 1e-6) in [0, 0.5] (0.5 on medial relay axis)
    "relay_facing",         # -grad(d1) . grad(d2) in [-1, 1] (+1 inside stepover corridor)
    "two_host_strike_cos2", # cos(2*(theta_host1 - theta_host2)) in [-1, 1]
    "superposed_log_len2",  # log(1 + L2), displacement proxy of partnering host component C2
)

TRANSITION_FEATURES = (
    "sense_transition",     # 4*rho_SS*rho_N / (rho_SS + rho_N)^2 * presence in [0, 1]
)

ALL_RELAY_BEND_FEATURES = (
    FEATURES + SENSE_FEATURES + BEND_FEATURES + RELAY_FEATURES + TRANSITION_FEATURES
)


def cached_scarp_geometry(root: Path, footprint: np.ndarray) -> dict[str, np.ndarray]:
    """Reuse the restored cached det_elev band (band 12); never read catalogue labels."""
    manifest = json.loads((root / "evidence/feature_cache.json").read_text())
    row = next(r for r in manifest["bands"] if r["band"] == 12)
    source = root / row["cache_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("cached det_elev band hash mismatch")
    target = root / ".cache/scarp_orientation"
    target.mkdir(parents=True, exist_ok=True)
    source_digest = row["sha256"]
    receipt_path = target / "scarp.json"
    if receipt_path.exists():
        rec = json.loads(receipt_path.read_text())
        if rec["source_sha256"] == source_digest and all(
            (target / f"{n}.npy").exists() for n in rec["fields"]
        ):
            return {n: np.load(target / f"{n}.npy", mmap_mode="r") for n in rec["fields"]}
    fields = magnetic_geometry(np.load(source, mmap_mode="r"), footprint)
    for name, arr in fields.items():
        np.save(target / f"{name}.npy", arr)
    receipt_path.write_text(
        json.dumps(
            dict(
                source_sha256=source_digest,
                band=12,
                source_short_name="det_elev",
                fields=list(fields),
                transform="Gaussian derivatives and axial scarp tangent, catalogue independent",
            ),
            indent=2,
        )
        + "\n"
    )
    return {n: np.load(target / f"{n}.npy", mmap_mode="r") for n in fields}


def host_bend_fields(visible: np.ndarray, fine_px: float = 3.0, coarse_px: float = 9.0) -> dict[str, np.ndarray]:
    """Multi-scale visible-trace structure tensor strike and turning fields.

    Computes local trace strike and coherence at ``fine_px`` (300 m) and
    ``coarse_px`` (900 m) strictly from ``visible``. Forms the spin-2 director
    vectors T_sigma = c_sigma * (cos(2*theta_sigma), sin(2*theta_sigma)) so that
    ``turn = ||T_fine - T_coarse||_2`` captures both angular turning on the bend
    limbs and coherence drop at the bend hinge, while ``sin2_turn`` captures the
    signed cross-product T_fine x T_coarse in [-1, 1].
    """
    vis = np.asarray(visible, bool)
    if vis.ndim != 2 or not vis.any():
        raise ValueError("host_bend_fields requires a non-empty 2D visible catalogue")
    s_fine, c_fine = local_strike(vis, smooth_px=fine_px)
    s_coarse, c_coarse = local_strike(vis, smooth_px=coarse_px)
    valid = np.isfinite(s_fine) & np.isfinite(s_coarse)
    th1 = np.where(valid, np.radians(2.0 * s_fine), 0.0)
    th2 = np.where(valid, np.radians(2.0 * s_coarse), 0.0)
    cf = np.where(valid, np.clip(c_fine, 0.0, 1.0), 0.0).astype(np.float32)
    cc = np.where(valid, np.clip(c_coarse, 0.0, 1.0), 0.0).astype(np.float32)
    t1x, t1y = cf * np.cos(th1).astype(np.float32), cf * np.sin(th1).astype(np.float32)
    t2x, t2y = cc * np.cos(th2).astype(np.float32), cc * np.sin(th2).astype(np.float32)
    turn = np.hypot(t1x - t2x, t1y - t2y).astype(np.float32)
    sin2_turn = (t1y * t2x - t1x * t2y).astype(np.float32)
    return dict(
        strike_fine=s_fine,
        coh_fine=c_fine,
        strike_coarse=s_coarse,
        coh_coarse=c_coarse,
        turn=turn,
        sin2_turn=sin2_turn,
    )


def two_host_frame(visible: np.ndarray) -> dict[str, np.ndarray]:
    """Exact nearest and second-nearest DISTINCT visible connected components.

    Uses a ceil(log2(n_components + 1))-bitplane Euclidean distance transform:
    any two distinct component IDs c1 != c2 differ in at least one binary bit b,
    so taking the minimum EDT over all bit-complement masks S_{b, 1 - bit_b(c1)}
    yields the exact Euclidean distance and anchor coordinates to the nearest
    visible pixel in a different connected component (c2 != c1) everywhere on the grid.
    """
    vis = np.asarray(visible, bool)
    if vis.ndim != 2 or not vis.any():
        raise ValueError("two_host_frame is undefined for an empty/non-2D visible mask")
    comp, ncomp = ndi.label(vis, structure=STRUCT3)
    if ncomp < 2:
        raise ValueError("two_host_frame requires at least two distinct visible components")
    comp_len = np.bincount(comp.ravel(), minlength=ncomp + 1).astype(np.float32)
    d1, (iy1, ix1) = ndi.distance_transform_edt(~vis, return_indices=True)
    d1 = d1.astype(np.float32)
    iy1 = iy1.astype(np.int32)
    ix1 = ix1.astype(np.int32)
    c1 = comp[iy1, ix1]

    nbits = int(np.ceil(np.log2(ncomp + 1)))
    d2 = np.full(vis.shape, np.inf, dtype=np.float32)
    iy2 = np.zeros(vis.shape, dtype=np.int32)
    ix2 = np.zeros(vis.shape, dtype=np.int32)
    for b in range(nbits):
        bit_c1 = (c1 >> b) & 1
        for val in (0, 1):
            sub = vis & (((comp >> b) & 1) == val)
            if not sub.any():
                continue
            db, (yb, xb) = ndi.distance_transform_edt(~sub, return_indices=True)
            use = (bit_c1 != val) & (db < d2)
            d2[use] = db[use].astype(np.float32)
            iy2[use] = yb[use].astype(np.int32)
            ix2[use] = xb[use].astype(np.int32)
            del db, yb, xb, use
    c2 = comp[iy2, ix2]
    if (c1 == c2).any() or (c2 == 0).any():
        raise AssertionError("two_host_frame failed to separate distinct components")
    return dict(
        d1=d1,
        iy1=iy1,
        ix1=ix1,
        c1=c1,
        d2=d2,
        iy2=iy2,
        ix2=ix2,
        c2=c2,
        comp_len=comp_len,
    )


def kinematic_transition_field(
    visible: np.ndarray, sense_src: np.ndarray, smooth_px: float = 15.0
) -> np.ndarray:
    """Visible-only slip-sense heterogeneity index between strike-slip and normal traces.

    Restricts ``sense_src`` strictly to ``visible`` pixels before smoothing so
    withheld traces never leak. Returns a float32 field in [0, 1] that peaks where
    strike-slip (RL=2, LL=3) and normal (N=1) visible traces coexist within a
    ``smooth_px`` (1.5 km) Gaussian neighborhood.
    """
    vis = np.asarray(visible, bool)
    sense = np.where(vis, np.asarray(sense_src, np.int8), 0)
    ss_mask = ((sense == 2) | (sense == 3)).astype(np.float32)
    n_mask = (sense == 1).astype(np.float32)
    if not ss_mask.any() or not n_mask.any():
        return np.zeros(vis.shape, dtype=np.float32)
    rho_ss = ndi.gaussian_filter(ss_mask, smooth_px, mode="constant", truncate=3.0)
    rho_n = ndi.gaussian_filter(n_mask, smooth_px, mode="constant", truncate=3.0)
    tot = rho_ss + rho_n
    denom = tot * tot
    mix = np.divide(4.0 * rho_ss * rho_n, denom, out=np.zeros_like(denom), where=denom > 1e-12)
    pos_tot = tot[tot > 1e-7]
    scale = float(np.percentile(pos_tot, 90)) if pos_tot.size else 1.0
    presence = np.clip(tot / max(scale, 1e-7), 0.0, 1.0)
    return np.clip(mix * presence, 0.0, 1.0).astype(np.float32)


def append_relay_bend_features(
    g,
    visible: np.ndarray,
    sense_src: np.ndarray,
    scarp_fields: dict[str, np.ndarray],
    destination: Path,
    cache_dir: Path | None = None,
    cache_tag: str = "",
):
    """Append H57-H, H57-I2, and H57-J columns to a base+sense FoldGeometry."""
    base_cols = len(FEATURES + SENSE_FEATURES)
    if g.X.shape[1] != base_cols:
        raise ValueError("recorded-sense geometry must precede relay/bend blocks")
    destination.parent.mkdir(parents=True, exist_ok=True)

    bend = host_bend_fields(visible, fine_px=3.0, coarse_px=9.0)
    if cache_dir is not None and cache_tag:
        cache_dir.mkdir(parents=True, exist_ok=True)
        th_file = cache_dir / f"two_host_{cache_tag}.npz"
        if th_file.exists():
            th = dict(np.load(th_file))
        else:
            th = two_host_frame(visible)
            np.savez(th_file, **th)
    else:
        th = two_host_frame(visible)
    trans = kinematic_transition_field(visible, sense_src, smooth_px=15.0)

    s_fine = bend["strike_fine"]
    # Fallback to segment strike where local tensor strike is NaN at isolated pixels
    seg_strike = g.seg.strike[np.clip(g.seg.seg_id, 0, len(g.seg.strike) - 1)]
    s_resolved = np.where(np.isfinite(s_fine), s_fine, seg_strike)
    s_resolved = np.where(np.isfinite(s_resolved), s_resolved, 0.0).astype(np.float32)

    out = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype="float32",
        shape=(len(g.y), len(ALL_RELAY_BEND_FEATURES)),
    )
    comp_len = th["comp_len"]
    for start in range(0, len(g.y), 200_000):
        sl = slice(start, min(start + 200_000, len(g.y)))
        yy, xx = g.rows[sl], g.cols[sl]
        x = g.X[sl]
        out[sl, :base_cols] = x

        ay1 = th["iy1"][yy, xx]
        ax1 = th["ix1"][yy, xx]
        ay2 = th["iy2"][yy, xx]
        ax2 = th["ix2"][yy, xx]
        d1 = th["d1"][yy, xx]
        d2 = th["d2"][yy, xx]
        c2 = th["c2"][yy, xx]

        # H57-H: Multi-scale host-bend damage asymmetry & scarp relative strike
        turn1 = bend["turn"][ay1, ax1]
        sin2_turn1 = bend["sin2_turn"][ay1, ax1]
        side = x[:, 3]
        bend_asym = side * sin2_turn1
        # cos(2*(theta_scarp - theta_host1)) weighted by scarp coherence
        scarp_rel = (
            scarp_fields["cos2"][yy, xx] * x[:, 6]
            + scarp_fields["sin2"][yy, xx] * x[:, 5]
        ) * scarp_fields["coherence"][yy, xx]

        # H57-I2: Two-host damage-zone superposition & relay stepover geometry
        relay_ratio = np.clip(d1 / np.maximum(d1 + d2, 1e-6), 0.0, 0.5)
        dy1 = (yy - ay1).astype(np.float32)
        dx1 = (xx - ax1).astype(np.float32)
        dy2 = (yy - ay2).astype(np.float32)
        dx2 = (xx - ax2).astype(np.float32)
        denom_d = np.maximum(d1 * d2, 1e-6)
        relay_facing = np.clip(-(dy1 * dy2 + dx1 * dx2) / denom_d, -1.0, 1.0)

        s1 = s_resolved[ay1, ax1]
        s2 = s_resolved[ay2, ax2]
        two_host_cos2 = np.cos(np.radians(2.0 * (s1 - s2))).astype(np.float32)
        log_len2 = np.log1p(comp_len[np.clip(c2, 0, len(comp_len) - 1)]).astype(np.float32)

        # H57-J: Along-host / inter-host slip-sense transition index
        sense_tr = trans[yy, xx]

        out[sl, base_cols + 0] = turn1
        out[sl, base_cols + 1] = np.clip(bend_asym, -1.0, 1.0)
        out[sl, base_cols + 2] = np.clip(scarp_rel, -1.0, 1.0)
        out[sl, base_cols + 3] = d2
        out[sl, base_cols + 4] = relay_ratio
        out[sl, base_cols + 5] = relay_facing
        out[sl, base_cols + 6] = np.clip(two_host_cos2, -1.0, 1.0)
        out[sl, base_cols + 7] = log_len2
        out[sl, base_cols + 8] = sense_tr
    out.flush()
    return SimpleNamespace(
        X=out,
        y=g.y,
        rows=g.rows,
        cols=g.cols,
        key=g.key,
        feature_names=ALL_RELAY_BEND_FEATURES,
    )


def build_relay_bend_geometry(
    grid,
    visible: np.ndarray,
    hidden: np.ndarray,
    domain: np.ndarray,
    sense_src: np.ndarray,
    scarp_fields: dict[str, np.ndarray],
    cache: Path,
    key: str,
):
    """Build full 22-column feature matrix from visible faults + cached scarp band."""
    base = fold_geometry(grid, visible, hidden, domain, key, sense_src=sense_src)
    return append_relay_bend_features(
        base,
        visible,
        sense_src,
        scarp_fields,
        cache / f"relay_bend_{key}.npy",
        cache_dir=cache,
        cache_tag=key,
    )
