"""Candidate-strand orientation inside a fitted damage zone (fault-zone lane).

Unlike the old offset-angle feature, this estimates a candidate edge tangent
from a catalogue-independent magnetic scalar field. Relative strike, distance
and mapped-host length interact in a fitted model; no Riedel angle is imposed.
Magnetic contacts and dikes can mimic the signal, so a magnetic edge is NOT a
verified fault, permeability estimate, geothermal vent, or resource discovery.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi
from .anatomy import FEATURES, SENSE_FEATURES, fold_geometry

MAG_FEATURES = ('mag_relative_cos2', 'mag_log_gradient', 'mag_coherence')
ALL_FEATURES = FEATURES + SENSE_FEATURES + MAG_FEATURES


def magnetic_geometry(scalar, valid):
    """Label-free, normalized smoothing, tangent and structure-tensor coherence.

    Derivatives near missing data are not treated as geology. A fixed three-pixel
    erosion is a numerical support guard, NOT a learned physical fault-zone width.
    Axial angles are represented by sin/cos(2 theta) to avoid wraparound.
    """
    a = np.asarray(scalar, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(a)
    if a.ndim != 2 or a.shape != valid.shape:
        raise ValueError('magnetic scalar / mask shape mismatch')
    mass = ndi.gaussian_filter(valid.astype(np.float32), 1.0, mode='constant', truncate=3)
    smooth = ndi.gaussian_filter(np.where(valid, a, 0), 1.0, mode='constant', truncate=3)
    smooth = np.divide(smooth, mass, out=np.zeros_like(smooth), where=mass > 1e-8)
    gy, gx = np.gradient(smooth)
    norm2 = gx * gx + gy * gy
    support = ndi.binary_erosion(valid, np.ones((3, 3), bool), iterations=4)
    good = support & (norm2 > 1e-12)
    # Edge tangent (dr, dc) = (-gx, gy); geological strike = atan2(gy, gx).
    cos2 = np.divide(gx*gx - gy*gy, norm2, out=np.zeros_like(norm2), where=good)
    sin2 = np.divide(2*gx*gy, norm2, out=np.zeros_like(norm2), where=good)
    jxx = ndi.gaussian_filter(gx*gx, 2.0, truncate=3)
    jyy = ndi.gaussian_filter(gy*gy, 2.0, truncate=3)
    jxy = ndi.gaussian_filter(gx*gy, 2.0, truncate=3)
    coh = np.sqrt((jxx-jyy)**2 + 4*jxy*jxy) / np.maximum(jxx+jyy, 1e-12)
    return dict(cos2=np.where(good, cos2, 0).astype(np.float32),
                sin2=np.where(good, sin2, 0).astype(np.float32),
                log_gradient=np.where(good, np.log1p(np.sqrt(norm2)), 0).astype(np.float32),
                coherence=np.where(good, np.clip(coh, 0, 1), 0).astype(np.float32),
                support=good)


def cached_magnetic_geometry(root: Path, footprint):
    """Reuse the restored cached TMI band; never read catalogue labels here."""
    manifest = json.loads((root / 'evidence/feature_cache.json').read_text())
    row = next(r for r in manifest['bands'] if r['band'] == 14)
    source = root / row['cache_file']
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('cached magnetic band hash mismatch')
    target = root / '.cache/strand_orientation'
    target.mkdir(parents=True, exist_ok=True)
    source_digest = row['sha256']
    receipt_path = target / 'magnetic.json'
    if receipt_path.exists():
        rec = json.loads(receipt_path.read_text())
        if rec['source_sha256'] == source_digest and all((target / f'{n}.npy').exists() for n in rec['fields']):
            return {n: np.load(target/f'{n}.npy', mmap_mode='r') for n in rec['fields']}
    fields = magnetic_geometry(np.load(source, mmap_mode='r'), footprint)
    for name, arr in fields.items():
        np.save(target/f'{name}.npy', arr)
    receipt_path.write_text(json.dumps(dict(source_sha256=source_digest, band=14,
        source_short_name='tmi', fields=list(fields), transform='Gaussian derivatives and axial tangent, catalogue independent'), indent=2)+'\n')
    return {n: np.load(target/f'{n}.npy', mmap_mode='r') for n in fields}


def append_magnetic(g, fields, destination: Path):
    """Append candidate-to-host relative strike without exposing hidden geometry."""
    if g.X.shape[1] != len(FEATURES + SENSE_FEATURES):
        raise ValueError('recorded-sense geometry must precede the magnetic block')
    destination.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(destination, mode='w+', dtype='float32',
                                   shape=(len(g.y), len(ALL_FEATURES)))
    for start in range(0, len(g.y), 200_000):
        sl = slice(start, min(start+200_000, len(g.y)))
        yy, xx = g.rows[sl], g.cols[sl]
        x = g.X[sl]
        out[sl, :len(FEATURES+SENSE_FEATURES)] = x
        # cos(2*(theta_edge-theta_host)); NOT the angle of the offset vector.
        relative = fields['cos2'][yy,xx]*x[:,6] + fields['sin2'][yy,xx]*x[:,5]
        out[sl, -3] = np.clip(relative, -1, 1)
        out[sl, -2] = fields['log_gradient'][yy,xx]
        out[sl, -1] = fields['coherence'][yy,xx]
    out.flush()
    return SimpleNamespace(X=out, y=g.y, rows=g.rows, cols=g.cols, key=g.key)


def geometry(grid, visible, hidden, domain, sense, magnetic, cache: Path, key):
    base = fold_geometry(grid, visible, hidden, domain, key, sense_src=sense)
    return append_magnetic(base, magnetic, cache / f'{key}.npy')
