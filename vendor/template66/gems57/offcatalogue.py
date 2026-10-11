"""Off-catalogue proxy-truth instrument for the fault-zone anatomy lane.

Why this module exists
----------------------
The inherited hide-and-recover instrument withholds *catalogue* segments and
asks the model to recover them.  That measures recovery of geometry the
catalogue itself contains, so it rewards placing dots tight against visible
traces.  The competition target is the opposite population: fault experts
manually identified faults that are **not** contained in the current public
USGS database (competition page 967, "For this challenge, we have consulted
with fault experts who have manually identified faults that are not contained
within the current public USGS database").  Forum thread 11536 extends the
definition to newly mapped geometry of an existing system.

This module builds the second instrument: a **proxy new-fault truth** that is
by construction disjoint from the scored catalogue, and a set of
catalogue-only covariates, so the fitted damage zone, the relative-strike
structure and the dot budget are measured against the population the lane
actually claims to predict.

Provenance and limits (declared, not hidden)
--------------------------------------------
* ``proxy`` is ``derived_sgmc_faults_100m.tif`` -- an in-repo auxiliary raster
  described by ``data/README.md`` as "historical off-catalogue proxy derived
  from SGMC minus the mapped union; not hidden truth".  It is a *proxy* for
  newly mapped faults, never an organizer label.  Absolute proxy DTI is not a
  score and is never reported as one.
* Every predictor is derived from the competition catalogue
  (``data/official/existing_faults.tif``) and the INGENIOUS trace database
  (``data/external/trace_segments_utm11.csv``) only.  The proxy layer is used
  **only** as a truth/response variable, never as a predictor, so it cannot
  leak into its own model.  :func:`assert_predictor_sources` enforces that.
* Spatial blocking is leave-one-quadrant-out (LOQO) with the same quadrant
  definition and boundary erosion the inherited holdout uses, so the fitted
  distance profile is never evaluated on the quadrant it was fitted on.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .anatomy import offset_components
from .faultzone import SENSE_CODE, trace_sense_raster
from .grid import Grid
from .holdout import DOMAIN_ERODE, FOLD_NAMES, quadrant_ids
from .network import STRUCT3, local_strike

VERSION = 'gems57-offcatalogue-loqo-v1'

# Distance bins (px = 100 m).  Wide at the tail because the proxy-new-fault
# population is a broad halo, not a tight damage zone.
DIST_EDGES = np.array([1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 35, 45, 58, 75, 95, 1e9])

# Only the sources below may feed a predictor column.
ALLOWED_PREDICTOR_SOURCES = frozenset({'catalogue', 'ingenious'})


def assert_predictor_sources(names) -> None:
    """Fail closed if a predictor column is not from an approved source."""
    for name in names:
        source = str(name).split(':', 1)[0]
        if source not in ALLOWED_PREDICTOR_SOURCES:
            raise ValueError(f'predictor {name!r} has unapproved source {source!r}')


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_proxy_truth(root: Path, grid: Grid) -> tuple[np.ndarray, dict]:
    """The off-catalogue proxy-new-fault mask plus a provenance receipt.

    Built **explicitly** as ``sgmc_faults_100m.tif`` minus the competition
    catalogue, both restricted to the scored footprint, rather than trusting the
    pre-existing ``derived_sgmc_faults_100m.tif``.  Measured discrepancy
    (IR-57-SGMC-01): that file is 1,446 px larger than the raw SGMC raster and
    still contains 260 catalogue pixels, i.e. it is a ~1 px dilation of SGMC,
    not the "SGMC minus the mapped union" that ``data/README.md`` describes.
    The derivation used here is reproduced from the pinned source bytes.
    """
    import rasterio
    root = Path(root)
    sgmc_path = root / 'data/external/sgmc_faults_100m.tif'
    derived_path = root / 'data/external/derived_sgmc_faults_100m.tif'
    with rasterio.open(sgmc_path) as src:
        sgmc = src.read(1) > 0
        grid_check = (str(src.crs), list(src.shape))
    with rasterio.open(derived_path) as src:
        derived = src.read(1) > 0
    if grid_check != ('EPSG:32611', list(grid.shape)):
        raise ValueError('SGMC raster is not on the pinned competition grid')
    proxy = sgmc & ~grid.catalogue & grid.footprint
    receipt = dict(
        file=str(sgmc_path.relative_to(root)), sha256=sha256(sgmc_path),
        bytes=sgmc_path.stat().st_size,
        derivation='proxy = (sgmc_faults_100m > 0) & ~catalogue & footprint',
        crs='EPSG:32611', transform=[float(v) for v in tuple(grid.transform)[:6]],
        shape=list(grid.shape),
        sgmc_pixels=int(sgmc.sum()),
        sgmc_pixels_on_catalogue=int((sgmc & grid.catalogue).sum()),
        proxy_pixels=int(proxy.sum()),
        proxy_fraction_of_footprint=float(proxy.sum() / grid.footprint.sum()),
        irregularity_IR_57_SGMC_01=dict(
            described_as='SGMC minus the mapped union (data/README.md)',
            measured='derived_sgmc_faults_100m.tif has %d px, %d more than SGMC, '
                     'and still contains %d catalogue px; it is a ~1 px dilation'
                     % (int(derived.sum()), int(derived.sum() - sgmc.sum()),
                        int((derived & grid.catalogue).sum() - (sgmc & grid.catalogue).sum())),
            action='this instrument rebuilds the proxy from the pinned SGMC bytes'),
        evidence_class='EXTERNAL-AUXILIARY-LAYER (proxy population, not organizer truth)',
        declared_limit=('public state/geologic fault compilation minus the competition '
                        'catalogue; a proxy for newly mapped faults, never an organizer '
                        'label and never a score'))
    return proxy, receipt


def load_sense(root: Path, grid: Grid) -> tuple[np.ndarray, dict]:
    """Recorded sense of slip rasterised from the INGENIOUS trace export."""
    csv = Path(root) / 'data/external/trace_segments_utm11.csv'
    sense = trace_sense_raster(csv, grid.shape, grid.transform)
    counts = {name: int((sense == code).sum()) for name, code in SENSE_CODE.items()}
    return sense, dict(file=str(csv), sha256=sha256(csv), codes=SENSE_CODE,
                       n_pixels=counts,
                       evidence_class='DATA-MEASUREMENT (public INGENIOUS trace export)')


@dataclass
class AnatomyCovariates:
    """Catalogue-only geometry evaluated on every footprint cell off-catalogue."""
    d: np.ndarray            # px to nearest catalogue pixel
    host: np.ndarray         # index of the nearest catalogue pixel (flat)
    host_comp: np.ndarray    # component id of that pixel
    log_len: np.ndarray      # log1p(host component length) - displacement proxy
    strike: np.ndarray       # local strike at the host pixel (deg)
    coh: np.ndarray          # strike coherence at the host pixel
    d_perp: np.ndarray       # across-strike offset to the host (signed)
    side: np.ndarray         # +1 / -1 side of the host
    sense: np.ndarray        # recorded sense of the host pixel (0/1/2/3)
    d2: np.ndarray           # px to the nearest *different* visible component
    theta_local: np.ndarray  # local orientation of the candidate neighbourhood


def catalogue_covariates(grid: Grid, sense_src: np.ndarray) -> AnatomyCovariates:
    """Compute every predictor from the catalogue alone (never from the proxy)."""
    cat = grid.catalogue
    strike, coh = local_strike(cat, smooth_px=3.0)
    comp, ncomp = ndi.label(cat, structure=STRUCT3)
    comp_len = np.bincount(comp.ravel(), minlength=ncomp + 1).astype(np.float64)

    d, (iy, ix) = ndi.distance_transform_edt(~cat, return_indices=True)
    host = (iy * grid.width + ix).astype(np.int64)
    host_comp = comp[iy, ix]
    log_len = np.log1p(comp_len[host_comp])
    s = np.where(np.isfinite(strike[iy, ix]), strike[iy, ix], 0.0)
    # axial orientation of the local neighbourhood at each cell
    theta_local, _ = local_strike(grid.footprint & ~cat, smooth_px=5.0)
    ygrid, xgrid = np.indices(grid.shape, dtype=np.float32)
    d_par, d_perp, side = offset_components(ygrid - iy, xgrid - ix, s)

    # Distance to the nearest pixel of a *different* catalogue component, using
    # the shared exact bitplane two-host frame (no private distance code).
    from .relay_bend_anatomy import two_host_frame
    two = two_host_frame(cat)
    d2 = np.asarray(two['d2'], np.float32)
    d1 = np.asarray(two['d1'], np.float32)
    np.testing.assert_allclose(d1, cov_d := d.astype(np.float32), rtol=1e-5, atol=1e-3)
    del cov_d

    names = ('catalogue:d', 'catalogue:host_len', 'catalogue:host_strike',
             'catalogue:d_perp', 'catalogue:side', 'catalogue:sense', 'catalogue:d2')
    assert_predictor_sources(names)
    return AnatomyCovariates(d=d.astype(np.float32), host=host, host_comp=host_comp,
                              log_len=log_len.astype(np.float32), strike=s,
                              coh=coh[iy, ix].astype(np.float32),
                              d_perp=np.zeros(grid.shape, np.float32), side=side,
                              sense=sense_src[iy, ix].astype(np.int8),
                              d2=d2, theta_local=theta_local.astype(np.float32))


def fold_domains(grid: Grid, proxy: np.ndarray, *, boundary_px: int = DOMAIN_ERODE):
    """Leave-one-quadrant-out domains, eroded so truth is never clipped."""
    quad = quadrant_ids(grid.footprint)
    domains = {}
    for k, name in enumerate(FOLD_NAMES):
        region = ndi.binary_erosion(quad == k, iterations=boundary_px)
        region &= grid.footprint
        truth = proxy & region
        if not truth.any():
            raise ValueError(f'fold {name} has no proxy positives')
        domains[name] = dict(region=region, truth=truth,
                             scored=(region & ~grid.catalogue).copy())
    return domains


def distance_profile(region: np.ndarray, truth: np.ndarray, cov: AnatomyCovariates):
    """Empirical P(truth | distance band) per area, i.e. the fitted damage zone."""
    d = cov.d[region]
    t = truth[region]
    counts, _ = np.histogram(d, bins=DIST_EDGES)
    pos, _ = np.histogram(d[t], bins=DIST_EDGES)
    area = np.maximum(counts, 1).astype(float)
    rate = pos / area
    return dict(edges=DIST_EDGES.tolist(), band_area=counts.astype(float).tolist(),
                band_positive=pos.astype(float).tolist(),
                band_rate=rate.tolist(), n_positive=float(pos.sum()),
                n_region=float(region.sum()))


def profile_field(profile: dict, cov: AnatomyCovariates) -> np.ndarray:
    """Map the fitted per-band rate onto every cell, smoothly interpolated."""
    edges = np.asarray(profile['edges'], float)
    rate = np.asarray(profile['band_rate'], float)
    centres = np.sqrt(edges[:-1] * edges[1:])
    centres[-1] = edges[-2] * 1.5
    rate = np.maximum.accumulate(np.maximum(rate, 0.0))
    x = np.clip(cov.d, centres[0], centres[-1])
    return np.interp(x, centres, rate).astype(np.float32)


def fold_angle(a, b):
    """Unsigned axial difference in degrees, folded to [0, 90]."""
    diff = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180.0
    return np.where(diff > 90.0, 180.0 - diff, diff)


def relative_strike_structure(grid: Grid, proxy: np.ndarray,
                              cov: AnatomyCovariates, bins: int = 9) -> dict:
    """Relative strike of proxy strands against their nearest catalogue strand.

    This is the lane's en-echelon test.  The null is *visible* catalogue pixels
    measured against the nearest catalogue pixel of a different connected
    component, i.e. the same measurement applied to the known population.
    """
    from .network import local_strike as _ls
    p_strike, p_coh = _ls(proxy, smooth_px=3.0)
    edges = np.linspace(0.0, 90.0, bins + 1)

    ys, xs = np.nonzero(proxy)
    flat_host = cov.host[ys, xs]
    keep = (np.isfinite(p_strike[ys, xs]) & (p_coh[ys, xs] > 0.2)
            & np.isfinite(cov.strike[ys, xs]) & (cov.coh[ys, xs] > 0.2))
    angle = fold_angle(p_strike[ys[keep], xs[keep]], cov.strike[ys[keep], xs[keep]])

    cat = grid.catalogue
    c_strike, c_coh = _ls(cat, smooth_px=3.0)
    vy, vx = np.nonzero(cat)
    if vy.size < 2:
        raise ValueError('catalogue too small for a relative-strike null')
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([vy, vx], 1))
    _, idx = tree.query(np.stack([vy, vx], 1), k=min(13, vy.size))
    comp, _ = ndi.label(cat, structure=STRUCT3)
    cid = comp[vy, vx]
    other = comp[vy[idx], vx[idx]] != cid[:, None]
    first = np.argmax(other, axis=1)
    has = other.any(axis=1)
    sel = np.flatnonzero(has)
    j = idx[sel, first[has]]
    a = c_strike[vy[sel], vx[sel]]
    b = c_strike[vy[j], vx[j]]
    ok = (np.isfinite(a) & np.isfinite(b) & (c_coh[vy[sel], vx[sel]] > 0.2)
          & (c_coh[vy[j], vx[j]] > 0.2))
    null = fold_angle(a[ok], b[ok])

    hp, _ = np.histogram(angle, bins=edges)
    hn, _ = np.histogram(null, bins=edges)
    enrich = np.divide(hp / max(hp.sum(), 1), hn / max(hn.sum(), 1),
                       out=np.ones(bins), where=hn > 0)
    return dict(
        evidence_class='HOLDOUT-STRUCTURE (descriptive, not a score)',
        edges=edges.tolist(), proxy_bins=hp.astype(float).tolist(),
        null_bins=hn.astype(float).tolist(),
        enrichment=[float(v) for v in enrich],
        n_proxy=int(angle.size), n_null=int(null.size),
        median_proxy=float(np.median(angle)) if angle.size else None,
        median_null=float(np.median(null)) if null.size else None,
        p25_proxy=float(np.percentile(angle, 25)) if angle.size else None,
        p75_proxy=float(np.percentile(angle, 75)) if angle.size else None,
        note=('unsigned axial difference folded to [0,90] deg; the null is a '
              'catalogue pixel against the nearest different-component catalogue '
              'pixel within the 13 nearest queried pixels; pixel-weighted, censored '
              'at the coherence floor, descriptive only'))


def sense_structure(region: np.ndarray, truth: np.ndarray,
                    cov: AnatomyCovariates) -> dict:
    """Proxy-new-fault rate by the *recorded* sense of the nearest catalogue host."""
    rows = []
    for name, code in (('N', 1), ('RL', 2), ('LL', 3), ('unrecorded', 0)):
        host_in = cov.sense[region] == code
        if not host_in.any():
            rows.append(dict(sense=name, area=0, positive=0, rate=None))
            continue
        rate = float(truth[region][host_in].mean())
        rows.append(dict(sense=name, area=int(host_in.sum()),
                         positive=int(truth[region][host_in].sum()),
                         rate=rate,
                         rate_ratio_to_all=float(rate / max(truth[region].mean(), 1e-12))))
    return dict(rows=rows, region_rate=float(truth[region].mean()),
                evidence_class='HOLDOUT-STRUCTURE (descriptive, not a score)',
                note='sense of the nearest catalogue host pixel from the INGENIOUS trace export')


def length_structure(region: np.ndarray, truth: np.ndarray,
                     cov: AnatomyCovariates) -> dict:
    """Proxy-new-fault rate by host-component size (the displacement proxy)."""
    upper = float(np.log1p(4000) + 1.0)   # finite sentinel keeps the JSON receipt strict
    edges = np.array([0.0, np.log1p(50), np.log1p(200), np.log1p(600),
                      np.log1p(1500), np.log1p(4000), upper])
    lab = np.digitize(cov.log_len[region], edges) - 1
    t = truth[region]
    rows = []
    for k in range(len(edges) - 1):
        sel = lab == k
        if not sel.any():
            continue
        rows.append(dict(bin=k, area=int(sel.sum()), positive=int(t[sel].sum()),
                         rate=float(t[sel].mean()),
                         rate_ratio_to_all=float(t[sel].mean() / max(t.mean(), 1e-12))))
    return dict(edges=edges.tolist(), rows=rows,
                evidence_class='HOLDOUT-STRUCTURE (descriptive, not a score)',
                note='host = nearest catalogue pixel; size = 8-connected component pixel count')