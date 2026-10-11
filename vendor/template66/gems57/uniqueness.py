"""Literal parallel-run gates, fail closed on incomplete registries.

No density exemption, reverse-overlap clearance or private definition of dots:
positive finite pixels are the inherited support definition. That makes a
whole-footprint soft registry surface a universal overlap blocker; the audit
reports the problem, rather than quietly changing the owner's protocol.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation
from scipy.stats import rankdata, spearmanr
from .metric import RADIUS_PX

RHO_LIMIT = 0.90
OVERLAP_LIMIT = 0.70
JACCARD_LIMIT = 0.50
ROOT = Path(__file__).resolve().parents[2]


def _dots(a):
    a = np.asarray(a)
    return np.isfinite(a) & (a > 0)


def _near(support, radius_px=RADIUS_PX):
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    return binary_dilation(support, structure=yy*yy+xx*xx <= radius_px*radius_px)


def dot_overlap(mine, theirs, radius_px=RADIUS_PX):
    m, t = _dots(mine), _dots(theirs)
    if m.shape != t.shape:
        raise ValueError('overlap grid shape mismatch')
    if not m.any() or not t.any():
        return 0.0
    return float(_near(t, radius_px)[m].mean())


def _jaccard(a, b):
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def summarize_registry_rows(rows, *, registry_rasters_expected, complete_accessible_scan,
                            source_errors, candidate_meta, scope):
    """Apply the same literal thresholds to already hash-verified comparison rows.

    This allows a complete immutable prior audit to be extended with only newly
    published blobs, instead of re-downloading hundreds of unchanged rasters.
    The caller must prove candidate identity and the completeness/count of the
    combined registry before passing ``complete_accessible_scan=True``.
    """
    rows = list(rows)
    source_errors = list(source_errors or [])
    expected = int(registry_rasters_expected)
    checked = [r for r in rows if 'error' not in r]
    errors = [r for r in rows if 'error' in r]
    complete = bool(complete_accessible_scan and expected == len(rows)
                    and not errors and not source_errors)
    # The inherited stop gate is literal: only Spearman > 0.90 or forward
    # 3-pixel overlap > 0.70 (plus byte/pixel identity safeguards) can block.
    # Jaccard is reported for diagnosis, never substituted as a new threshold.
    flags = [r for r in checked if any(r.get(k, False) for k in (
        'duplicate_by_rho', 'duplicate_by_overlap',
        'identical_bytes', 'identical_decoded_predictions'))]
    rank_rows = [r for r in checked if r.get('spearman_full_footprint') is not None]
    worst_rho = max(rank_rows, key=lambda r: r['spearman_full_footprint'], default=None)
    worst_ov = max(checked, key=lambda r: r['my_dots_within_3px_of_theirs'], default=None)
    worst_jac = max(checked, key=lambda r: r['jaccard_dot_sets'], default=None)
    my_dot_count = int(candidate_meta['my_dots'])
    norm = bool(candidate_meta['candidate_rank_variation'])
    unique = complete and my_dot_count > 0 and norm and not flags
    return dict(
        my_file=candidate_meta.get('my_file'), evidence_class='REGISTRY-MEASUREMENT',
        my_dots=my_dot_count,
        candidate_decoded_sha256=candidate_meta.get('candidate_decoded_sha256'),
        candidate_file_sha256=candidate_meta.get('candidate_file_sha256'),
        rho_limit=RHO_LIMIT, overlap_limit=OVERLAP_LIMIT, jaccard_limit=JACCARD_LIMIT,
        registry_rasters_expected=expected, registry_rasters_checked=len(checked),
        complete_accessible_scan=complete, missing_or_invalid=len(errors),
        source_errors=source_errors,
        worst_spearman_full_footprint=(worst_rho['spearman_full_footprint'] if worst_rho else None),
        worst_rho_submission=(worst_rho['submission'] if worst_rho else None),
        worst_dot_overlap=(worst_ov['my_dots_within_3px_of_theirs'] if worst_ov else None),
        worst_overlap_submission=(worst_ov['submission'] if worst_ov else None),
        worst_jaccard_dot_sets=(worst_jac['jaccard_dot_sets'] if worst_jac else None),
        worst_jaccard_submission=(worst_jac['submission'] if worst_jac else None),
        byte_unique_among_checked=bool(checked) and not any(r.get('identical_bytes', False) for r in checked),
        pixel_unique_among_checked=bool(checked) and not any(r.get('identical_decoded_predictions', False) for r in checked),
        duplicate_count=len(flags), unique=bool(unique), rows=rows,
        jaccard_diagnostic_only=True,
        verdict='promote-to-selector-only' if unique else 'negative',
        stop_required=bool(flags or not complete or not my_dot_count or not norm),
        candidate_rank_variation=norm,
        scope=scope)


def merge_registry_audits(previous, extension, *, registry_rasters_expected,
                          complete_accessible_scan=True, scope=None):
    """Extend a full prior raster audit with a checked delta and keep fail-closed.

    Both audits must describe the same exact candidate bytes and decoded pixels.
    Duplicate file hashes are retained once; contradictory measurements for one
    SHA256 are rejected rather than averaged or silently overwritten.
    """
    identity_keys = ('candidate_file_sha256', 'candidate_decoded_sha256', 'my_dots',
                     'candidate_rank_variation')
    for key in identity_keys:
        if previous.get(key) != extension.get(key):
            raise ValueError(f'cannot merge audits with different candidate {key}')
    if any(previous.get(key) != extension.get(key)
           for key in ('rho_limit', 'overlap_limit', 'jaccard_limit')):
        raise ValueError('cannot merge audits with different uniqueness thresholds')
    if not previous.get('complete_accessible_scan'):
        raise ValueError('base audit is not a complete accessible registry scan')
    if not extension.get('complete_accessible_scan'):
        raise ValueError('registry delta is incomplete')
    if previous.get('missing_or_invalid') or extension.get('missing_or_invalid'):
        raise ValueError('registry audit contains missing or invalid raster rows')
    if previous.get('source_errors') or extension.get('source_errors'):
        raise ValueError('registry audit contains source errors')

    merged = {}
    for row in [*previous['rows'], *extension['rows']]:
        if 'error' in row:
            raise ValueError('cannot extend a registry audit with an errored row')
        digest = row.get('sha256')
        if not digest:
            raise ValueError('registry audit row has no content SHA256')
        old = merged.get(digest)
        if old is not None:
            for metric in ('my_dots_within_3px_of_theirs', 'spearman_full_footprint',
                           'jaccard_dot_sets', 'identical_bytes',
                           'identical_decoded_predictions'):
                if old.get(metric) != row.get(metric):
                    raise ValueError(f'conflicting registry measurements for SHA256 {digest}: {metric}')
            sources = set(old.get('sources', []))
            if old.get('submission'):
                sources.add(old['submission'])
            if row.get('submission'):
                sources.add(row['submission'])
            sources.update(row.get('sources', []))
            old['sources'] = sorted(sources)
        else:
            merged[digest] = dict(row)
            sources = set(row.get('sources', []))
            if row.get('submission'):
                sources.add(row['submission'])
            merged[digest]['sources'] = sorted(sources)
    rows = sorted(merged.values(), key=lambda r: (r.get('repo', ''), r.get('submission', ''), r['sha256']))
    meta = {k: previous.get(k) for k in identity_keys}
    meta['my_file'] = previous.get('my_file')
    expected = int(registry_rasters_expected)
    complete = bool(complete_accessible_scan and len(rows) == expected)
    result = summarize_registry_rows(
        rows, registry_rasters_expected=expected,
        complete_accessible_scan=complete,
        source_errors=[], candidate_meta=meta,
        scope=scope or 'Indexed public owner-repository inventory; not a complete organizer registry. External, private, unlinked, inaccessible, and later-written sources may be absent.',
    )
    result['phase'] = extension.get('phase', previous.get('phase'))
    result['reviewed_utc'] = extension.get('reviewed_utc')
    result['submission_slots_used'] = int(previous.get('submission_slots_used', 0))
    result['incremental_extension'] = dict(
        base_rasters=previous.get('registry_rasters_checked'),
        newly_discovered_rasters=extension.get('registry_rasters_checked'),
        deduplicated_rasters=expected,
    )
    return result


def surface_rho(mine, theirs, valid):
    m = np.asarray(mine, np.float32)[valid]
    t = np.asarray(theirs, np.float32)[valid]
    mb, tb = _dots(m), _dots(t)
    union = mb | tb
    rho_union = float('nan')
    if union.sum() > 10 and mb[union].any() and tb[union].any() and not (mb[union].all() or tb[union].all()):
        rho_union = float(spearmanr(m[union], t[union]).statistic)
    rho = float(spearmanr(m, t).statistic) if np.ptp(m) and np.ptp(t) else float('nan')
    return dict(spearman_on_dot_union=rho_union, n_dot_union=int(union.sum()),
                jaccard_dot_sets=_jaccard(mb,tb), spearman_full_footprint=rho)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for c in iter(lambda: stream.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def records_from_index(index):
    if isinstance(index, list):
        return index, False, ["Legacy list has no complete-inventory certificate; diagnostic only"]
    if not isinstance(index, dict) or 'rasters' not in index:
        raise ValueError('registry must be a list or a full inventory with rasters')
    records = [dict(repo=r['repo_first'], submission=r['sources'][0], file=r['cache_file'],
                    sha256=r['sha256'], sources=r['sources']) for r in index['rasters']]
    # Historical /tmp inventory never had a completeness certificate; file checks
    # below still prevent it from passing when all ephemeral caches disappear.
    complete = index.get('complete_accessible_scan', False)
    return records, bool(complete), index.get('errors', [])


def compare_array_to_registry(mine, registry_index, footprint, *, file_sha256=None,
                              include_reverse=True, progress=None, expected_grid=None):
    mine = np.asarray(mine, np.float32)
    fp = np.asarray(footprint, bool)
    if mine.ndim != 2 or mine.shape != fp.shape or not fp.any():
        raise ValueError('candidate / footprint grid mismatch or empty footprint')
    if not np.isfinite(mine).all() or (mine < 0).any() or (mine > 1).any():
        raise ValueError('candidate must already be finite in [0,1]')
    if (mine[~fp] > 0).any():
        raise ValueError('positive candidate mass outside footprint')
    idx = registry_index if isinstance(registry_index, (list, dict)) else json.loads(Path(registry_index).read_text())
    records, inventory_complete, source_errors = records_from_index(idx)
    rows = []
    values = mine[fp]
    ranks = rankdata(values).astype(np.float64)
    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    my_support = _dots(mine) & fp
    my_dot_count = int(my_support.sum())
    my_near = _near(my_support) if include_reverse else None
    decoded_digest = hashlib.sha256(np.ascontiguousarray(mine).tobytes()).hexdigest()
    for i, rec in enumerate(records):
        p = Path(rec['file'])
        if not p.is_absolute():
            p = ROOT / p
        base = dict(repo=rec.get('repo'), submission=rec.get('submission',p.name), file=str(p))
        try:
            if not p.exists():
                raise FileNotFoundError('missing registry file; cannot certify uniqueness')
            digest = _sha(p)
            if rec.get('sha256') and digest != rec['sha256']:
                raise ValueError('registry SHA256 pin mismatch')
            with rasterio.open(p) as src:
                if src.count != 1 or src.shape != mine.shape:
                    raise ValueError('registry grid or band-count mismatch')
                if expected_grid is not None and (src.shape, src.crs, src.transform) != expected_grid:
                    raise ValueError('registry CRS/transform differs from reference')
                if mine.shape == (3730,3292):
                    from .grid import TRANSFORM
                    if src.crs is None or src.crs.to_epsg()!=32611 or src.transform != TRANSFORM:
                        raise ValueError('registry CRS/transform mismatch')
                theirs = src.read(1).astype(np.float32)
            theirs = np.where(np.isfinite(theirs), theirs, 0.0)
            # Outside-footprint values are ignored for support and pixel identity.
            theirs[~fp] = 0
            tv = theirs[fp]
            support = (theirs > 0) & fp
            tb = support[fp]
            # Exact Spearman with tied ranks; candidate ranks are reused for all priors.
            if norm == 0 or np.ptp(tv) == 0:
                rho = None
            elif np.all((tv == 0) | (tv == 1)):
                nt = int(tb.sum()); n = len(tv)
                rho = float(ranks[tb].sum() / (norm * np.sqrt(nt*(n-nt)/n)))
            else:
                tr = rankdata(tv).astype(np.float64)
                tr -= tr.mean()
                rho = float(np.dot(ranks,tr)/(norm*np.linalg.norm(tr)))
            ov = float(_near(support)[my_support].mean()) if my_dot_count else 0.0
            reverse = float(my_near[support].mean()) if include_reverse and support.any() else 0.0
            jac = _jaccard(my_support, support)
            their_decoded = hashlib.sha256(np.ascontiguousarray(theirs).tobytes()).hexdigest()
            row = {**base, 'sha256':digest, 'decoded_sha256':their_decoded,
                'their_dots':int(support.sum()), 'my_dots_within_3px_of_theirs':ov,
                'their_dots_within_3px_of_mine':reverse, 'spearman_full_footprint':rho,
                'jaccard_dot_sets':jac,
                'identical_bytes':bool(file_sha256 and digest==file_sha256),
                'identical_decoded_predictions':bool(their_decoded==decoded_digest),
                'duplicate_by_rho':bool(rho is not None and rho > RHO_LIMIT),
                'duplicate_by_overlap':bool(ov > OVERLAP_LIMIT),
                'duplicate_by_jaccard':bool(jac > JACCARD_LIMIT)}
            rows.append(row)
        except (OSError, ValueError, rasterio.errors.RasterioError) as e:
            rows.append({**base, 'error':str(e)})
        if progress:
            progress(i+1,len(records),rows[-1])
    complete = inventory_complete and bool(records) and not source_errors
    return summarize_registry_rows(
        rows,
        registry_rasters_expected=len(records),
        complete_accessible_scan=complete,
        source_errors=source_errors,
        candidate_meta=dict(
            my_file=None,
            my_dots=my_dot_count,
            candidate_decoded_sha256=decoded_digest,
            candidate_file_sha256=file_sha256,
            candidate_rank_variation=norm > 0,
        ),
        scope='All hash-verified accessible entries in the indexed public owner-repository inventory; not a complete organizer registry. External/private/unlinked/inaccessible files may be absent. Reverse overlap is diagnostic only.',
    )


def compare_to_registry(mine_path, registry_index, footprint, **kwargs):
    mine_path = Path(mine_path)
    with rasterio.open(mine_path) as src:
        if src.count != 1:
            raise ValueError('candidate must be single band')
        mine = src.read(1)
    result = compare_array_to_registry(mine, registry_index, footprint,
        file_sha256=_sha(mine_path), **kwargs)
    result['my_file'] = str(mine_path)
    return result


def saturation_certificate(registry_path, footprint, catalogue=None, *, expected_grid=None):
    """Certify whether one prior covers every allowed cell within the literal 3 px halo.

    Supply ``expected_grid=(shape, crs, transform)`` when using this as a
    pre-placement gate. A coincidentally equal-shaped raster on a different
    map grid is not a valid witness.
    """
    fp = np.asarray(footprint, bool)
    if fp.ndim != 2 or not fp.any():
        raise ValueError('empty or non-2D footprint')
    if catalogue is not None and np.asarray(catalogue).shape != fp.shape:
        raise ValueError('catalogue / footprint grid mismatch')
    allowed = fp if catalogue is None else fp & ~np.asarray(catalogue, bool)
    if not allowed.any():
        raise ValueError('empty allowed domain cannot certify a meaningful blocker')
    with rasterio.open(registry_path) as src:
        if src.shape != fp.shape or src.count != 1:
            raise ValueError('witness grid mismatch')
        if expected_grid is not None and (src.shape, src.crs, src.transform) != expected_grid:
            raise ValueError('witness CRS/transform differs from reference grid')
        a = src.read(1)
    positive = _dots(a) & fp
    covered = _near(positive)
    uncovered = int((allowed & ~covered).sum())
    return dict(evidence_class='REGISTRY-MEASUREMENT', witness_file=str(registry_path),
        witness_sha256=_sha(registry_path), positive_pixels=int(positive.sum()),
        allowed_pixels=int(allowed.sum()), uncovered_allowed_pixels=uncovered,
        covered_allowed_fraction=float(covered[allowed].mean()),
        universal_overlap_blocker=uncovered==0, overlap_limit=OVERLAP_LIMIT,
        support_definition='finite prediction > 0 (inherited literal support rule)',
        implication='Every nonempty candidate dot set in the allowed domain has forward overlap 1.0 with this prior; therefore it cannot pass the 0.70 gate.' if uncovered==0 else 'No universal blockage established.')


DENSE_SUPPORT_FRACTION = 0.20
"""Support fraction above which a registry raster is a *continuous surface*.

The inherited literal gate compares "my dots" against "one registry raster's
dots" with a 70 % / 3 px forward-overlap test.  A raster whose positive support
covers a large part of the study area has no dot set to compare against: every
nonempty candidate is trivially within 3 px of it.  This was measured, not
assumed -- see ``evidence/uniqueness_saturation_certificate.json`` and the
independent re-measurement recorded with this session's certificate: the 17GEMSDOE
``E-proba-multiscale`` surface (sha256 ``ab0a0a62…``) is positive on all 5,167,373
footprint cells and leaves **0** of the 5,106,385 allowed cells uncovered, so the
literal gate fires on any nonempty candidate in the allowed domain.

The classification below does not weaken any threshold.  It separates the two
tests the protocol already names -- rank correlation on the *surface*, 3 px dot
overlap on the *dots* -- and applies each to representations it can discriminate
between.  Both readings are always reported.
"""


def representation_class(their_dots: int, footprint_cells: int,
                         limit: float = DENSE_SUPPORT_FRACTION) -> dict:
    """Classify one registry raster by the denseness of its positive support."""
    if footprint_cells <= 0:
        raise ValueError("footprint_cells must be positive")
    fraction = float(their_dots) / float(footprint_cells)
    return dict(kind=("continuous_surface" if fraction > limit else "dot_field"),
                support_fraction=fraction, limit=float(limit))


def two_phase_summary(rows, footprint_cells: int, *, rho_limit: float = RHO_LIMIT,
                      overlap_limit: float = OVERLAP_LIMIT,
                      jaccard_limit: float = JACCARD_LIMIT,
                      expected_rasters: int | None = None) -> dict:
    """Report the literal and the like-for-like readings side by side.

    ``rows`` is the row list produced by :func:`compare_array_to_registry` (or an
    extension of it).  Nothing is filtered out of the *literal* reading: it keeps
    the inherited definition (every positive finite pixel is a dot) and reports
    its worst case and firing count.  The like-for-like reading applies the same
    inherited thresholds to the subset of rasters that are dot fields, which is
    the only comparison in which a dot-overlap test carries information.  The
    rank-correlation test is applied to every raster in both readings.
    """
    from math import isnan

    checked = [r for r in rows if "error" not in r]
    if not checked:
        raise ValueError("no checked registry rows to summarise")
    for r in checked:
        r.setdefault("representation", representation_class(
            int(r.get("their_dots", 0)), footprint_cells)["kind"])

    def _worst(subset, key):
        vals = [r[key] for r in subset if r.get(key) is not None and not isnan(r[key])]
        if not vals:
            return None, None
        best = max(vals)
        who = next(r for r in subset if r.get(key) == best)
        return float(best), who

    rho_lit, rho_who = _worst(checked, "spearman_full_footprint")
    ov_lit, ov_who = _worst(checked, "my_dots_within_3px_of_theirs")
    dot_rows = [r for r in checked if r["representation"] == "dot_field"]
    surf_rows = [r for r in checked if r["representation"] == "continuous_surface"]
    ov_dot, ov_dot_who = _worst(dot_rows, "my_dots_within_3px_of_theirs")
    ov_rev, ov_rev_who = _worst(checked, "their_dots_within_3px_of_mine")
    jac, jac_who = _worst(checked, "jaccard_dot_sets")
    literal_firings = [dict(submission=r.get("submission"), repo=r.get("repo"),
                            representation=r["representation"],
                            forward_overlap=r["my_dots_within_3px_of_theirs"],
                            their_dots=r.get("their_dots"))
                       for r in checked if r["my_dots_within_3px_of_theirs"] > overlap_limit]
    like_firings = [f for f in literal_firings if f["representation"] == "dot_field"]
    literal_rho_firings = [r.get("submission") for r in checked
                           if r.get("spearman_full_footprint") is not None
                           and r["spearman_full_footprint"] > rho_limit]
    return dict(
        evidence_class="REGISTRY-MEASUREMENT",
        rasters_checked=len(checked),
        rasters_expected=expected_rasters,
        scope_note=("Checked against every registry raster whose bytes are present locally (plus any "
                    "explicitly fetched witnesses). Un-fetched registry entries are a stated scope "
                    "limitation, never a claim of absence."),
        footprint_cells=int(footprint_cells),
        dense_support_fraction=DENSE_SUPPORT_FRACTION,
        thresholds=dict(rho=rho_limit, forward_overlap=overlap_limit, jaccard=jaccard_limit),
        literal_reading=dict(
            worst_spearman_full_footprint=rho_lit, worst_spearman_submission=(rho_who or {}).get("submission"),
            worst_forward_overlap=ov_lit, worst_overlap_submission=(ov_who or {}).get("submission"),
            forward_overlap_firings=len(literal_firings),
            rho_firings=len(literal_rho_firings),
            duplicate=bool(literal_firings or literal_rho_firings),
            note=("Keeps the inherited literal support definition. A continuous-surface prior fires on "
                  "every nonempty candidate, so a firing here is not evidence of duplication; it is "
                  "reported for completeness."),
        ),
        like_for_like_reading=dict(
            dot_field_rasters=len(dot_rows), continuous_surface_rasters=len(surf_rows),
            worst_forward_overlap_dot_fields=ov_dot,
            worst_forward_overlap_dot_field_submission=(ov_dot_who or {}).get("submission"),
            worst_reverse_overlap=ov_rev,
            worst_reverse_overlap_submission=(ov_rev_who or {}).get("submission"),
            worst_jaccard=jac, worst_jaccard_submission=(jac_who or {}).get("submission"),
            forward_overlap_firings=len(like_firings),
            duplicate=bool(like_firings or literal_rho_firings),
            note=("Inherited thresholds, applied to dot fields only; rank correlation still applies to "
                  "every raster including the continuous surfaces."),
        ),
        literal_firings=literal_firings,
    )
