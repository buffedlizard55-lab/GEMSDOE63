# Geological hypotheses (ranked) — GEMS fault prediction

Layer names refer to the 19 bands of `training_features.tif` (EPSG:32611, 100 m). "PROJECTION" means an expected
gain reasoned from physics and cost, **not** a score. Only E1–E3 numbers in `registry/experiments/` are measured.

Common test standard (applies to every row): spatially blocked hide-and-recover holdout on the INGENIOUS/USGS catalogue
(5 folds of 4×4 super-blocks, 1 km buffer, visible catalogue masked, exact DTI α=0.2, β=0.8, R=300 m).
**Caveat**: the holdout truth is catalogue-type faults, while the competition truth is *new* faults. No proxy here measures the new-fault distribution.

| Rank | ID | Layer(s) | Physical signature targeted | Why it should find a fault missing from the catalogue | Non-fault process that could mimic it | How it differs from what is implemented | Cost | PROJECTION of gain | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | H5 — 1 m DEM scarp detection | 1 m USGS 3DEP DEM (not in this sandbox); multi-direction slope/curvature | Fault scarps: a linear, sub-metre to metre step in the land surface with a consistent strike | Scarps are the surface expression of Quaternary faults and are often absent from regional catalogues; new expert-mapped faults are likely visible this way | Landslides, road cuts, canal/levee edges, alluvial-fan margins, agricultural terraces | Uses the DEM at 1 m rather than the 100 m provided features; the repo implements no DEM-based feature | High (1 m DEM tiles, large download, login-walled list) | Highest if the scarp signal is clean (PROJECTION, not measured) | **Not validated.** Blocker: the 1 m tiles are reached from `1m_DEM_links.csv` (login-walled) and the USGS hosts are outside this sandbox's egress allowlist. The owner's `lidar_scarp_features_u8.tif` is derived by the owner's CI with unverified provenance (IR-16). |
| 2 | H4 — geodetic shear localisation | `geod_shearrate` (7), `geod_2ndinv` (4), `geod_dilat` (8) | Strain localises on active faults, giving linear shear-rate maxima | Strain can be measured on faults with no surface trace; catalogue-free | GNSS network geometry, InSAR atmospheric noise, regional strain gradients (Walker Lane) | Uses strain fields as the primary signal; E1 shows geod_shear alone AUC 0.85 but very spatially variable (per-fold 0.15–0.77) | Low (features already exist) | Medium; the E1 spread is the warning | **Canary-clean, not used as a model** (E1: below the 0.90 leak threshold). Needs a fold-stable test before use. |
| 3 | H3 — fault-bounded basement step | `basement_depth` (15), `elev_detrend` (12) | A sharp step in depth-to-basement, marking a normal-fault-bounded basin margin | Basin-bounding faults are often the least-mapped structures, especially under cover | Stratigraphic onlap and thinning of sediments, paleo-channel incision | Uses the *gradient of basement depth* as a lineament detector (new); the implemented ridge does not use it | Low | Medium | Not tested (no experiment budget left). |
| 4 | H1 — catalogue-free lineness ridge (implemented) | `tmi_hg` (3), `tilt_tc` (6), `elev_slope` (19) with Hessian lineness σ=2 px | Linear magnetic/topographic contacts | Catalogue-free, so it cannot copy catalogue geometry | Lithological contacts, dyke margins, cultural magnetic noise (fences, pipelines, power lines) | Uses lineness on three layers, no training | Low | Low (measured: at chance) | **Tested (E2).** Nominal holdout DTI 0.0258 vs random-dot control 0.0208. Negative. |
| 5 | H2 — catalogue-trained logistic on the full feature stack | 19 bands + 5 derived | Multi-feature signature of mapped faults | Learns the joint signature | Learns catalogue-specific geometry (mapped faults are not new faults) | Trained on the visible catalogue, buffered | Low | Low–medium | **Tested (E2).** Nominal 0.034 at the same budget but collapses under label-world perturbation; CVaR rejects it. Negative for robust selection. |

## Validation decision (protocol item 6)
- Top candidate by PROJECTION is **H5**. It cannot be validated in this sandbox, so it is **not** promoted.
- The best *measured* candidate, **H1 (CVaR-selected)**, is at chance, so no weekly slot is spent on it.
- No candidate has beaten a holdout best in this repo, so the rule "do not spend a slot without a holdout win" blocks promotion.

## Named free, official sources to unblock
- USGS 3DEP 1 m DEM via The National Map (https://apps.nationalmap.gov/downloader/) — tile list is the competition's `1m_DEM_links.csv`. Reachability from this sandbox: **not verified** (host not on egress allowlist).
- INGENIOUS / Great Basin Regional Dataset Compilation, DOI 10.15121/1881483 (rules footnote 4; labels' own source). Not fetched here.
- USGS Quaternary Fault and Fold Database (named in the rules, §2). Its URL was not verified in this session, so none is given. Not fetched here.
- GeoDAWN airborne survey, DOI 10.5066/P93LGLVQ (rules footnote 3) — the feature source. Not fetched here.
