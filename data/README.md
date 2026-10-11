# Data provenance — read this before trusting any number

## What is here

| file | bytes | sha256 (first 16) | what it is |
| --- | --- | --- | --- |
| `official/labels.tif` | 425,830 | `7ba308ccdc4418b3` | USGS/INGENIOUS known-fault raster, int8, values `{-1, 0, 1}`, nodata `-1` |
| `official/existing_faults.tif` | 425,830 | `7ba308ccdc4418b3` | **byte-identical to `labels.tif`** — see the caveat below |
| `official/sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd` | the organizer's submission template |
| `cache/gems-geodawn-numerical-features.tif` | 418,912,844 | `4371c82e3b8339b8` | the 19-band GeoDAWN feature stack (gitignored) |

Full hashes are in [`../evidence/data_sha256.txt`](../evidence/data_sha256.txt).

Verified metadata, read from the files themselves:

```
3292 x 3730, EPSG:32611, Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
bounds: left 243350.0, bottom 4135550.0, right 572550.0, top 4508550.0
sample_submission: 1 band, float32, nodata = nan
features: 19 bands, float32, nodata = -3.4028234663852886e+38
footprint (feature data present): 5,165,852 px = 42.07 % of the raster
catalogue positive pixels: 60,988 (60,894 inside the footprint)
```

The 19 bands, as tagged inside the file:

| # | description | category |
| --- | --- | --- |
| 1 | Magnetic anomaly | magnetic_data |
| 2 | Reduced to pole magnetic data | magnetic_data |
| 3 | Total magnetic intensity horizontal gradient | magnetic_data |
| 4 | Geodetic second invariant | geodetic_strain |
| 5 | Isostatic gravity anomaly slope | gravity_data |
| 6 | Tilt angle or total curvature | magnetic_data |
| 7 | Geodetic shear rate | geodetic_strain |
| 8 | Geodetic dilatation rate | geodetic_strain |
| 9 | Total magnetic intensity vertical gradient | magnetic_data |
| 10 | Distance to earthquake | seismic |
| 11 | Isostatic gravity anomaly vertical gradient | gravity_data |
| 12 | Detrended elevation | topographic |
| 13 | Isostatic gravity anomaly | gravity_data |
| 14 | Total magnetic intensity | magnetic_data |
| 15 | Depth to basement surface | subsurface |
| 16 | Earthquake intensity or density | seismic |
| 17 | Conductivity surface | subsurface |
| 18 | Isostatic gravity anomaly horizontal gradient | gravity_data |
| 19 | Detrended elevation slope | topographic |

This matches the feature list published on
[competition page 967](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/).

## Where it came from — and the caveat that matters

**This sandbox cannot reach `www.drivendata.org` or `usgs.gov` (HTTP 000), and the
competition data page requires a login in any case.** The files above were therefore
recovered from sibling repositories in the same GitHub account:

- `labels.tif`, `existing_faults.tif`, `sample_submission.tif` — from
  `buffedlizard55-lab/62GEMSDOE`, path `data/`.
- `gems-geodawn-numerical-features.tif` — committed to
  `buffedlizard55-lab/6GEMSDOE` as five split parts at
  `data/bridge/gems-geodawn-numerical-features.tif.part-00{0..4}`, concatenated in order
  here.

**So this is a second-hand copy, not a download from the organizer.** Every number this
repository produces inherits that provenance. To replace it with a first-hand copy:

```bash
# on any machine with a competition login
bash scripts/download_competition_data.sh   # into data/
```

## Caveats found while reading the data

1. **`labels.tif` and `existing_faults.tif` are byte-identical.** Same sha256, same size.
   They are distributed as separate artifacts. Any lane that treats them as carrying
   different information is silently using one file twice. This pipeline reads only
   `labels.tif`. Flagged for a forum question.
2. **94 catalogue pixels fall outside the feature footprint.** They are excluded from the
   scored domain by construction.
3. **The feature stack's nodata is the float32 sentinel `-3.4028234663852886e+38`**, not
   NaN. The reference solution notebook treats any value below `-1e38` as missing; this
   repository compares against the declared nodata value instead, which is the stricter
   and more explicit test.

## What is not here

- `1m_DEM_links.csv` — the list of 1 m DEM download URLs.
- The 1 m DEM tiles themselves (300–380 MB each, from the USGS 3DEP TNM Access API).
- Any ground truth for the *new* faults. The organizers will not share the construction
  (thread 11527) and the private label set is, by design, unavailable.
