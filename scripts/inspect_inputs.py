#!/usr/bin/env python3
"""Record grid facts from the files themselves -> registry/grid_facts.json (no values assumed)."""
import json, sys
from pathlib import Path
import numpy as np, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems63.grid import DATA_DIR, REPO
out = {}
with rasterio.open(DATA_DIR / "labels.tif") as s:
    L = s.read(1)
    out["labels"] = dict(crs=s.crs.to_string(), shape=[s.height, s.width], transform=list(s.transform)[:6],
                         dtype=s.dtypes[0], nodata=s.nodata, values={str(int(v)): int(c) for v, c in zip(*np.unique(L, return_counts=True))})
with rasterio.open(DATA_DIR / "sample_submission.tif") as s:
    S = s.read(1)
    out["sample_submission"] = dict(crs=s.crs.to_string(), shape=[s.height, s.width], dtype=s.dtypes[0], nodata=s.nodata,
                                    nan_px=int(np.isnan(S).sum()), ones_px=int((S == 1).sum()), zeros_px=int((S == 0).sum()),
                                    equals_labels_positive=bool(np.array_equal(S == 1, L == 1)),
                                    nan_equals_labels_nodata=bool(np.array_equal(np.isnan(S), L == -1)))
with rasterio.open(DATA_DIR / "training_features.tif") as s:
    out["training_features"] = dict(crs=s.crs.to_string(), shape=[s.height, s.width], bands=s.count, dtype=s.dtypes[0],
                                    nodata=s.nodata, band_descriptions=[s.tags(i).get("description") for i in range(1, s.count + 1)])
(REPO / "registry" / "grid_facts.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: {kk: out[k][kk] for kk in list(out[k])[:4]} for k in out}, indent=1)[:1200])
