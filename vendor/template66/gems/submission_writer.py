"""Write a competition-format submission GeoTIFF from a probability array.

Refuses to write unless a reference raster supplies the CRS and geotransform, so the
output can never silently carry a guessed grid. Output: single-band float32, nodata tag
absent, NaN exactly outside the footprint, finite values in [0, 1] inside it, plus a JSON
receipt (sha256, counts, extrema) written next to the file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CLIP_TOLERANCE = 1e-6  # float round-off only; anything larger is a bug and raises


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


NOTE_MAX_CHARS = 140  # brief: note must be at most 140 characters


def write_submission(out_path: str | Path, prediction, footprint, reference_path: str | Path,
                     overwrite: bool = False, name: str | None = None, note: str | None = None) -> dict:
    import rasterio

    if name is not None and (not name.strip() or len(name) > 120):
        raise ValueError("name must be non-empty and at most 120 characters")
    if note is not None and len(note) > NOTE_MAX_CHARS:
        raise ValueError(f"note is {len(note)} characters; the limit is {NOTE_MAX_CHARS}")
    out_path = Path(out_path)
    if out_path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("output must be a .tif file")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (pass overwrite=True)")

    pred = np.asarray(prediction, dtype=np.float64)
    fp = np.asarray(footprint, dtype=bool)
    with rasterio.open(reference_path) as ref:
        ref_shape = (ref.height, ref.width)
        crs, transform = ref.crs, ref.transform
    if pred.shape != ref_shape or fp.shape != ref_shape:
        raise ValueError(f"prediction {pred.shape} / footprint {fp.shape} != reference {ref_shape}")
    if np.isnan(pred[fp]).any():
        raise ValueError("NaN inside the footprint is not allowed")
    if np.isinf(pred[fp]).any():
        raise ValueError("infinite values inside the footprint are not allowed")

    inside = pred[fp]
    below = int((inside < -CLIP_TOLERANCE).sum())
    above = int((inside > 1.0 + CLIP_TOLERANCE).sum())
    if below or above:
        raise ValueError(f"{below + above} values outside [0, 1] (refusing to clip silently)")
    out = np.full(ref_shape, np.nan, dtype=np.float32)
    out[fp] = np.clip(inside, 0.0, 1.0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", height=ref_shape[0], width=ref_shape[1],
                       count=1, dtype="float32", crs=crs, transform=transform, nodata=None,
                       compress="deflate") as dst:
        dst.write(out, 1)
        tags = {"GEMS_NAME": name or out_path.stem, "GEMS_NOTE": note or "",
                "GEMS_GENERATOR": "gems.submission_writer"}
        dst.update_tags(**tags)

    receipt = {
        "file": out_path.name,
        "sha256": sha256_file(out_path),
        "shape_hw": list(ref_shape),
        "crs": crs.to_string() if crs else None,
        "footprint_cells": int(fp.sum()),
        "emitted_cells_gt0": int((out[fp] > 0).sum()),
        "min_inside": float(out[fp].min()) if fp.any() else None,
        "max_inside": float(out[fp].max()) if fp.any() else None,
        "nan_inside": int(np.isnan(out[fp]).sum()),
        "nan_outside_count": int(np.isnan(out[~fp]).sum()),
        "name": name or out_path.stem,
        "note": note or "",
        "note_chars": len(note or ""),
        "reference": Path(reference_path).name,
        "receipt_kind": "local receipt written by submission_writer; NOT an organiser receipt",
    }
    Path(str(out_path) + ".receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
