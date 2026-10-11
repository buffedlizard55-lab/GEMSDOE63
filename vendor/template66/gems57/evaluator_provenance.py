"""Version and source/input hashes for spatial leave-one-quadrant-out evaluation.

A score without these hashes is not evidence from this evaluator version. Keep
this source list synchronized with the modules that construct folds, features,
fit the model, allocate dots, and calculate DTI; callers share the exact helper.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CV_VERSION = "gems57-spatial-loqo-v2"
ROOT = Path(__file__).resolve().parents[2]
CV_IMPLEMENTATION_FILES = (
    "scripts/run_cv.py",
    "src/gems57/evaluator_provenance.py",
    "src/gems57/fitting.py",
    "src/gems57/anatomy.py",
    "src/gems57/holdout.py",
    "src/gems57/metric.py",
    "src/gems57/emit.py",
    "src/gems57/grid.py",
    "src/gems57/spatial.py",
    "src/gems57/network.py",
)
CV_INPUT_FILES = (
    "data/bridge/existing_faults.tif",
    "data/bridge/sample_submission.tif",
)
FAULTZONE_VERSION = "gems57-faultzone-pooled-hide-v2"
FAULTZONE_IMPLEMENTATION_FILES = (
    "scripts/run_holdout.py",
    "src/gems57/evaluator_provenance.py",
    "src/gems57/faultzone.py",
    "src/gems57/fit.py",
    "src/gems57/evaluate_holdout.py",
    "src/gems57/holdout.py",
    "src/gems57/metric.py",
    "src/gems57/grid.py",
    "src/gems57/spatial.py",
    "src/gems57/network.py",
)
FAULTZONE_INPUT_FILES = (
    "data/official/labels.tif",
    "data/official/sample_submission.tif",
    "data/external/trace_segments_utm11.csv",
    "evidence/exp1b_pooled.npz",
    "evidence/exp1b_sgmc.npz",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_files(names, root: Path = ROOT) -> dict[str, str]:
    result = {}
    missing = []
    for name in names:
        path = root / name
        if not path.is_file():
            missing.append(name)
        else:
            result[name] = sha256_file(path)
    if missing:
        raise FileNotFoundError(f"cannot version evaluator; missing files: {missing}")
    return result


def cv_implementation_hashes(root: Path = ROOT) -> dict[str, str]:
    """Return exact source hashes for the pinned spatial CV implementation."""
    return _hash_files(CV_IMPLEMENTATION_FILES, Path(root))


def cv_input_hashes(root: Path = ROOT) -> dict[str, str]:
    """Return exact hashes for the grid rasters consumed by spatial CV."""
    return _hash_files(CV_INPUT_FILES, Path(root))


def faultzone_implementation_hashes(root: Path = ROOT) -> dict[str, str]:
    """Return exact source hashes for the fitted fault-zone hide/recover run."""
    return _hash_files(FAULTZONE_IMPLEMENTATION_FILES, Path(root))


def faultzone_input_hashes(root: Path = ROOT) -> dict[str, str]:
    """Hash every raster/table/fitted measurement consumed by the fault-zone run."""
    return _hash_files(FAULTZONE_INPUT_FILES, Path(root))
