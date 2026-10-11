#!/usr/bin/env python3
"""Fetch every earlier raster in the registry and verify it byte-for-byte.

The uniqueness gate in :mod:`gems63.uniqueness` is only as good as the rasters it
compares against, so this script downloads each registry entry from the GitHub
contents API, checks its sha256 against the manifest, and records what it got in
``registry/cache.json``.  A manifest entry whose hash does not match is reported and
excluded -- a comparison against the wrong bytes would be worse than no comparison.

    python scripts/fetch_registry.py [--manifest registry/registry_manifest.json]

Rasters are cached under ``registry/rasters/`` which is gitignored (they are
re-derivable from the sibling repositories named in the manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
API = "https://api.github.com/repos/buffedlizard55-lab/{repo}/contents/{path}"
ORG = "buffedlizard55-lab"


def fetch(repo: str, path: str, dest: Path, timeout: int = 180) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = API.format(repo=repo, path=path)
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.raw",
        "User-Agent": "gems63-registry-fetch",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return dest.stat().st_size


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "registry" / "registry_manifest.json"))
    ap.add_argument("--out", default=str(REPO / "registry" / "cache.json"))
    ap.add_argument("--skip-existing", action="store_true", default=True)
    a = ap.parse_args()

    man = json.loads(Path(a.manifest).read_text())
    entries = man["entries"]
    out = []
    ok = bad = 0
    for e in entries:
        dest = REPO / "registry" / "rasters" / f"{e['alias']}.tif"
        if a.skip_existing and dest.exists() and sha256(dest) == e["sha256"]:
            got = sha256(dest)
        else:
            try:
                fetch(e["repo"], e["path"], dest)
                got = sha256(dest)
            except Exception as exc:                      # noqa: BLE001
                print(f"  FETCH-FAILED {e['alias']}: {exc}", file=sys.stderr)
                continue
        if got != e["sha256"]:
            print(f"  HASH-MISMATCH {e['alias']}: manifest {e['sha256'][:16]} got {got[:16]}",
                  file=sys.stderr)
            bad += 1
            continue
        ok += 1
        out.append({
            "alias": e["alias"], "repo": e["repo"], "path": e["path"],
            "local_path": str(dest.relative_to(REPO)),
            "bytes": dest.stat().st_size, "sha256": got,
            "owner_reported_score": e.get("owner_reported_score"),
            "note": e.get("note"),
        })
        print(f"  ok {e['alias']:<44s} {dest.stat().st_size:>10,}  "
              f"score={e.get('owner_reported_score')}")

    payload = {
        "note": ("Prior artifacts used ONLY for the parallel-run lane-drift check. "
                 "Owner-reported scores are the values quoted on the sibling sites and "
                 "the DrivenData public leaderboard; they are NOT organizer receipts "
                 "for these bytes."),
        "source_host": "https://api.github.com/repos/buffedlizard55-lab/<repo>/contents/<path>",
        "verified": ok, "hash_mismatch": bad, "total_in_manifest": len(entries),
        "entries": out,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"\nverified {ok}/{len(entries)} entries -> {a.out}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
