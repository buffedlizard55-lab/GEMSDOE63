#!/usr/bin/env bash
# Restore the hash-pinned inputs lane 63 needs, from pinned owner-mirror repos.
# FAIL CLOSED on any digest mismatch. Nothing here contacts DrivenData or Dropbox.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data/bridge" "$ROOT/data/external" "$ROOT/data/registry_rasters"

fetch() { # repo ref path dest sha256
  local repo="$1" ref="$2" path="$3" dest="$4" sha="$5"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok (cached) $dest"; return 0
  fi
  gh api "repos/$repo/contents/$path?ref=$ref" -H 'Accept: application/vnd.github.raw' > "$dest"
  local got; got="$(sha256sum "$dest" | cut -d' ' -f1)"
  if [ "$got" != "$sha" ]; then
    echo "DIGEST MISMATCH for $dest: expected $sha got $got" >&2; rm -f "$dest"; exit 1
  fi
  echo "ok (fetched) $dest"
}

fetch buffedlizard55-lab/GEMSDOE24 07345ea0604953d7efb858d9cfbc21e20c7aca0b \
  data/bridge/labels.tif "$ROOT/data/bridge/labels.tif" \
  7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093
cp -f "$ROOT/data/bridge/labels.tif" "$ROOT/data/bridge/existing_faults.tif"
fetch buffedlizard55-lab/GEMSDOE24 07345ea0604953d7efb858d9cfbc21e20c7aca0b \
  data/bridge/sample_submission.tif "$ROOT/data/bridge/sample_submission.tif" \
  2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc
fetch buffedlizard55-lab/GEMSDOE24 07345ea0604953d7efb858d9cfbc21e20c7aca0b \
  data/external/geodawn_rad_u8.tif "$ROOT/data/external/geodawn_rad_u8.tif" \
  c22420f75999030d7cc65c9e31e50d232ea6158423bca051613a18a8b20ba682

echo "bridge + external inputs restored and verified."
echo "registry rasters: restore with the GEMSDOE65 registry index"
echo "(registry/prior_rasters.json; 730 prediction-like blobs; see evidence/registry_download.log for the fetch pattern)."
