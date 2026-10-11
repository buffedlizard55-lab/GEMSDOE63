#!/usr/bin/env bash
# Fetch the competition data first-hand, on a machine that has a competition login.
#
# This sandbox cannot reach www.drivendata.org (HTTP 000) and the data page requires
# authentication in any case, so the files in data/ here are a second-hand copy
# recovered from sibling repositories -- see data/README.md for why that matters.
# Running this script on a logged-in machine replaces that copy with a first-hand one.
#
# Usage:
#   export GEMS_USERNAME=... GEMS_PASSWORD=...      # or let it prompt
#   bash scripts/download_competition_data.sh
#
# Nothing here is interactive beyond the credential prompt, and no credential is ever
# written to disk by this script.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO/data/official"
URL="https://www.drivendata.org/competitions/306/competition-doe-gems/data/"

mkdir -p "$DATA"

if [[ -z "${GEMS_USERNAME:-}" || -z "${GEMS_PASSWORD:-}" ]]; then
  echo "Set GEMS_USERNAME and GEMS_PASSWORD, or edit this script." >&2
  echo "Credentials are used only for this download and are never stored." >&2
  read -r -p "DrivenData username: " GEMS_USERNAME
  read -r -s -p "DrivenData password: " GEMS_PASSWORD
  echo
  export GEMS_USERNAME GEMS_PASSWORD
fi

# 1. log in and keep the session cookie
JAR="$(mktemp)"
trap 'rm -f "$JAR"' EXIT
curl -sS -c "$JAR" -o /dev/null \
  -d "username=${GEMS_USERNAME}" -d "password=${GEMS_PASSWORD}" \
  -d "csrfmiddlewaretoken=$(curl -sS -c "$JAR" https://www.drivendata.org/accounts/login/ \
      | sed -n "s/.*name='csrfmiddlewaretoken' value='\([^']*\)'.*/\1/p" | head -1)" \
  https://www.drivendata.org/accounts/login/

# 2. download each artifact the pipeline needs
for f in training_features.tif labels.tif sample_submission.tif 1m_DEM_links.csv; do
  echo "-> $f"
  curl -sS -b "$JAR" -L -o "$DATA/$f" "${URL%/}/${f}" || echo "   FAILED: $f" >&2
done

# 3. record what was actually obtained, and check it is a GeoTIFF and not an HTML
#    login page -- the failure mode that silently poisons every downstream number.
echo
python3 - "$DATA" <<'PY'
import hashlib, sys
from pathlib import Path
d = Path(sys.argv[1])
for p in sorted(d.iterdir()):
    head = p.open("rb").read(4)
    kind = "GeoTIFF" if head[:2] in (b"II", b"MM") else f"NOT A TIFF (starts {head!r})"
    print(f"  {p.name:<28s} {p.stat().st_size:>12,}  {kind}  "
          f"sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
PY

echo
echo "Done. Next: python3 scripts/run_lane.py --worlds 12 --name <your-run>"
