#!/usr/bin/env python3
"""Restore the competition inputs into GEMS_DATA_DIR (default /tmp/gems_data), outside the git repo.

SOURCE CAVEAT (registry/irregularities.md IR-04): these are third-party mirrors of DrivenData files
(the organiser page is login-walled). The SHA-256 pins prove mirror consistency, not organiser authentication.
Only public git objects are used (github.com). Raw competition rasters are never committed to this repo.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GEMS_DATA_DIR", "/tmp/gems_data"))
PINS = json.loads((ROOT / "registry" / "data_pins.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_at(repo, sha_ref, dest):
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                        f"https://github.com/{repo}", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", sha_ref], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha_ref], check=True)
    return dest


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    ok = True
    for item in PINS["files"]:
        target = DATA / item["dest"]
        if not target.exists() or sha(target) != item["sha256"]:
            if item.get("parts"):
                repo_dir = git_at(item["repo"], item["ref"], DATA / "_mirror" / item["repo"].split("/")[1])
                with open(target, "wb") as out:
                    for part in item["parts"]:
                        out.write((repo_dir / part).read_bytes())
            else:
                repo_dir = git_at(item["repo"], item["ref"], DATA / "_mirror" / item["repo"].split("/")[1])
                target.write_bytes((repo_dir / item["path"]).read_bytes())
        got = sha(target)
        status = "OK  " if got == item["sha256"] else "FAIL"
        ok &= got == item["sha256"]
        print(f"{status} {item['id']:24s} {got[:16]}  {target}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
