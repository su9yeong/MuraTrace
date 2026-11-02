#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and unpack the MuraTrace data package.

  python tools/download.py --zenodo <record_id> --out data/MuraTrace          # archive copy (DOI)
  python tools/download.py --hf <org>/MuraTrace --out data/MuraTrace           # Hugging Face mirror

The Zenodo path needs only the standard library; the Hugging Face path needs `pip install huggingface_hub`.
After unpacking, run `python tools/verify_release.py --root data/MuraTrace --sha`.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"  exists, skipping {dst.name}")
        return
    print(f"  {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 2**20:8.0f} / {total / 2**20:.0f} MB", end="")
        print()
    tmp.rename(dst)


def from_zenodo(record: str, out: Path, keep_zips: bool) -> None:
    api = f"https://zenodo.org/api/records/{record}"
    with urllib.request.urlopen(api) as r:
        meta = json.load(r)
    files = meta.get("files", [])
    print(f"Zenodo record {record}: {meta.get('metadata', {}).get('title', '')} ({len(files)} files)")
    zips = out / "_downloads"
    for f in files:
        name = f.get("key") or f.get("filename")
        url = f["links"].get("self") or f["links"].get("download")
        download(url, zips / name)
    for z in sorted(zips.glob("*.zip")):
        print(f"  unpacking {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        if not keep_zips:
            z.unlink()
    for other in zips.iterdir() if zips.exists() else []:
        if other.suffix != ".zip":
            other.replace(out / other.name)
    print(f"done -> {out}")


def from_hf(repo: str, out: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")
    path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(out))
    print(f"done -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", help="Zenodo record id (the number at the end of the DOI)")
    ap.add_argument("--hf", help="Hugging Face dataset repo id, e.g. <org>/MuraTrace")
    ap.add_argument("--out", default="data/MuraTrace")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.zenodo:
        from_zenodo(args.zenodo, out, args.keep_zips)
    elif args.hf:
        from_hf(args.hf, out)
    else:
        ap.error("give --zenodo <record_id> or --hf <repo>")


if __name__ == "__main__":
    main()
