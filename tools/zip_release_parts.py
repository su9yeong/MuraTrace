#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pack a built MuraTrace release directory into Zenodo-sized zip parts (default cap 1.9 GB each).

All archives extract into the same tree, so unpacking every part into one folder restores the package.
PNG members are stored uncompressed (they are already compressed); JSON/text members are deflated.

  python tools/zip_release_parts.py --root release_build/MuraTrace_v1.0 --out release_build/zenodo_upload
"""
import argparse, hashlib, os, time, zipfile
from pathlib import Path

VERSION = "1.0"

GROUPS = [
    # (name, subpaths, splittable)
    ("meta", ["manifest.jsonl", "README.md", "DATASHEET.md", "dataset_card.md", "LICENSE", "SHA256SUMS",
              "release_summary.json", "splits", "configs", "expert_study", "previews", "glass"], False),
    ("ambiguous", ["ambiguous"], False),
    ("backgrounds", ["backgrounds"], False),
    ("panels_boxes", ["panels/boxes", "panels/train_boxes", "panels/yolo", "panels/train_yolo"], False),
    ("panels_masks", ["panels/masks", "panels/train_masks", "panels/inst"], True),
    ("panels_images", ["panels/images"], True),
]
STORED_EXT = {".png", ".jpg", ".jpeg", ".zip", ".npz"}


def iter_files(root: Path, subpaths):
    for sp in subpaths:
        p = root / sp
        if p.is_file():
            yield p
        elif p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.is_file():
                    yield q


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_zip(zpath: Path, root: Path, files, log):
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as zf:
        for i, f in enumerate(files, 1):
            mode = zipfile.ZIP_STORED if f.suffix.lower() in STORED_EXT else zipfile.ZIP_DEFLATED
            zf.write(f, f.relative_to(root).as_posix(), compress_type=mode)
            if i % 20000 == 0:
                log(f"    {zpath.name}: {i}/{len(files)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-gb", type=float, default=1.9)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = int(a.cap_gb * 2**30)
    t0 = time.time()
    logf = open(out / "zip_build.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    covered = set()
    plan = []  # (zipname, files)
    for name, subs, splittable in GROUPS:
        files = list(iter_files(root, subs))
        covered.update(files)
        if not files:
            log(f"  group {name}: nothing found, skipped"); continue
        if not splittable:
            plan.append((f"MuraTrace_v{VERSION}_{name}.zip", files)); continue
        parts, cur, size = [], [], 0
        for f in files:
            s = f.stat().st_size
            if cur and size + s > cap:
                parts.append(cur); cur, size = [], 0
            cur.append(f); size += s
        if cur:
            parts.append(cur)
        for k, part in enumerate(parts, 1):
            plan.append((f"MuraTrace_v{VERSION}_{name}_part{k:02d}of{len(parts):02d}.zip", part))
    everything = {q for q in root.rglob("*") if q.is_file()}
    left = sorted(everything - covered)
    if left:
        log(f"  WARNING: {len(left)} files not covered by any group, adding them to meta: {[p.relative_to(root).as_posix() for p in left[:5]]}")
        for zname, files in plan:
            if zname.endswith("_meta.zip"):
                files.extend(left)

    manifest_lines = []
    for zname, files in plan:
        zpath = out / zname
        n_bytes = sum(f.stat().st_size for f in files)
        log(f"  writing {zname}: {len(files)} files, {n_bytes / 2**30:.2f} GB uncompressed")
        write_zip(zpath, root, files, log)
        zs = zpath.stat().st_size
        manifest_lines.append(f"{zname}\t{len(files)} files\t{zs / 2**30:.2f} GB")
        log(f"  done {zname}: {zs / 2**30:.2f} GB ({time.time() - t0:.0f} s elapsed)")

    (out / "PARTS.txt").write_text(
        "MuraTrace v1.0 data package, zip parts. Unpack every archive into the same folder.\n"
        "Verify with: python tools/verify_release.py --root <folder> --sha  (code repository)\n\n"
        + "\n".join(manifest_lines) + "\n", encoding="utf-8")
    with open(out / "ZIP_SHA256SUMS", "w", encoding="utf-8") as f:
        for zname, _ in plan:
            f.write(f"{sha256(out / zname)}  {zname}\n")
    log(f"all parts written in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
