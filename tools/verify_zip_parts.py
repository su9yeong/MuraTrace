#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify Zenodo zip parts without extracting them: every member is hashed while streaming from the zip
and compared with the package SHA256SUMS; missing and unexpected members are reported.

  python tools/verify_zip_parts.py --zips release_build/zenodo_upload --sums release_build/MuraTrace_v1.0/SHA256SUMS
"""
import argparse, hashlib, sys, time, zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True)
    ap.add_argument("--sums", required=True)
    a = ap.parse_args()
    expected = {}
    for line in Path(a.sums).read_text(encoding="utf-8").splitlines():
        if "  " in line:
            h, name = line.split("  ", 1)
            expected[name.strip().replace("\\", "/")] = h.strip()
    seen, bad, t0 = {}, [], time.time()
    zips = sorted(Path(a.zips).glob("*.zip"))
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            names = [i for i in zf.infolist() if not i.is_dir()]
            print(f"{zp.name}: {len(names)} members", flush=True)
            for k, info in enumerate(names, 1):
                h = hashlib.sha256()
                with zf.open(info) as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        h.update(chunk)
                name = info.filename
                if name in seen:
                    bad.append(f"duplicate member {name} ({zp.name} and {seen[name]})")
                seen[name] = zp.name
                exp = expected.get(name)
                if exp is None:
                    if name != "SHA256SUMS":
                        bad.append(f"not in SHA256SUMS: {name} ({zp.name})")
                elif exp != h.hexdigest():
                    bad.append(f"hash mismatch: {name} ({zp.name})")
                if k % 20000 == 0:
                    print(f"    {k}/{len(names)} ({time.time() - t0:.0f} s)", flush=True)
    missing = sorted(set(expected) - set(seen))
    print(f"members hashed: {len(seen)}; expected entries: {len(expected)}; missing: {len(missing)}; problems: {len(bad)}")
    for m in missing[:10]:
        print("  missing:", m)
    for b in bad[:10]:
        print("  ", b)
    if missing or bad:
        print("FAILED"); sys.exit(1)
    print("OK: every zip member matches SHA256SUMS and nothing is missing")


if __name__ == "__main__":
    main()
