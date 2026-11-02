#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify a MuraTrace data package (as built by tools/make_release.py or as downloaded).

Checks
  * manifest rows, referenced files exist, PNG dtype/shape (uint16 image, uint8 masks, uint16 inst)
  * dataset statistics against the paper: 30,240 panels, 540 glasses, 11,941 defective panels,
    5,736 visible instances (1,675 reject; spot 2600 / line 2131 / region 638 / gradient 187 / band 180)
  * split sizes (background-level 18/6/6 backgrounds) and the glass-disjoint split file
  * optional: SHA256SUMS (--sha), bit-exact comparison with the generation output (--src <v1 root>, random sample)

Usage
  python tools/verify_release.py --root release/MuraTrace_v1.0 [--sha] [--src data/mura_rca_wacv_realbg_visible] [--sample 200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

PAPER = {"panels": 30240, "glasses": 540, "defective_panels": 11928, "visible_instances": 5736,
         "reject_instances": 1675, "reject_instances_panel_semu": 1715, "reject_instances_expert": 2188,
         "per_class": {"spot": 2600, "line": 2131, "region": 638, "gradient": 187, "band": 180},
         "backgrounds": {"train": 18, "val": 6, "test": 6}}
# Definitions (paper Sec. 3.5 / Supp. S5): an instance is identified by (glass_id, instance_id) and counted once even
# when it spans several panels; "visible" = any panel record with is_visible; "reject" = any record with is_reject
# (label-policy flag on `semu`); the expert-threshold comparison of Sec. 3.3 uses panel_instance_semu
# (1,715 author -> 2,188 expert, +28%); "defective panels" = panels with a locally visible (training-target) defect.


def read_png(path: Path):
    if cv2 is None:
        from PIL import Image
        return np.asarray(Image.open(path))
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sha", action="store_true", help="verify SHA256SUMS (slow)")
    ap.add_argument("--src", default=None, help="generation output root for a bit-exact sample comparison")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--expect-rows", type=int, default=PAPER["panels"])
    args = ap.parse_args()
    root = Path(args.root)
    problems: list[str] = []

    rows = [json.loads(l) for l in open(root / "manifest.jsonl", encoding="utf-8") if l.strip()]
    print(f"manifest rows: {len(rows)}")
    if len(rows) != args.expect_rows:
        problems.append(f"expected {args.expect_rows} rows, found {len(rows)}")

    # files + dtypes on a sample, existence on all
    missing = 0
    keys = ["image", "mask", "train_mask", "inst", "boxes", "train_boxes", "route_graph"]
    for r in rows:
        for k in keys:
            if k in r and not (root / r[k]).exists():
                missing += 1
    print(f"missing referenced files: {missing}")
    if missing:
        problems.append(f"{missing} referenced files missing")
    rng = random.Random(0)
    sample = rng.sample(rows, min(args.sample, len(rows)))
    bad_dtype = 0
    for r in sample:
        img = read_png(root / r["image"])
        msk = read_png(root / r["mask"])
        ins = read_png(root / r["inst"])
        if img is None or img.dtype != np.uint16 or img.shape != (359, 727):
            bad_dtype += 1
        if msk is None or msk.dtype != np.uint8 or msk.max() > 5:
            bad_dtype += 1
        if ins is None or ins.dtype != np.uint16:
            bad_dtype += 1
    print(f"sample of {len(sample)} panels: dtype/shape problems = {bad_dtype}")
    if bad_dtype:
        problems.append(f"{bad_dtype} dtype/shape problems in the sample")

    # statistics
    glasses = {r["glass_id"] for r in rows}
    defective = sum(1 for r in rows if r.get("panel_has_train_defect"))
    vis_set, rej_set, rej_ps_set, rej_e_set = set(), set(), set(), set()
    cls = {}
    for r in rows:
        for d in r.get("defects", []):
            k = (r["glass_id"], d.get("instance_id"))
            if d.get("is_visible"):
                vis_set.add(k)
                cls[k] = d["mask_type"]
                if d.get("is_reject"):
                    rej_set.add(k)
                if d.get("instance_is_reject"):
                    rej_ps_set.add(k)
                if d.get("is_reject_expert"):
                    rej_e_set.add(k)
    vis, rej, rej_ps, rej_expert = len(vis_set), len(rej_set), len(rej_ps_set), len(rej_e_set)
    per_class: Counter = Counter(cls.values())
    print(f"glasses {len(glasses)} (paper {PAPER['glasses']}); panels with a locally visible defect {defective} (paper {PAPER['defective_panels']})")
    print(f"unique visible instances {vis} (paper {PAPER['visible_instances']}); reject {rej} (paper {PAPER['reject_instances']}); "
          f"reject on panel_instance_semu {rej_ps} (paper 1,715) -> expert thresholds {rej_expert} (paper 2,188, +28%)")
    print("per class:", dict(per_class), "(paper", PAPER["per_class"], ")")
    if len(rows) == PAPER["panels"]:
        if len(glasses) != PAPER["glasses"]:
            problems.append("glass count differs from the paper")
        if defective != PAPER["defective_panels"]:
            problems.append("defective-panel count differs from the paper")
        if vis != PAPER["visible_instances"] or rej != PAPER["reject_instances"]:
            problems.append("instance counts differ from the paper")
        if rej_ps != PAPER["reject_instances_panel_semu"] or rej_expert != PAPER["reject_instances_expert"]:
            problems.append("expert-threshold reject counts differ from the paper (1,715 -> 2,188)")
        if dict(per_class) != PAPER["per_class"]:
            problems.append("per-class instance counts differ from the paper")

    # splits
    cap = json.load(open(root / "splits" / "background_level.json", encoding="utf-8"))
    n_caps = {k: len(v) for k, v in cap["backgrounds"].items()}
    print("background-level split:", n_caps, "panels per split:", Counter(r["split"] for r in rows))
    if len(rows) == PAPER["panels"] and n_caps != PAPER["backgrounds"]:
        problems.append(f"background split sizes {n_caps} differ from 18/6/6")
    bad_split = sum(1 for r in rows if cap["source_to_split"].get(r["source_id"]) != r["split"])
    if bad_split:
        problems.append(f"{bad_split} rows whose split disagrees with splits/background_level.json")
    gd = root / "splits" / "glass_disjoint.json"
    if gd.exists():
        g = json.load(open(gd, encoding="utf-8"))
        print("glass-disjoint split:", Counter(g["glass_to_split"].values()))
        if set(g["glass_to_split"]) != glasses:
            problems.append("glass_disjoint.json does not cover exactly the released glasses")

    # ambiguous labels
    amb = root / "ambiguous" / "manifest.jsonl"
    if amb.exists():
        arows = [json.loads(l) for l in open(amb, encoding="utf-8") if l.strip()]
        print(f"ambiguous-cause manifest rows: {len(arows)}")
        if len(arows) != len(rows):
            problems.append("ambiguous manifest row count differs from the main manifest")
        amissing = sum(1 for r in arows for k in ("boxes", "route_graph") if not (root / r[k]).exists())
        if amissing:
            problems.append(f"{amissing} ambiguous label files missing")

    # bit-exact sample against the generation output
    if args.src:
        src = Path(args.src)
        src_rows = {}
        for l in open(src / "manifest.jsonl", encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                src_rows[r["panel_id"]] = r
        mism = 0
        for r in sample:
            s = src_rows[r["panel_id"]]
            for k in ("image", "mask", "train_mask", "inst"):
                a = np.load(src / s[k].replace("\\", "/"))
                b = read_png(root / r[k])
                if b is None or not np.array_equal(a, b):
                    mism += 1
        print(f"bit-exact sample vs source: mismatches = {mism} over {4 * len(sample)} arrays")
        if mism:
            problems.append(f"{mism} arrays differ from the generation output")

    # checksums
    if args.sha:
        sums = root / "SHA256SUMS"
        bad = 0
        n = 0
        for line in open(sums, encoding="utf-8"):
            if not line.strip():
                continue
            h, rel = line.rstrip("\n").split("  ", 1)
            p = root / rel
            n += 1
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
                bad += 1
        print(f"SHA256SUMS: {n} entries, {bad} mismatching")
        if bad:
            problems.append(f"{bad} checksum mismatches")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        return 1
    print("\nOK: release verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
