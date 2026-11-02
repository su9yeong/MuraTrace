#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute the expert-threshold flags of an existing data package in place.

Rewrites is_visible_expert / is_reject_expert (and the two expert_*_threshold fields) in
panels/boxes, panels/train_boxes, ambiguous/boxes, ambiguous/train_boxes and in the
`defects[]` of manifest.jsonl and ambiguous/manifest.jsonl, from the thresholds file given,
then rebuilds SHA256SUMS. Panel arrays are not touched.

  python tools/update_expert_flags.py --root release_build/MuraTrace_v1.0 \
      --thresholds configs/muratrace_thresholds_expert.json [--no-sha]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def visibility_threshold(th: dict, mask_type: str, subtype: str) -> dict:
    d = dict(th.get("default", {"engineer_visible": 0.5, "reject_visible": 1.0}))
    d.update(th.get("by_mask_type", {}).get(mask_type, {}))
    d.update(th.get("by_subtype", {}).get(subtype, {}))
    return {"engineer_visible": float(d["engineer_visible"]), "reject_visible": float(d["reject_visible"])}


def flag(rec: dict, th: dict) -> None:
    t = visibility_threshold(th, str(rec.get("mask_type", "")), str(rec.get("subtype", "")))
    field = th.get("applies_to", "panel_instance_semu")
    semu = float(rec.get(field, rec.get("semu", rec.get("panel_label_semu", 0.0))) or 0.0)
    rec["expert_engineer_visible_threshold"] = t["engineer_visible"]
    rec["expert_reject_visible_threshold"] = t["reject_visible"]
    rec["is_visible_expert"] = bool(semu >= t["engineer_visible"])
    rec["is_reject_expert"] = bool(semu >= t["reject_visible"])


def rewrite_manifest(path: Path, th: dict) -> Counter:
    tmp = path.with_suffix(".tmp")
    stats: Counter = Counter()
    seen_vis, seen_rej, seen_rej_e = set(), set(), set()
    with open(path, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            for d in r.get("defects") or []:
                flag(d, th)
                k = (r["glass_id"], d.get("instance_id"))
                if d.get("is_visible"):
                    seen_vis.add(k)
                    if d.get("is_reject"):
                        seen_rej.add(k)
                    if d.get("is_reject_expert"):
                        seen_rej_e.add(k)
            fout.write(json.dumps(r) + "\n")
            stats["rows"] += 1
    tmp.replace(path)
    stats["unique_visible"] = len(seen_vis)
    stats["unique_reject"] = len(seen_rej)
    stats["unique_reject_expert"] = len(seen_rej_e)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--thresholds", default=str(REPO / "configs" / "muratrace_thresholds_expert.json"))
    ap.add_argument("--no-sha", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    th = json.load(open(args.thresholds, encoding="utf-8"))

    n_files = 0
    for sub in ("panels/boxes", "panels/train_boxes", "ambiguous/boxes", "ambiguous/train_boxes"):
        d = root / sub
        if not d.exists():
            continue
        for q in d.glob("*.json"):
            obj = json.load(open(q, encoding="utf-8"))
            for b in obj.get("boxes", []):
                flag(b, th)
            with open(q, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=1)
            n_files += 1
    print(f"box files rewritten: {n_files}")
    for m in ("manifest.jsonl", "ambiguous/manifest.jsonl"):
        if (root / m).exists():
            print(m, dict(rewrite_manifest(root / m, th)))
    shutil.copy2(args.thresholds, root / "configs" / Path(args.thresholds).name)

    if not args.no_sha:
        out_file = root / "SHA256SUMS"
        n = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for path in sorted(q for q in root.rglob("*") if q.is_file() and q.name != "SHA256SUMS"):
                h = hashlib.sha256()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                f.write(f"{h.hexdigest()}  {str(path.relative_to(root)).replace(chr(92), '/')}\n")
                n += 1
        print(f"SHA256SUMS rebuilt: {n} entries")


if __name__ == "__main__":
    main()
