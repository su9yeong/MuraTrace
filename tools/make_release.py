#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the public MuraTrace data package from the generation output.

Source = the dataset folder the paper's numbers were computed on
(data/mura_rca_wacv_realbg_visible) plus the ambiguous-cause label overlay
(data/mura_rca_wacv_ambiguous), which shares the same panels.

What it does
  * panel images / semantic masks / training masks / instance maps: .npy -> lossless PNG
    (uint16, uint8, uint8, uint16; bit-exact, verified by tools/verify_release.py)
  * label files (boxes, train_boxes, yolo, train_yolo): copied; every box/defect record gets
    is_visible_expert / is_reject_expert from configs/muratrace_thresholds_expert.json
  * manifest.jsonl rewritten with portable '/' paths, new file names, no internal file names
  * ambiguous/: the primary-protocol labels (boxes, route graphs, manifest) pointing at the same panels
  * glass/route_graphs (default configuration), optional glass images, optional backgrounds
  * splits/capture_level.json (the paper's protocol) and splits/glass_disjoint.json
  * configs, expert-study files, previews, README / LICENSE / DATASHEET, SHA256SUMS
  * optional zip parts for Zenodo (--zip DIR)

Usage
  python tools/make_release.py --src data/mura_rca_wacv_realbg_visible \
      --ambiguous data/mura_rca_wacv_ambiguous --out release_build/MuraTrace_v1.0 \
      [--with-backgrounds] [--with-glass-images] [--limit 60] [--compression 6] [--no-sha] [--zip DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
    from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VERSION = "1.0"
PANEL_KEYS = {  # manifest key -> (release sub-folder, new extension, kind)
    "image": ("panels/images", ".png", "array"),
    "mask": ("panels/masks", ".png", "array"),
    "train_mask": ("panels/train_masks", ".png", "array"),
    "inst": ("panels/inst", ".png", "array"),
    "boxes": ("panels/boxes", ".json", "boxes"),
    "train_boxes": ("panels/train_boxes", ".json", "boxes"),
    "yolo": ("panels/yolo", ".txt", "copy"),
    "train_yolo": ("panels/train_yolo", ".txt", "copy"),
}
AMB_KEYS = {"boxes": "ambiguous/boxes", "train_boxes": "ambiguous/train_boxes",
            "yolo": "ambiguous/yolo", "train_yolo": "ambiguous/train_yolo"}
DROP_KEYS = ("source_file",)


# ----------------------------------------------------------------------------- helpers
def p(s: str) -> str:
    return s.replace("\\", "/")


def load_json(path: Path):
    return json.load(open(path, encoding="utf-8"))


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def visibility_threshold(th: dict, mask_type: str, subtype: str) -> dict:
    d = dict(th.get("default", {"engineer_visible": 0.5, "reject_visible": 1.0}))
    d.update(th.get("by_mask_type", {}).get(mask_type, {}))
    d.update(th.get("by_subtype", {}).get(subtype, {}))
    return {"engineer_visible": float(d["engineer_visible"]), "reject_visible": float(d["reject_visible"])}


def add_expert_flags(rec: dict, th_expert: dict) -> dict:
    thr = visibility_threshold(th_expert, str(rec.get("mask_type", "")), str(rec.get("subtype", "")))
    field = th_expert.get("applies_to", "panel_instance_semu")
    semu = float(rec.get(field, rec.get("semu", rec.get("panel_label_semu", 0.0))) or 0.0)
    rec["expert_engineer_visible_threshold"] = thr["engineer_visible"]
    rec["expert_reject_visible_threshold"] = thr["reject_visible"]
    rec["is_visible_expert"] = bool(semu >= thr["engineer_visible"])
    rec["is_reject_expert"] = bool(semu >= thr["reject_visible"])
    return rec


def write_png(arr: np.ndarray, dst: Path, compression: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        if not cv2.imwrite(str(dst), arr, [cv2.IMWRITE_PNG_COMPRESSION, compression]):
            raise RuntimeError(f"cv2.imwrite failed for {dst}")
    else:
        Image.fromarray(arr).save(dst, compress_level=compression)


def convert_array(src: Path, dst: Path, compression: int) -> None:
    arr = np.load(src)
    if arr.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"{src}: dtype {arr.dtype} is not PNG-representable without loss")
    write_png(arr, dst, compression)


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def glass_disjoint_split(glass_id: str, train_ratio: float = 0.6, val_ratio: float = 0.2) -> str:
    """Same rule as training/mura_rca_data.py::assigned_split(row, 'random_glass')."""
    u = (int(hashlib.md5(glass_id.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000) / 1_000_000.0
    return "train" if u < train_ratio else ("val" if u < train_ratio + val_ratio else "test")


def sha256sums(root: Path, out_file: Path) -> int:
    n = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for path in sorted(q for q in root.rglob("*") if q.is_file() and q.name != out_file.name):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            f.write(f"{h.hexdigest()}  {p(str(path.relative_to(root)))}\n")
            n += 1
    return n


def folder_size(root: Path) -> dict:
    sizes: dict = defaultdict(int)
    for q in root.rglob("*"):
        if q.is_file():
            top = q.relative_to(root).parts[0]
            sizes[top] += q.stat().st_size
    return {k: round(v / 2**30, 3) for k, v in sorted(sizes.items())}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="generation output used by the paper (data/mura_rca_wacv_realbg_visible)")
    ap.add_argument("--ambiguous", default=None, help="ambiguous-cause label overlay (data/mura_rca_wacv_ambiguous)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N manifest rows only (smoke test)")
    ap.add_argument("--compression", type=int, default=6)
    ap.add_argument("--with-backgrounds", action="store_true", help="include the 30 clean + 540 augmented background captures")
    ap.add_argument("--with-glass-images", action="store_true")
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--zip", default=None, help="also write zip parts into this directory")
    ap.add_argument("--org", default="ORG", help="GitHub org/user for the README links")
    ap.add_argument("--doi", default="10.5281/zenodo.XXXXXXX")
    args = ap.parse_args()

    t0 = time.time()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    th_author = load_json(REPO / "configs" / "muratrace_thresholds.json")
    th_expert = load_json(REPO / "configs" / "muratrace_thresholds_expert.json")

    rows = [json.loads(l) for l in open(src / "manifest.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    pids = {r["panel_id"] for r in rows}
    glasses = sorted({r["glass_id"] for r in rows})
    print(f"source rows: {len(rows)}, glasses: {len(glasses)}")

    # --- panels + manifest ------------------------------------------------------------
    man_out = open(out / "manifest.jsonl", "w", encoding="utf-8")
    stats = Counter()
    for i, r in enumerate(rows):
        pid, gid = r["panel_id"], r["glass_id"]
        new = dict(r)
        for key, (sub, ext, kind) in PANEL_KEYS.items():
            if key not in r or not r[key]:
                continue
            s = src / p(r[key])
            d = out / sub / f"{pid}{ext}"
            if kind == "array":
                convert_array(s, d, args.compression)
            elif kind == "boxes":
                obj = load_json(s)
                for b in obj.get("boxes", []):
                    add_expert_flags(b, th_expert)
                dump_json(obj, d)
            else:
                copy(s, d)
            new[key] = f"{sub}/{pid}{ext}"
            stats[key] += 1
        new["route_graph"] = f"glass/route_graphs/{gid}.json"
        for k in DROP_KEYS:
            new.pop(k, None)
        for dft in new.get("defects", []) or []:
            add_expert_flags(dft, th_expert)
        man_out.write(json.dumps(new) + "\n")
        if (i + 1) % 1000 == 0 or i + 1 == len(rows):
            print(f"  panels {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)")
    man_out.close()

    # --- glass --------------------------------------------------------------------------
    for gid in glasses:
        copy(src / "glass" / "route_graphs" / f"{gid}.json", out / "glass" / "route_graphs" / f"{gid}.json")
        if args.with_glass_images:
            cands = list((src / "glass" / "images").glob(f"{gid}*"))
            for c in cands:
                if c.suffix == ".npy":
                    convert_array(c, out / "glass" / "images" / f"{gid}.png", args.compression)
                else:
                    copy(c, out / "glass" / "images" / c.name)

    # --- ambiguous-cause overlay --------------------------------------------------------
    if args.ambiguous:
        amb = Path(args.ambiguous)
        arows = [json.loads(l) for l in open(amb / "manifest.jsonl", encoding="utf-8") if l.strip()]
        arows = [r for r in arows if r["panel_id"] in pids]
        (out / "ambiguous").mkdir(parents=True, exist_ok=True)
        with open(out / "ambiguous" / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in arows:
                pid, gid = r["panel_id"], r["glass_id"]
                new = dict(r)
                for key, (sub, ext, kind) in PANEL_KEYS.items():
                    if key not in r or not r[key]:
                        continue
                    val = p(r[key])
                    if val.startswith("../") or key in ("image", "mask", "train_mask", "inst"):
                        new[key] = f"{sub}/{pid}{ext}"  # shared panel file of the main package
                        continue
                    s = amb / val
                    asub = AMB_KEYS[key]
                    d = out / asub / f"{pid}{ext}"
                    if kind == "boxes":
                        obj = load_json(s)
                        for b in obj.get("boxes", []):
                            add_expert_flags(b, th_expert)
                        dump_json(obj, d)
                    else:
                        copy(s, d)
                    new[key] = f"{asub}/{pid}{ext}"
                new["route_graph"] = f"ambiguous/route_graphs/{gid}.json"
                for k in DROP_KEYS:
                    new.pop(k, None)
                for dft in new.get("defects", []) or []:
                    add_expert_flags(dft, th_expert)
                f.write(json.dumps(new) + "\n")
        for gid in glasses:
            copy(amb / "route_graphs" / f"{gid}.json", out / "ambiguous" / "route_graphs" / f"{gid}.json")
        amb_cfg = load_json(amb / "config.json")
        amb_cfg.get("runtime", {}).pop("output_dir", None)
        dump_json(amb_cfg, out / "ambiguous" / "config.json")
        print(f"  ambiguous overlay: {len(arows)} rows, {len(glasses)} route graphs")

    # --- backgrounds ---------------------------------------------------------------------
    if args.with_backgrounds:
        for q in sorted((src / "backgrounds_clean").glob("*.png")):
            copy(q, out / "backgrounds" / "clean" / q.name)
        for q in sorted((src / "backgrounds_aug").glob("*.png")):
            copy(q, out / "backgrounds" / "aug" / q.name)
        with open(out / "backgrounds" / "backgrounds_manifest.jsonl", "w", encoding="utf-8") as f:
            for l in open(src / "backgrounds_manifest.jsonl", encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                r.pop("source_file", None)
                r["clean_path"] = f"backgrounds/clean/{Path(p(r['clean_path'])).name}"
                r["aug_path"] = f"backgrounds/aug/{Path(p(r['aug_path'])).name}"
                f.write(json.dumps(r) + "\n")
        print("  backgrounds: 30 clean + 540 augmented")

    # --- configs, docs, extras -----------------------------------------------------------
    for q in sorted((REPO / "configs").glob("*.json")):
        copy(q, out / "configs" / q.name)
    cfg = load_json(src / "config.json")
    rt = cfg.get("runtime", {})
    for k in list(rt):
        if "dir" in k or "path" in k or "input" in k:
            rt.pop(k)
    dump_json(cfg, out / "configs" / "generation_config_as_run.json")
    if (REPO / "expert_study").exists():
        for q in (REPO / "expert_study").iterdir():
            if q.is_file():
                copy(q, out / "expert_study" / q.name)
    if (src / "previews").exists():
        for q in sorted((src / "previews").glob("*.png")):
            copy(q, out / "previews" / q.name)
    if (src / "dataset_card.md").exists():
        copy(src / "dataset_card.md", out / "dataset_card.md")
    for name in ("DATASHEET.md", "release_LICENSE_CC-BY-4.0.txt"):
        q = REPO / "docs" / name
        if q.exists():
            copy(q, out / ("LICENSE" if name.startswith("release_LICENSE") else name))

    # --- splits ----------------------------------------------------------------------------
    s2s: dict = {}
    for r in rows:
        s2s.setdefault(r["source_id"], r["split"])
        if s2s[r["source_id"]] != r["split"]:
            raise ValueError(f"capture {r['source_id']} appears in two splits")
    captures = {sp: sorted(k for k, v in s2s.items() if v == sp) for sp in ("train", "val", "test")}
    dump_json({"protocol": "capture-level (background-source-disjoint): every capture and all panels derived from it belong to one split",
               "captures": captures, "source_to_split": dict(sorted(s2s.items())),
               "panels_per_split": dict(Counter(r["split"] for r in rows))}, out / "splits" / "capture_level.json")
    g2s = {g: glass_disjoint_split(g) for g in glasses}
    dump_json({"protocol": "glass-disjoint: md5(glass_id)[:8] -> u in [0,1); train if u<0.6, val if u<0.8, else test "
                           "(training/mura_rca_data.py, split_mode=random_glass)",
               "glass_to_split": g2s, "glasses_per_split": dict(Counter(g2s.values()))}, out / "splits" / "glass_disjoint.json")

    # --- README --------------------------------------------------------------------------
    n_vis_records = sum(1 for r in rows for d in (r.get("defects") or []) if d.get("is_visible"))
    n_vis = len({(r.get("glass_id"), d.get("instance_id")) for r in rows for d in (r.get("defects") or [])
                 if d.get("is_visible")})
    tmpl = REPO / "docs" / "release_README_template.md"
    if tmpl.exists():
        txt = tmpl.read_text(encoding="utf-8")
        for k, v in {"{VERSION}": VERSION, "{DATE}": time.strftime("%Y-%m-%d"), "{N_PANELS}": f"{len(rows):,}",
                     "{N_GLASSES}": f"{len(glasses):,}", "{N_INSTANCES}": f"{n_vis:,}", "{ORG}": args.org, "{DOI}": args.doi}.items():
            txt = txt.replace(k, v)
        (out / "README.md").write_text(txt, encoding="utf-8")

    # --- checksums + summary ------------------------------------------------------------
    n_sha = 0 if args.no_sha else sha256sums(out, out / "SHA256SUMS")
    summary = {"version": VERSION, "source": str(src), "rows": len(rows), "glasses": len(glasses),
               "visible_instances": n_vis, "visible_instance_records": n_vis_records,
               "files_converted": dict(stats), "sha256_entries": n_sha,
               "sizes_gb": folder_size(out), "elapsed_s": round(time.time() - t0, 1),
               "with_backgrounds": args.with_backgrounds, "with_glass_images": args.with_glass_images}
    dump_json(summary, out / "release_summary.json")
    print(json.dumps(summary, indent=1))

    # --- zip parts -------------------------------------------------------------------------
    if args.zip:
        zdir = Path(args.zip)
        zdir.mkdir(parents=True, exist_ok=True)
        parts = {
            f"MuraTrace_v{VERSION}_panels_images.zip": (["panels/images"], zipfile.ZIP_STORED),
            f"MuraTrace_v{VERSION}_panels_labels.zip": (["panels/masks", "panels/train_masks", "panels/inst", "panels/boxes",
                                                         "panels/train_boxes", "panels/yolo", "panels/train_yolo"], zipfile.ZIP_DEFLATED),
            f"MuraTrace_v{VERSION}_glass_and_ambiguous.zip": (["glass", "ambiguous"], zipfile.ZIP_DEFLATED),
            f"MuraTrace_v{VERSION}_backgrounds.zip": (["backgrounds"], zipfile.ZIP_STORED),
        }
        covered = {q for _, (subs, _) in parts.items() for q in subs}
        meta = [q.name for q in out.iterdir() if q.is_file()] + [q.name for q in out.iterdir() if q.is_dir() and not any(
            (q.name == c or c.startswith(q.name + "/")) for c in covered)]
        parts[f"MuraTrace_v{VERSION}_meta.zip"] = (meta, zipfile.ZIP_DEFLATED)
        for zname, (subs, mode) in parts.items():
            files = []
            for s in subs:
                q = out / s
                if q.is_file():
                    files.append(q)
                elif q.is_dir():
                    files += [f for f in q.rglob("*") if f.is_file()]
            if not files:
                continue
            with zipfile.ZipFile(zdir / zname, "w", compression=mode, allowZip64=True) as zf:
                for f in files:
                    zf.write(f, arcname=f"MuraTrace_v{VERSION}/{p(str(f.relative_to(out)))}")
            print(f"  zip {zname}: {len(files)} files, {(zdir / zname).stat().st_size / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
