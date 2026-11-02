# -*- coding: utf-8 -*-
"""Per-real-background breakdown of a segmentation checkpoint on a split.

Camera-ready analysis (reviewers sDje / Sogs, "are 30 backgrounds enough / does
the model generalize to unseen backgrounds?"): the test split consists of six
real backgrounds that are never seen in training, so a per-source breakdown of
the test metrics measures generalization to each unseen texture separately.

For every source id in the split we accumulate the 6x6 pixel confusion (ignore
label 255 excluded, same `train_mask` protocol as the paper), and report per-class
IoU, mIoU over the five defect classes, the defect-pixel false-positive rate on
the source's defect-free panels, and the panel-level false-alarm rate.

  python training/eval_seg_per_source.py \
      --checkpoint runs_wacv_realbg_visible/seg_adv_swin_small/best.pt \
      --data_root data/mura_rca_wacv_realbg_visible --split test \
      --out analysis_camera_ready/seg_per_source_swin.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mura_rca_data import SEG_IGNORE_INDEX, assigned_split, load_manifest, normalize_image, rel_path, resize_tensor_image, resize_tensor_mask, row_path_value
from seg_advanced_models import build_advanced_segmentation_model

CLASSES = ["background", "spot", "line", "region", "gradient", "band"]


def iou_from_conf(conf: np.ndarray) -> dict:
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    present = conf.sum(1) > 0  # class has ground-truth pixels in this group
    out = {f"iou_{CLASSES[c]}": (float(iou[c]) if present[c] else None) for c in range(6)}
    def_ious = [iou[c] for c in range(1, 6) if present[c]]
    out["miou_defect_present"] = float(np.mean(def_ious)) if def_ious else None
    out["n_defect_classes_present"] = int(len(def_ious))
    out["pixel_acc"] = float(tp.sum() / max(conf.sum(), 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--alarm_frac", type=float, default=0.001, help="clean panel = false alarm if > this fraction of pixels is predicted defect")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = dict(ckpt.get("args", {}))
    image_size = int(ckpt_args.get("image_size", 512))
    split_mode = str(ckpt_args.get("split_mode", "manifest"))
    mask_key = str(ckpt_args.get("mask_key", "train_mask"))
    amp_dtype = torch.bfloat16 if str(ckpt_args.get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_advanced_segmentation_model(
        arch=str(ckpt_args.get("arch", "deeplabv3plus")), encoder_name=str(ckpt_args.get("encoder_name") or "resnet34"),
        num_classes=6, pretrained=False, in_channels=int(ckpt_args.get("in_channels", 3)), image_size=image_size,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(args.data_root)
    rows = [r for r in load_manifest(root) if assigned_split(r, split_mode) == args.split]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} panels in split={args.split}; sources={sorted({r['source_id'] for r in rows})}", flush=True)

    conf = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
    clean = defaultdict(lambda: {"panels": 0, "pixels": 0, "defect_pixels": 0, "alarms": 0})
    bs = max(1, args.batch_size)
    with torch.no_grad():
        for start in range(0, len(rows), bs):
            chunk = rows[start:start + bs]
            xs, ys = [], []
            for r in chunk:
                img = normalize_image(np.load(rel_path(root, r["image"])))
                msk = np.load(rel_path(root, row_path_value(r, mask_key, "mask"))).astype(np.int64)
                xs.append(resize_tensor_image(torch.from_numpy(img)[None], image_size))
                ys.append(resize_tensor_mask(torch.from_numpy(msk).long(), image_size))
            x = torch.stack(xs).to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(x)
            pred = torch.argmax(logits.float(), dim=1).cpu().numpy()
            for r, p, y in zip(chunk, pred, ys):
                y = y.numpy()
                valid = y != SEG_IGNORE_INDEX
                c = np.bincount(6 * y[valid] + p[valid], minlength=36).reshape(6, 6)
                conf[r["source_id"]] += c
                conf["__all__"] += c
                if not bool(r.get("panel_has_train_defect", r.get("panel_has_defect", False))):
                    d = int((p > 0).sum())
                    for key in (r["source_id"], "__all__"):
                        cl = clean[key]
                        cl["panels"] += 1; cl["pixels"] += int(p.size); cl["defect_pixels"] += d
                        cl["alarms"] += int(d / p.size > args.alarm_frac)
            if (start // bs) % 50 == 0:
                print(f"  {min(start + bs, len(rows))}/{len(rows)}", flush=True)

    result = {"checkpoint": args.checkpoint, "split": args.split, "n_panels": len(rows), "per_source": {}, "all": {}}
    for key in sorted(conf):
        m = iou_from_conf(conf[key])
        cl = clean[key]
        m["clean_panels"] = cl["panels"]
        m["clean_fp_pixel_rate_pct"] = 100.0 * cl["defect_pixels"] / max(cl["pixels"], 1)
        m["clean_false_alarm_rate_pct"] = 100.0 * cl["alarms"] / max(cl["panels"], 1)
        m["n_panels"] = int(sum(1 for r in rows if r["source_id"] == key)) if key != "__all__" else len(rows)
        m["confusion"] = conf[key].tolist()  # 6x6 [gt, pred] so subsets of sources can be re-aggregated offline
        if key == "__all__":
            result["all"] = m
        else:
            result["per_source"][key] = m
    vals = [v["miou_defect_present"] for v in result["per_source"].values() if v["miou_defect_present"] is not None]
    result["per_source_miou_mean"] = float(np.mean(vals)) if vals else None
    result["per_source_miou_std"] = float(np.std(vals)) if vals else None
    result["per_source_miou_min"] = float(np.min(vals)) if vals else None
    result["per_source_miou_max"] = float(np.max(vals)) if vals else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("per_source_miou_mean", "per_source_miou_std", "per_source_miou_min", "per_source_miou_max")}, indent=1))
    def _f(x):
        return "n/a" if x is None else f"{x:.3f}"

    for k, v in result["per_source"].items():
        print(f"  {k}: mIoU_def={_f(v['miou_defect_present'])} spot={_f(v['iou_spot'])} line={_f(v['iou_line'])} region={_f(v['iou_region'])} "
              f"gradient={_f(v['iou_gradient'])} band={_f(v['iou_band'])} cleanFP={v['clean_fp_pixel_rate_pct']:.3f}% alarms={v['clean_false_alarm_rate_pct']:.2f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
