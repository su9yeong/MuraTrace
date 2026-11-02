#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write the checkpoint table (file, paper model, size, SHA256) for the released weights-only checkpoints.

  python tools/checkpoint_index.py --dir release_build/checkpoints --out docs/checkpoints.md
"""
import argparse, hashlib, json
from pathlib import Path

PAPER = {
    "seg_adv_swin_small.pt": ("Task 1", "UPerNet Swin-S (Table 3, mIoU_def 0.871)"),
    "seg_adv_upernet_convnext_small.pt": ("Task 1", "UPerNet ConvNeXt-S (0.865)"),
    "seg_adv_deeplab_convnext_small.pt": ("Task 1", "DeepLabV3+ ConvNeXt-S (0.839)"),
    "seg_adv_segformer_mit_b2.pt": ("Task 1", "SegFormer MiT-b2 (0.807)"),
    "seg_adv_hrnet_w32.pt": ("Task 1", "UPerNet HRNet-W32 (0.646)"),
    "seg_adv_deeplab_r34_pretrained.pt": ("Task 1", "DeepLabV3+ ResNet-34, ImageNet init (0.645)"),
    "seg_adv_deeplab_r34_mae.pt": ("Task 1", "DeepLabV3+ ResNet-34, SimMIM init (0.633)"),
    "seg_deeplabv3plus_auto_w4.pt": ("Task 1", "DeepLabV3+ ResNet-34, class-weighted (0.655)"),
    "seg_deeplabv3plus_nofocal_unweighted.pt": ("Task 1", "DeepLabV3+ ResNet-34, unweighted (0.633)"),
    "rca_image_position.pt": ("Task 2, default configuration", "small CNN, image+position (Table 4 top, mean Hit@1 0.941)"),
    "rca_visionroute.pt": ("Task 2, default configuration", "small CNN, cross-fusion (0.900)"),
    "rca_image_only_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, image-only, seed 7"),
    "rca_image_position_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, image+position, seed 7"),
    "rca_image_route_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, image+route, seed 7"),
    "rca_route_only_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, route-only, seed 7"),
    "rca_late_fusion_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, late-fusion, seed 7"),
    "rca_rcfilm_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, FiLM-fusion, seed 7"),
    "rca_visionroute_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, cross-fusion, seed 7 (Table 4 bottom: three-seed mean 0.855)"),
    "rca_visionroute_swap_s7.pt": ("Task 2, ambiguous-cause", "ConvNeXt-S, cross-fusion with the route graph of another glass (swap control), seed 7"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    index = {}
    if (d / "index.json").exists():
        index = json.loads((d / "index.json").read_text(encoding="utf-8"))
        print("index.json:", json.dumps(index, indent=1)[:1500])
    rows = []
    for p in sorted(d.glob("*.pt")):
        task, model = PAPER.get(p.name, ("", p.stem))
        rows.append((p.name, task, model, p.stat().st_size / 2**20, sha256(p)))
    lines = ["# Baseline checkpoints (weights only)", "",
             "Released as assets of GitHub release v1.0 of https://github.com/su9yeong/MuraTrace.",
             "Load with the evaluation scripts in `training/` (segmentation: `evaluate_segmentation_checkpoint.py`,",
             "attribution: `evaluate_rca_checkpoint.py`); the Task 3 regressors are not released as checkpoints",
             "(their predictions are in `results/`).", "",
             "| file | task | model (paper) | size (MB) | SHA256 |", "|---|---|---|---:|---|"]
    for name, task, model, mb, h in rows:
        lines.append(f"| `{name}` | {task} | {model} | {mb:.1f} | `{h}` |")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
