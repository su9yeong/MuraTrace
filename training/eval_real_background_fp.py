"""Real-texture specificity check: false-positive Mura rate on defect-free panels.

The defect-free panels are real backgrounds assembled into the panel grid with NO
synthetic defect applied. Running a trained segmentation model on them measures
whether it hallucinates Mura on genuine clean OLED texture -- a real-data-side
sanity check that needs no real defective panels.

  python training/eval_real_background_fp.py \
      --checkpoint runs_wacv_realbg_visible/seg_adv_upernet_convnext_small/best.pt \
      --data_root data/mura_rca_wacv_realbg_visible --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mura_rca_data import assigned_split, load_manifest, normalize_image, rel_path, resize_tensor_image
from seg_advanced_models import build_model_from_ckpt_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=500,
                    help="number of clean panels to evaluate (500 is plenty; 0 = all ~3.8k)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--defect_pixel_frac_alarm", type=float, default=0.001,
                    help="a clean panel counts as a false alarm if >this fraction is predicted defect")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = dict(ckpt.get("args", {}))
    image_size = int(ckpt_args.get("image_size", 512))
    split_mode = str(ckpt_args.get("split_mode", "manifest"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} (cuda available={torch.cuda.is_available()})", flush=True)
    if device.type == "cpu":
        print("WARNING: running on CPU -> slow. Use a smaller --limit, or fix the CUDA torch build.", flush=True)

    model = build_model_from_ckpt_args(ckpt_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(args.data_root)
    rows = [r for r in load_manifest(root)
            if assigned_split(r, split_mode) == args.split
            and not bool(r.get("panel_has_train_defect", r.get("panel_has_defect", False)))]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no defect-free panels found for this split")
    print(f"evaluating {len(rows)} clean panels (batch={args.batch_size})", flush=True)

    total_pix = 0
    defect_pix = 0
    alarms = 0
    bs = max(1, args.batch_size)
    with torch.no_grad():
        for start in range(0, len(rows), bs):
            chunk = rows[start:start + bs]
            xs = [resize_tensor_image(torch.from_numpy(normalize_image(np.load(rel_path(root, r["image"]))))[None], image_size)
                  for r in chunk]
            x = torch.stack(xs).to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()  # [B,H,W]
            for pred in preds:
                n = pred.size
                d = int((pred > 0).sum())
                total_pix += n
                defect_pix += d
                if d / n > args.defect_pixel_frac_alarm:
                    alarms += 1
            print(f"  {min(start + bs, len(rows))}/{len(rows)} panels", end="\r", flush=True)
    print()

    fp_pixel_rate = 100.0 * defect_pix / max(total_pix, 1)
    alarm_rate = 100.0 * alarms / len(rows)
    print(f"clean panels evaluated: {len(rows)}")
    print(f"false-positive defect pixel rate: {fp_pixel_rate:.4f}%")
    print(f"panels with >{args.defect_pixel_frac_alarm*100:.2f}% defect pixels (false alarms): "
          f"{alarm_rate:.2f}%  ({alarms}/{len(rows)})")
    print(f"\n-> paper: false-positive defect rate of {fp_pixel_rate:.3f}% of pixels on real clean OLED texture")


if __name__ == "__main__":
    main()
