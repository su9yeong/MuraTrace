# -*- coding: utf-8 -*-
"""Zero-shot real-data check (camera-ready, reviewers AC/i4b1/Sogs): run MuraTrace
segmentation checkpoints on the public OLED weak-Mura tiles of Qiu et al. 2024
(figshare 24532633, CC BY 4.0) and score box-level recall / false alarms.

The release contains 10,020 grayscale 416x416 tiles cut from industrial-camera
images of lit OLED phone screens under four gray-level test patterns (WHITE,
128W2, 64W1, 32W1); 4,412 tiles carry 5,790 YOLO boxes around POINT Mura
(one defect family ~ MuraTrace `spot`), 5,608 tiles are defect-free. Tiles
include the black bezel / rounded screen corners and a strong illumination
gradient, so a fixed, physically motivated preprocessing is applied before the
unchanged MuraTrace models see them (nothing is trained or tuned on this data):

  1. screen ROI : pixels brighter than 35% of the tile's 99th percentile, eroded
                  by --roi_erode px (drops the bezel edge); scoring is restricted to it
  2. flat-field : divide by a Gaussian field (sigma=--flat_sigma px) estimated over
                  ROI pixels only (normalized convolution); outside the ROI the
                  relative luminance is set to 1 -> no artificial edge
  3. optional denoise (Gaussian --denoise_sigma px) and rescale (--scale) to probe
     the camera-noise / pixel-pitch mismatch with MuraTrace panels
  4. gain to the MuraTrace panel mean (0.5): the perturbation is multiplicative,
     so relative Mura contrast is preserved
  5. the tile is placed on a MuraTrace-sized canvas (727x359) with reflect padding,
     predicted at the model input size (512x512; two overlapping windows when the
     tile is taller than the canvas), votes are merged, and the prediction is
     cropped back to the tile

Scoring (within the ROI): a GT box is RECALLED if >= --min_hit_pixels predicted
defect pixels (any defect class, or spot only with --spot_only) fall inside it;
every connected component of predicted defect pixels (area >= --min_fa_area) that
touches no GT box is a FALSE ALARM. Defect-free tiles give the clean-tile false-alarm
rate (fraction of clean tiles with >= 1 false alarm), the real-data analogue of the
paper's real-background specificity check.

  python training/eval_real_weakmura.py --data_root external/qiu2024_weakmura/extracted/datasets \
      --checkpoint runs_wacv_realbg_visible/seg_adv_swin_small/best.pt \
      --out analysis_camera_ready/sim2real_swin_s1.json --examples_dir analysis_camera_ready/sim2real_examples_swin
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from seg_advanced_models import build_advanced_segmentation_model

PANEL_W, PANEL_H = 727, 359
BG_MEAN = 0.5
CLASSES = ["background", "spot", "line", "region", "gradient", "band"]


# ----------------------------------------------------------------------------- data
def discover(data_root: Path) -> list[tuple[Path, list[list[float]]]]:
    img_dir = data_root / "images" if (data_root / "images").is_dir() else data_root
    lab_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    items = []
    for im in sorted(img_dir.glob("*.bmp")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
        lp = lab_dir / (im.stem + ".txt")
        boxes: list[list[float]] = []
        if lp.exists():
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                q = line.split()
                if len(q) >= 5:
                    try:
                        cx, cy, bw, bh = map(float, q[1:5])
                    except ValueError:
                        continue
                    boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])  # normalized xyxy
        items.append((im, boxes))
    return items


# ----------------------------------------------------------------------------- preprocessing
def gaussian(a: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a [1,1,H,W] float tensor with reflect padding."""
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=a.dtype, device=a.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
    H, W = a.shape[-2:]
    ry, rx = min(r, H - 1), min(r, W - 1)
    a = F.pad(a, (rx, rx, ry, ry), mode="reflect")
    a = F.conv2d(a, k[: 2 * rx + 1].view(1, 1, 1, -1) / k[: 2 * rx + 1].sum()) if rx < r else F.conv2d(a, k.view(1, 1, 1, -1))
    a = F.conv2d(a, k[: 2 * ry + 1].view(1, 1, -1, 1) / k[: 2 * ry + 1].sum()) if ry < r else F.conv2d(a, k.view(1, 1, -1, 1))
    return a


def screen_roi(a: torch.Tensor, erode_px: int, rel_thr: float = 0.35) -> torch.Tensor:
    """Lit-screen mask: brighter than rel_thr * p99, then eroded (min-pool) by erode_px."""
    p99 = torch.quantile(a.flatten(), 0.99)
    m = (a > rel_thr * p99).float()
    if erode_px > 0:
        k = 2 * erode_px + 1
        m = 1.0 - F.max_pool2d(1.0 - m, k, stride=1, padding=erode_px)
    return m


def preprocess(tile_u8: np.ndarray, scale: float, flat_sigma: float, denoise_sigma: float, roi_erode: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.from_numpy(tile_u8.astype(np.float32) / 255.0)[None, None].to(device)
    if abs(scale - 1.0) > 1e-6:
        a = F.interpolate(a, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
    roi = screen_roi(a, int(round(roi_erode * scale)))
    if roi.sum() < 64:  # tile is almost entirely bezel
        return None, roi
    num = gaussian(a * roi, flat_sigma * scale)
    den = gaussian(roi, flat_sigma * scale)
    field = num / den.clamp_min(1e-3)
    rel = torch.where(roi > 0.5, a / field.clamp_min(1e-4), torch.ones_like(a))
    if denoise_sigma > 0:
        rel = gaussian(rel, denoise_sigma)
    x = (rel * BG_MEAN).clamp(0.0, 1.0)
    return x, roi


# ----------------------------------------------------------------------------- inference
def reflect_pad_to(x: torch.Tensor, target_w: int) -> tuple[torch.Tensor, int]:
    """Reflect-pad [1,1,h,w] horizontally to target_w (in several steps if the pad exceeds w)."""
    pl = 0
    while x.shape[-1] < target_w:
        step = min(x.shape[-1] - 1, target_w - x.shape[-1])
        a_, b_ = step // 2, step - step // 2
        x = F.pad(x, (a_, b_, 0, 0), mode="reflect")
        pl += a_
    return x, pl


def make_windows(x: torch.Tensor, image_size: int):
    """x: [1,1,h,w] -> list of (window [1,1,S,S], y, pt, wh) plus the padded canvas geometry."""
    _, _, h, w = x.shape
    canvas, pl = reflect_pad_to(x, PANEL_W)
    H = canvas.shape[-2]
    ys = [0] if H <= PANEL_H else sorted({0, H - PANEL_H, *(list(range(0, H - PANEL_H, PANEL_H // 2)))})
    wins = []
    for y in ys:
        win = canvas[:, :, y:y + PANEL_H, :]
        wh = win.shape[-2]
        pt = 0
        if wh < PANEL_H:
            pt = (PANEL_H - wh) // 2
            win = F.pad(win, (0, 0, pt, PANEL_H - wh - pt), mode="reflect")
        wins.append((F.interpolate(win, size=(image_size, image_size), mode="bilinear", align_corners=False), y, pt, wh))
    return wins, (H, canvas.shape[-1], pl, w)


@torch.no_grad()
def predict_tiles(model, xs: list[torch.Tensor], image_size: int, amp_dtype, device, batch: int = 16) -> list[torch.Tensor]:
    """Batched inference over several preprocessed tiles; returns one [h,w] class map per tile."""
    all_wins, geoms, owner = [], [], []
    for i, x in enumerate(xs):
        wins, geom = make_windows(x, image_size)
        geoms.append(geom)
        for win in wins:
            all_wins.append(win); owner.append(i)
    preds = []
    for s in range(0, len(all_wins), batch):
        inp = torch.cat([w[0] for w in all_wins[s:s + batch]])
        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(inp)
        p = torch.argmax(logits.float(), dim=1, keepdim=True).float()
        p = F.interpolate(p, size=(PANEL_H, PANEL_W), mode="nearest")[:, 0].long()
        preds.extend(list(p))
    votes = [torch.zeros(6, g[0], g[1], device=device) for g in geoms]
    for (win, y, pt, wh), i, p in zip(all_wins, owner, preds):
        votes[i][:, y:y + wh, :] += F.one_hot(p[pt:pt + wh, :], 6).permute(2, 0, 1).float()
    out = []
    for v, (H, Wc, pl, w) in zip(votes, geoms):
        out.append(v.argmax(0)[:, pl:pl + w])
    return out


def components(mask_u8: np.ndarray, min_area: int):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append((int(x), int(y), int(x + w), int(y + h), int(area)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples_dir", default=None)
    ap.add_argument("--n_examples", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.0, help="rescale factor applied to the 416 px tiles before inference")
    ap.add_argument("--flat_sigma", type=float, default=48.0, help="flat-field Gaussian sigma in native px")
    ap.add_argument("--denoise_sigma", type=float, default=0.0, help="Gaussian denoise sigma in (scaled) px; 0 = off")
    ap.add_argument("--roi_erode", type=int, default=24, help="ROI erosion in native px")
    ap.add_argument("--spot_only", action="store_true", help="count only predicted spot pixels")
    ap.add_argument("--min_hit_pixels", type=int, default=1)
    ap.add_argument("--min_fa_area", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="seed for the --limit subsample / example selection")
    ap.add_argument("--device", default=None)
    ap.add_argument("--group", type=int, default=16, help="tiles preprocessed per step")
    ap.add_argument("--batch", type=int, default=16, help="windows per forward pass")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ca = dict(ckpt.get("args", {}))
    image_size = int(ca.get("image_size", 512))
    amp_dtype = torch.bfloat16 if str(ca.get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_advanced_segmentation_model(arch=str(ca.get("arch", "deeplabv3plus")), encoder_name=str(ca.get("encoder_name") or "resnet34"),
                                              num_classes=6, pretrained=False, in_channels=int(ca.get("in_channels", 3)), image_size=image_size).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    items = discover(Path(args.data_root))
    rng = random.Random(args.seed)
    if args.limit and args.limit < len(items):
        items = rng.sample(items, args.limit)
    n_def = sum(1 for _, b in items if b)
    print(f"{len(items)} tiles ({n_def} with boxes); scale={args.scale} flat_sigma={args.flat_sigma} denoise={args.denoise_sigma} "
          f"roi_erode={args.roi_erode} spot_only={args.spot_only}", flush=True)
    ex_dir = Path(args.examples_dir) if args.examples_dir else None
    if ex_dir:
        ex_dir.mkdir(parents=True, exist_ok=True)
    ex_pick = set(rng.sample([i for i, (_, b) in enumerate(items) if b], min(args.n_examples, n_def))) if ex_dir else set()

    per_pattern: dict[str, dict] = {}
    tot = {"boxes": 0, "hit": 0, "fa_def": 0, "fa_clean": 0, "clean_tiles": 0, "clean_tiles_alarm": 0, "def_tiles": 0, "skipped": 0}
    class_pix = np.zeros(6, dtype=np.int64); roi_pix = 0
    per_tile = []
    group = max(1, int(args.group))
    for g0 in range(0, len(items), group):
        chunk = items[g0:g0 + group]
        pre = []
        for im, boxes in chunk:
            tile = np.asarray(Image.open(im).convert("L"))
            x, roi = preprocess(tile, args.scale, args.flat_sigma, args.denoise_sigma, args.roi_erode, device)
            pre.append((x, roi))
        valid = [i for i, (x, _) in enumerate(pre) if x is not None]
        preds_valid = predict_tiles(model, [pre[i][0] for i in valid], image_size, amp_dtype, device, batch=args.batch) if valid else []
        preds = {i: p for i, p in zip(valid, preds_valid)}
        for j, (im, boxes) in enumerate(chunk):
            idx = g0 + j
            x, roi = pre[j]
            pat = im.stem.split("_")[1] if "_" in im.stem else "all"
            pp = per_pattern.setdefault(pat, {"boxes": 0, "hit": 0, "fa_def": 0, "fa_clean": 0, "clean_tiles": 0, "clean_tiles_alarm": 0, "def_tiles": 0})
            if x is None:
                tot["skipped"] += 1
                continue
            pred = preds[j]
            roi_b = roi[0, 0] > 0.5
            pred = torch.where(roi_b, pred, torch.zeros_like(pred))
            pred_np = pred.cpu().numpy().astype(np.uint8)
            roi_np = roi_b.cpu().numpy()
            class_pix += np.bincount(pred_np[roi_np].ravel(), minlength=6); roi_pix += int(roi_np.sum())
            det_any = pred_np > 0
            det_spot = pred_np == 1
            det = det_spot if args.spot_only else det_any
            h, w = det.shape
            comps = components(det.astype(np.uint8), args.min_fa_area)
            pix_boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes]

            def box_hit(mask, bx0, by0, bx1, by1):
                x0, y0 = max(0, int(bx0)), max(0, int(by0)); x1, y1 = min(w, int(np.ceil(bx1))), min(h, int(np.ceil(by1)))
                return bool(x1 > x0 and y1 > y0 and mask[y0:y1, x0:x1].sum() >= args.min_hit_pixels)

            hit_any = sum(box_hit(det_any, *b) for b in pix_boxes)
            hit_spot = sum(box_hit(det_spot, *b) for b in pix_boxes)
            # chance level: one random box of the same size per GT box, uniformly placed inside the
            # ROI and not overlapping any GT box (measures how often an unrelated box is "recalled")
            rand_any = rand_spot = 0
            if pix_boxes:
                ys_roi, xs_roi = np.nonzero(roi_np)
                for (bx0, by0, bx1, by1) in pix_boxes:
                    bw, bh = bx1 - bx0, by1 - by0
                    for _try in range(20):
                        k = rng.randrange(len(ys_roi))
                        cx, cy = xs_roi[k], ys_roi[k]
                        r = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                        if r[0] < 0 or r[1] < 0 or r[2] > w or r[3] > h:
                            continue
                        if any(r[0] < gx1 and r[2] > gx0 and r[1] < gy1 and r[3] > gy0 for (gx0, gy0, gx1, gy1) in pix_boxes):
                            continue
                        rand_any += int(box_hit(det_any, *r)); rand_spot += int(box_hit(det_spot, *r))
                        break
            hit = hit_spot if args.spot_only else hit_any

            def n_false_alarms(cc):
                return sum(1 for (cx0, cy0, cx1, cy1, _) in cc
                           if not any(cx0 < bx1 and cx1 > bx0 and cy0 < by1 and cy1 > by0 for (bx0, by0, bx1, by1) in pix_boxes))
            fa = n_false_alarms(comps)
            fa_spot = n_false_alarms(components(det_spot.astype(np.uint8), args.min_fa_area))  # spot-class-only false alarms
            rec = {"tile": im.stem, "pattern": pat, "n_boxes": len(boxes), "n_hit": hit, "n_hit_any": hit_any, "n_hit_spot": hit_spot,
                   "n_rand_any": rand_any, "n_rand_spot": rand_spot, "n_fa": fa, "n_fa_spot": fa_spot, "roi_frac": float(roi_np.mean())}
            per_tile.append(rec)
            for d in (tot, pp):
                d["boxes"] += len(boxes); d["hit"] += hit
                d["hit_any"] = d.get("hit_any", 0) + hit_any; d["hit_spot"] = d.get("hit_spot", 0) + hit_spot
                d["rand_any"] = d.get("rand_any", 0) + rand_any; d["rand_spot"] = d.get("rand_spot", 0) + rand_spot
                if boxes:
                    d["def_tiles"] += 1; d["fa_def"] += fa; d["fa_spot_def"] = d.get("fa_spot_def", 0) + fa_spot
                else:
                    d["clean_tiles"] += 1; d["fa_clean"] += fa; d["clean_tiles_alarm"] += int(fa > 0)
                    d["fa_spot_clean"] = d.get("fa_spot_clean", 0) + fa_spot; d["clean_tiles_alarm_spot"] = d.get("clean_tiles_alarm_spot", 0) + int(fa_spot > 0)
            if idx in ex_pick:
                base = (x[0, 0].cpu().numpy() / BG_MEAN)
                m, s = base[roi_np].mean(), base[roi_np].std() + 1e-6
                vis = np.clip((base - (m - 3 * s)) / (6 * s), 0, 1)
                vis = np.stack([vis, vis, vis], -1)
                colors = {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0.4, 1), 4: (1, 1, 0), 5: (1, 0, 1)}
                over = vis.copy()
                for c, col in colors.items():
                    over[pred_np == c] = col
                vis = 0.55 * vis + 0.45 * over
                img = Image.fromarray((vis * 255).astype(np.uint8))
                from PIL import ImageDraw
                dr = ImageDraw.Draw(img)
                for (bx0, by0, bx1, by1) in pix_boxes:
                    dr.rectangle([bx0 - 2, by0 - 2, bx1 + 2, by1 + 2], outline=(0, 255, 255), width=2)
                img.save(ex_dir / f"{im.stem}_hit{hit}of{len(boxes)}_fa{fa}.png")
            if idx % 500 == 0:
                print(f"  {idx + 1}/{len(items)} boxes={tot['boxes']} hit={tot['hit']} fa(def)={tot['fa_def']} clean_alarm={tot['clean_tiles_alarm']}/{tot['clean_tiles']}", flush=True)

    def summarize(d):
        nb = max(d["boxes"], 1)
        return {**d, "recall": d["hit"] / nb,
                "recall_any": d.get("hit_any", 0) / nb, "recall_spot": d.get("hit_spot", 0) / nb,
                "chance_any": d.get("rand_any", 0) / nb, "chance_spot": d.get("rand_spot", 0) / nb,
                "fa_per_defective_tile": d["fa_def"] / max(d["def_tiles"], 1),
                "fa_per_clean_tile": d["fa_clean"] / max(d["clean_tiles"], 1),
                "clean_tile_alarm_rate": d["clean_tiles_alarm"] / max(d["clean_tiles"], 1),
                "fa_spot_per_defective_tile": d.get("fa_spot_def", 0) / max(d["def_tiles"], 1),
                "fa_spot_per_clean_tile": d.get("fa_spot_clean", 0) / max(d["clean_tiles"], 1),
                "clean_tile_alarm_rate_spot": d.get("clean_tiles_alarm_spot", 0) / max(d["clean_tiles"], 1)}
    result = {"checkpoint": args.checkpoint, "data_root": args.data_root, "settings": vars(args), "n_tiles": len(items),
              "overall": summarize(tot), "per_pattern": {k: summarize(v) for k, v in per_pattern.items()},
              "pred_class_frac_in_roi": {CLASSES[c]: float(class_pix[c] / max(roi_pix, 1)) for c in range(6)},
              "per_tile": per_tile}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    o = result["overall"]
    print(json.dumps({"recall_any": round(o["recall_any"], 4), "chance_any": round(o["chance_any"], 4),
                      "recall_spot": round(o["recall_spot"], 4), "chance_spot": round(o["chance_spot"], 4),
                      "boxes": o["boxes"], "fa_per_defective_tile": round(o["fa_per_defective_tile"], 3),
                      "fa_per_clean_tile": round(o["fa_per_clean_tile"], 3), "clean_tile_alarm_rate": round(o["clean_tile_alarm_rate"], 4),
                      "pred_class_frac_in_roi": {k: round(v, 5) for k, v in result["pred_class_frac_in_roi"].items()}}, indent=1))
    for k, v in result["per_pattern"].items():
        print(f"  {k}: recall any {v['recall_any']:.3f} (chance {v['chance_any']:.3f}) spot {v['recall_spot']:.3f} (chance {v['chance_spot']:.3f}) "
              f"n={v['boxes']}, fa/def-tile {v['fa_per_defective_tile']:.2f}, clean alarm {v['clean_tile_alarm_rate']:.3f}")


if __name__ == "__main__":
    main()
