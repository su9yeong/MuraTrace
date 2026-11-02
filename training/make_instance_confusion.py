"""Exploratory: INSTANCE-level confusion for the seg models.

For each ground-truth defect instance, assign ONE predicted label = the majority
predicted class over the instance's pixels (inst-id map). Then count instances
(each instance = one vote), unlike the pixel-weighted version.

Produces, for both models:
  - class-level instance confusion   (5 GT classes x 6 predicted classes)
  - subtype-level instance cross-tab  (8 subtypes x 6 predicted classes)

NOT for the paper -- writes to analysis_wacv_realbg_visible/ only.

  python training/make_instance_confusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from mura_rca_data import PanelSegDataset, resize_tensor_mask, rel_path  # noqa: E402
from seg_advanced_models import build_advanced_segmentation_model  # noqa: E402
from train_segmentation import SEG_NAMES  # noqa: E402

OUTDIR = ROOT / "analysis_wacv_realbg_visible"
MODELS = {
    "UPerNet-Swin-S": "runs_wacv_realbg_visible/seg_adv_swin_small/best.pt",
    "UPerNet-ConvNeXt-S": "runs_wacv_realbg_visible/seg_adv_upernet_convnext_small/best.pt",
}
SUBTYPES = ["particle_spot", "random_streak", "stitch_line", "puddle_blob",
            "thermal_blob", "anneal_radial_gradient", "row_banding", "column_banding"]
SUB_IDX = {s: i for i, s in enumerate(SUBTYPES)}
DEFECT_CLASSES = SEG_NAMES[1:]  # spot..band (GT instances are defects)


def confusion_for(ckpt_rel, device):
    ckpt = torch.load(ROOT / ckpt_rel, map_location="cpu")
    a = dict(ckpt.get("args", {}))
    isz = int(a.get("image_size", 512))
    ds = PanelSegDataset(a["data_root"], "test", image_size=isz,
                         split_mode=str(a.get("split_mode", "manifest")),
                         defective_only=False, mask_key=str(a.get("mask_key", "train_mask")))
    loader = DataLoader(ds, batch_size=int(a.get("batch_size", 8)), shuffle=False, num_workers=0)
    model = build_advanced_segmentation_model(
        arch=str(a.get("arch", "deeplabv3plus")), encoder_name=str(a.get("encoder_name") or "resnet34"),
        num_classes=6, pretrained=False, in_channels=int(a.get("in_channels", 3)), image_size=isz).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    cls_mat = np.zeros((5, 6), dtype=np.int64)     # GT class (1..5) -> pred class (0..5)
    sub_mat = np.zeros((8, 6), dtype=np.int64)     # subtype -> pred class
    base, n = 0, len(ds)
    with torch.no_grad():
        for b in loader:
            pred = model(b["image"].to(device)).argmax(1).cpu().numpy()
            for j in range(pred.shape[0]):
                row = ds.rows[base + j]
                if not row.get("train_boxes_count", 0):
                    continue
                tb = json.load(open(rel_path(ds.root, row["train_boxes"]), encoding="utf-8"))
                boxes = tb.get("boxes", [])
                if not boxes:
                    continue
                inst = resize_tensor_mask(torch.from_numpy(np.load(rel_path(ds.root, row["inst"])).astype(np.int64)), isz).numpy()
                for bx in boxes:
                    iid = int(bx["instance_id"]); cid = int(bx.get("class_id", 0)); sub = bx.get("subtype")
                    m = inst == iid
                    if not m.any():
                        continue
                    maj = int(np.bincount(pred[j][m], minlength=6).argmax())  # majority predicted class
                    if 1 <= cid <= 5:
                        cls_mat[cid - 1, maj] += 1
                    if sub in SUB_IDX:
                        sub_mat[SUB_IDX[sub], maj] += 1
            base += pred.shape[0]
            if base % 160 == 0:
                print(f"    {base}/{n}", flush=True)
    return cls_mat, sub_mat


def heat(ax, mat, ylabels, title):
    norm = mat / np.maximum(mat.sum(1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(SEG_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("predicted class (instance majority)", fontsize=10)
    ax.set_title(title, fontsize=12)
    for r in range(len(ylabels)):
        for c in range(6):
            v = norm[r, c]
            if v >= 0.005:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.5 else "black")
    return im


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    res = {}
    for name, ckpt in MODELS.items():
        print(f"[{name}]", flush=True)
        res[name] = confusion_for(ckpt, device)

    # class-level instance confusion
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
    for ax, (name, (cls, _)) in zip(axes, res.items()):
        im = heat(ax, cls, DEFECT_CLASSES, name); ax.set_ylabel("ground-truth class", fontsize=10)
    fig.suptitle("INSTANCE-level class confusion (each instance = majority vote, row-normalized, test)", fontsize=11, y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="fraction of instances")
    fig.savefig(OUTDIR / "seg_instance_class_confusion.png", bbox_inches="tight"); plt.close(fig)

    # subtype-level instance cross-tab
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.0), dpi=160)
    for ax, (name, (_, sub)) in zip(axes, res.items()):
        im = heat(ax, sub, SUBTYPES, name); ax.set_ylabel("ground-truth subtype", fontsize=10)
    fig.suptitle("INSTANCE-level subtype CROSS-TABULATION (each instance = majority vote; rows=subtype, cols=class)", fontsize=11, y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="fraction of instances")
    fig.savefig(OUTDIR / "seg_instance_subtype_crosstab.png", bbox_inches="tight"); plt.close(fig)

    (OUTDIR / "seg_instance_confusion.json").write_text(json.dumps(
        {n: {"class": c.tolist(), "subtype": s.tolist()} for n, (c, s) in res.items()}, indent=2), encoding="utf-8")
    print("wrote seg_instance_class_confusion.png + seg_instance_subtype_crosstab.png")


if __name__ == "__main__":
    main()
