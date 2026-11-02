"""Exploratory: per-SUBTYPE -> predicted-CLASS confusion for the seg models.

Segmentation predicts 6 classes (not subtype); this maps each ground-truth defect
pixel to its subtype (via the inst-id map + train_boxes) and tabulates the model's
predicted class. Rows = 8 subtypes (GT), cols = 6 predicted classes.

NOT for the paper -- writes to analysis_wacv_realbg_visible/ only.

  python training/make_subtype_confusion.py
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


def confusion_for(ckpt_rel: str, device) -> np.ndarray:
    ckpt = torch.load(ROOT / ckpt_rel, map_location="cpu")
    a = dict(ckpt.get("args", {}))
    image_size = int(a.get("image_size", 512))
    ds = PanelSegDataset(a["data_root"], "test", image_size=image_size,
                         split_mode=str(a.get("split_mode", "manifest")),
                         defective_only=False, mask_key=str(a.get("mask_key", "train_mask")))
    loader = DataLoader(ds, batch_size=int(a.get("batch_size", 8)), shuffle=False, num_workers=0)
    model = build_advanced_segmentation_model(
        arch=str(a.get("arch", "deeplabv3plus")), encoder_name=str(a.get("encoder_name") or "resnet34"),
        num_classes=6, pretrained=False, in_channels=int(a.get("in_channels", 3)), image_size=image_size,
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    mat = np.zeros((len(SUBTYPES), 6), dtype=np.int64)
    base, n = 0, len(ds)
    with torch.no_grad():
        for b in loader:
            pred = model(b["image"].to(device)).argmax(1).cpu().numpy()  # B,H,W
            for j in range(pred.shape[0]):
                row = ds.rows[base + j]
                if not row.get("train_boxes_count", 0):
                    continue
                tb = json.load(open(rel_path(ds.root, row["train_boxes"]), encoding="utf-8"))
                id2sub = {int(bx["instance_id"]): bx.get("subtype") for bx in tb.get("boxes", [])}
                if not id2sub:
                    continue
                inst = np.load(rel_path(ds.root, row["inst"])).astype(np.int64)
                inst512 = resize_tensor_mask(torch.from_numpy(inst), image_size).numpy()
                for iid, sub in id2sub.items():
                    si = SUB_IDX.get(sub)
                    if si is None:
                        continue
                    m = inst512 == iid
                    if not m.any():
                        continue
                    mat[si] += np.bincount(pred[j][m], minlength=6)
            base += pred.shape[0]
            if base % 160 == 0:
                print(f"    {base}/{n}", flush=True)
    return mat


def plot(cms: dict[str, np.ndarray]):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(cms), figsize=(7.2 * len(cms), 6.0), dpi=160)
    if len(cms) == 1:
        axes = [axes]
    for ax, (name, mat) in zip(axes, cms.items()):
        row = mat.sum(1, keepdims=True)
        norm = mat / np.maximum(row, 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6)); ax.set_yticks(range(len(SUBTYPES)))
        ax.set_xticklabels(SEG_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(SUBTYPES, fontsize=9)
        ax.set_xlabel("predicted class", fontsize=10); ax.set_ylabel("ground-truth subtype", fontsize=10)
        ax.set_title(name, fontsize=12)
        for r in range(len(SUBTYPES)):
            for c in range(6):
                v = norm[r, c]
                if v >= 0.005:
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.5 else "black")
    fig.suptitle("Subtype -> predicted-class CROSS-TABULATION (row-normalized, test)  "
                 "-- not a square confusion matrix: the model predicts 6 classes, not 8 subtypes",
                 fontsize=11, y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalized")
    fig.savefig(OUTDIR / "seg_subtype_crosstab.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUTDIR / "seg_subtype_crosstab.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    cms = {}
    for name, ckpt in MODELS.items():
        print(f"[{name}]", flush=True)
        cms[name] = confusion_for(ckpt, device)
    (OUTDIR / "seg_subtype_crosstab.json").write_text(
        json.dumps({k: v.tolist() for k, v in cms.items()}, indent=2), encoding="utf-8")
    plot(cms)


if __name__ == "__main__":
    main()
