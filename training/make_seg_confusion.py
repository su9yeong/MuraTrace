"""Per-class segmentation confusion matrices (pixel-level, row-normalized) for the
two competing global-context models: UPerNet-Swin-S and UPerNet-ConvNeXt-S.

Runs inference on the test split (same loading as evaluate_segmentation_advanced),
accumulates a 6x6 confusion, and writes a side-by-side figure.

  python training/make_seg_confusion.py

-> tex/figures/fig_seg_confusion.png  (+ seg_confusion.json)
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
from mura_rca_data import PanelSegDataset  # noqa: E402
from seg_advanced_models import build_advanced_segmentation_model  # noqa: E402
from train_segmentation import SEG_NAMES, SEG_IGNORE_INDEX  # noqa: E402

OUT = ROOT / "tex" / "figures"
MODELS = {
    "UPerNet-Swin-S": "runs_wacv_realbg_visible/seg_adv_swin_small/best.pt",
    "UPerNet-ConvNeXt-S": "runs_wacv_realbg_visible/seg_adv_upernet_convnext_small/best.pt",
}


def confusion_for(ckpt_rel: str, device) -> np.ndarray:
    ckpt = torch.load(ROOT / ckpt_rel, map_location="cpu")
    a = dict(ckpt.get("args", {}))
    image_size = int(a.get("image_size", 512))
    ds = PanelSegDataset(
        a["data_root"], "test", image_size=image_size,
        split_mode=str(a.get("split_mode", "manifest")), defective_only=False,
        mask_key=str(a.get("mask_key", "train_mask")),
    )
    loader = DataLoader(ds, batch_size=int(a.get("batch_size", 8)), shuffle=False, num_workers=0, pin_memory=True)
    model = build_advanced_segmentation_model(
        arch=str(a.get("arch", "deeplabv3plus")),
        encoder_name=str(a.get("encoder_name") or "resnet34"),
        num_classes=6, pretrained=False, in_channels=int(a.get("in_channels", 3)),
        image_size=image_size,
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    cm = torch.zeros(36, dtype=torch.long)
    n = len(ds)
    with torch.no_grad():
        for i, b in enumerate(loader):
            logits = model(b["image"].to(device))
            pred = logits.argmax(1).cpu().view(-1)
            true = b["mask"].view(-1)
            keep = (true != SEG_IGNORE_INDEX) & (true >= 0) & (true < 6)
            idx = true[keep] * 6 + pred[keep]
            cm += torch.bincount(idx, minlength=36)
            if (i + 1) % 20 == 0:
                print(f"    {min((i + 1) * loader.batch_size, n)}/{n}", flush=True)
    return cm.reshape(6, 6).numpy()


def plot(cms: dict[str, np.ndarray]):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(cms), figsize=(6.6 * len(cms), 5.4), dpi=160)
    if len(cms) == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        row = cm.sum(1, keepdims=True)
        norm = cm / np.maximum(row, 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(SEG_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(SEG_NAMES, fontsize=9)
        ax.set_xlabel("predicted", fontsize=10); ax.set_ylabel("ground truth", fontsize=10)
        ax.set_title(name, fontsize=12)
        for r in range(6):
            for c in range(6):
                v = norm[r, c]
                if v >= 0.005:
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalized")
    fig.savefig(OUT / "fig_seg_confusion.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_seg_confusion.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    cms = {}
    for name, ckpt in MODELS.items():
        print(f"[{name}] {ckpt}")
        cms[name] = confusion_for(ckpt, device)
    (OUT / "seg_confusion.json").write_text(
        json.dumps({k: v.tolist() for k, v in cms.items()}, indent=2), encoding="utf-8")
    plot(cms)


if __name__ == "__main__":
    main()
