# -*- coding: utf-8 -*-
"""Final version of the background-diversity figure: panels (a)/(b) as in
cr_background_diversity.py, panel (c) = per-test-background mIoU of the top
segmentation models (from training/eval_seg_per_source.py outputs), with the
strictly-unseen backgrounds marked.

    python scripts/cr_fig_bg_diversity_final.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANA = ROOT / "analysis_camera_ready"
OUT = [ANA / "fig_bg_diversity.png", ROOT / "tex_camera_ready" / "figures" / "fig_bg_diversity.png"]
SCALES = [1, 2, 4, 8, 16, 32, 64]
FEATS = ["cv"] + [f"c{s}" for s in SCALES] + ["lf64", "psd_slope"]
REPEAT = {"bg07", "bg18", "bg23"}
MODELS = [("swin_small", "UPerNet Swin-S"), ("upernet_convnext_small", "UPerNet ConvNeXt-S"), ("deeplab_convnext_small", "DeepLabV3+ ConvNeXt-S")]


def main() -> None:
    src = list(csv.DictReader(open(ANA / "bg_features_sources.csv", encoding="utf-8")))
    aug = list(csv.DictReader(open(ANA / "bg_features_aug.csv", encoding="utf-8")))
    X = np.array([[float(r[k]) for k in FEATS] for r in src]); A = np.array([[float(r[k]) for k in FEATS] for r in aug])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z, ZA = (X - mu) / sd, (A - mu) / sd
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs, pcsA = Z @ Vt[:2].T, ZA @ Vt[:2].T
    expl = (S ** 2 / (S ** 2).sum())[:2]
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#C44E52"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), dpi=200)
    ax = axes[0]
    for r in src:
        ax.plot(SCALES, [float(r[f"c{s}"]) * 100 for s in SCALES], "-o", ms=2.5, lw=0.9, color=colors[r["split"]], alpha=0.85)
    for s_, c_ in colors.items():
        ax.plot([], [], "-o", ms=2.5, color=c_, label=s_)
    ax.set_xscale("log", base=2); ax.set_xticks(SCALES); ax.set_xticklabels([str(s) for s in SCALES])
    ax.set_xlabel("band-pass scale $s$ [px]"); ax.set_ylabel("RMS contrast [% of mean]")
    ax.set_title("(a) octave contrast spectra, 30 backgrounds", fontsize=9); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    ax.scatter(pcsA[:, 0], pcsA[:, 1], s=4, c=[colors[r["split"]] for r in aug], alpha=0.18, edgecolors="none")
    ax.scatter(pcs[:, 0], pcs[:, 1], s=26, c=[colors[r["split"]] for r in src], edgecolors="k", linewidths=0.4, zorder=3)
    for r, p in zip(src, pcs):
        if r["split"] == "test":
            ax.annotate(r["source_id"], p, fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({expl[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({expl[1]*100:.0f}%)")
    ax.set_title("(b) texture-feature PCA: backgrounds (solid), aug (faint)", fontsize=9); ax.grid(alpha=0.25, lw=0.5)
    ax = axes[2]
    have = [(k, n) for k, n in MODELS if (ANA / f"seg_per_source_{k}.json").exists()]
    if have:
        caps = None
        width = 0.8 / len(have)
        for i, (k, name) in enumerate(have):
            d = json.load(open(ANA / f"seg_per_source_{k}.json", encoding="utf-8"))
            caps = sorted(d["per_source"])
            vals = [d["per_source"][c]["miou_defect_present"] for c in caps]
            xs = np.arange(len(caps)) + (i - (len(have) - 1) / 2) * width
            ax.bar(xs, vals, width=width, label=name, alpha=0.9)
            ax.axhline(d["all"]["miou_defect_present"], color=f"C{i}", lw=0.8, ls="--", alpha=0.8)
        ax.set_xticks(np.arange(len(caps)))
        ax.set_xticklabels(caps, fontsize=8)
        lo = min(json.load(open(ANA / f"seg_per_source_{k}.json"))["per_source"][c]["miou_defect_present"] for k, _ in have for c in caps)
        ax.set_ylim(max(0.0, lo - 0.1), 1.0)
        ax.set_ylabel("test mIoU$_{\\mathrm{def}}$"); ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        ax.set_title("(c) per held-out background", fontsize=9); ax.grid(alpha=0.25, lw=0.5, axis="y")
    else:
        ax.text(0.5, 0.5, "per-background evaluation pending", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout(pad=0.5)
    for o in OUT:
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, bbox_inches="tight", pad_inches=0.02)
        print("wrote", o)


if __name__ == "__main__":
    main()
