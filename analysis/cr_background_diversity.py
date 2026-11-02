# -*- coding: utf-8 -*-
"""Camera-ready analysis (reviewers sDje / Sogs): texture-diversity analysis of
the 30 real defect-free backgrounds.

For every background source (the 30 mean/std-matched clean panels that enter the
compositor) and every augmented variant (30 x 18 = 540) we compute
texture statistics that govern Mura masking:
  * global coefficient of variation          cv      = std(L)/mean(L)
  * octave band-pass RMS contrast            c_s     = std(DoG_s(L))/mean(L), s = 1..64 px
    (difference-of-Gaussians between sigma=s and 2s; s spans the spot/line scale
    up to the region/gradient/band scale on the 727x359 panel)
  * low-frequency residual non-uniformity    lf      = std(G_64(L))/mean(L)
  * radially-averaged power-spectrum slope   beta    (log P ~ -beta log f)
and then
  * the spread of each statistic across the 30 sources and across splits,
  * nearest-neighbour distances in standardized feature space: test->train vs
    train->train (are the held-out test textures outliers?),
  * per-feature range coverage of val/test sources by the train sources,
  * variance decomposition over the 540 augmented variants: eta^2 = fraction of
    feature variance explained by source identity (between-source) vs. the
    imaging-environment augmentation (within-source).

Outputs -> analysis_camera_ready/
    bg_features_sources.csv, bg_features_aug.csv, bg_diversity_summary.json,
    bg_diversity_summary.md, fig_bg_diversity.png (+ copy to tex_camera_ready/figures)

Run with a Python that has numpy/scipy/matplotlib (system python 3.14 here):
    python scripts\\cr_background_diversity.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mura_rca_wacv_realbg_visible_v2"
OUT = ROOT / "analysis_camera_ready"
OUT.mkdir(exist_ok=True)
FIG_COPY = ROOT / "tex_camera_ready" / "figures" / "fig_bg_diversity.png"

SPLITS = {
    "train": ["bg00", "bg02", "bg03", "bg06", "bg08", "bg10", "bg11", "bg13", "bg16", "bg17",
              "bg19", "bg20", "bg21", "bg22", "bg26", "bg27", "bg28", "bg29"],
    "val": ["bg01", "bg04", "bg05", "bg09", "bg12", "bg24"],
    "test": ["bg07", "bg14", "bg15", "bg18", "bg23", "bg25"],
}
SPLIT_OF = {b: s for s, bs in SPLITS.items() for b in bs}
SCALES = [1, 2, 4, 8, 16, 32, 64]
FEATS = ["cv"] + [f"c{s}" for s in SCALES] + ["lf64", "psd_slope"]


def load_lum(path: Path) -> np.ndarray:
    im = Image.open(path)
    a = np.asarray(im).astype(np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    hi = a.max()
    if hi > 255.0:
        a = a / (4095.0 if hi <= 4095.0 else 65535.0)
    elif hi > 1.5:
        a = a / 255.0
    return a


def psd_slope(L: np.ndarray) -> float:
    x = L - L.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(x * win))
    P = np.abs(F) ** 2
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(P.shape)
    # cycles per pixel, using the shorter side for the radial bin width
    fr = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    nb = 60
    edges = np.logspace(np.log10(1.0 / 128.0), np.log10(0.25), nb + 1)
    prof, fc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fr >= lo) & (fr < hi)
        if m.sum() >= 8:
            prof.append(P[m].mean())
            fc.append(np.sqrt(lo * hi))
    prof, fc = np.log10(np.array(prof)), np.log10(np.array(fc))
    slope = np.polyfit(fc, prof, 1)[0]
    return float(-slope)


def features(L: np.ndarray) -> dict:
    mu = float(L.mean())
    f = {"mean": mu, "std": float(L.std()), "cv": float(L.std() / mu)}
    for s in SCALES:
        band = ndimage.gaussian_filter(L, s, mode="reflect") - ndimage.gaussian_filter(L, 2 * s, mode="reflect")
        f[f"c{s}"] = float(band.std() / mu)
    f["lf64"] = float(ndimage.gaussian_filter(L, 64, mode="reflect").std() / mu)
    f["psd_slope"] = psd_slope(L)
    return f


def main() -> None:
    # ---- sources (clean, matched) ----
    src_rows = []
    for p in sorted((DATA / "backgrounds_clean").glob("bg*_clean.png")):
        sid = p.stem.split("_")[0]
        L = load_lum(p)
        f = features(L)
        f.update({"source_id": sid, "split": SPLIT_OF.get(sid, "?"), "file": p.name, "h": L.shape[0], "w": L.shape[1]})
        src_rows.append(f)
        print(f"  {sid} {SPLIT_OF.get(sid)} {L.shape} mean={f['mean']:.3f} cv={f['cv']:.4f} c1={f['c1']:.4f} c16={f['c16']:.4f} beta={f['psd_slope']:.2f}", flush=True)
    assert len(src_rows) == 30, len(src_rows)
    # ---- augmented variants ----
    aug_rows = []
    for p in sorted((DATA / "backgrounds_aug").glob("bg*_aug*.png")):
        m = re.match(r"(bg\d+)_aug(\d+)", p.stem)
        sid = m.group(1)
        L = load_lum(p)
        f = features(L)
        f.update({"source_id": sid, "aug_id": p.stem, "split": SPLIT_OF.get(sid, "?"), "file": p.name})
        aug_rows.append(f)
    print(f"  {len(aug_rows)} augmented variants", flush=True)

    cols_src = ["source_id", "split", "file", "h", "w", "mean", "std"] + FEATS
    with open(OUT / "bg_features_sources.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols_src); w.writeheader()
        for r in src_rows: w.writerow({k: r[k] for k in cols_src})
    cols_aug = ["aug_id", "source_id", "split", "file", "mean", "std"] + FEATS
    with open(OUT / "bg_features_aug.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols_aug); w.writeheader()
        for r in aug_rows: w.writerow({k: r[k] for k in cols_aug})

    # ---- spread / coverage / NN analysis on sources ----
    X = np.array([[r[k] for k in FEATS] for r in src_rows])
    split = np.array([r["split"] for r in src_rows])
    ids = [r["source_id"] for r in src_rows]
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    tr, va, te = split == "train", split == "val", split == "test"
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn_train_train = D[tr][:, tr].min(1)
    nn_test_train = D[te][:, tr].min(1)
    nn_val_train = D[va][:, tr].min(1)
    coverage = {}
    for j, k in enumerate(FEATS):
        lo, hi = X[tr, j].min(), X[tr, j].max()
        inside_te = np.mean((X[te, j] >= lo) & (X[te, j] <= hi))
        inside_va = np.mean((X[va, j] >= lo) & (X[va, j] <= hi))
        # relative position of the test values within the train range (0..1 inside)
        coverage[k] = {"train_min": float(lo), "train_max": float(hi), "train_median": float(np.median(X[tr, j])),
                       "test_min": float(X[te, j].min()), "test_max": float(X[te, j].max()),
                       "val_min": float(X[va, j].min()), "val_max": float(X[va, j].max()),
                       "test_inside_train_range": float(inside_te), "val_inside_train_range": float(inside_va),
                       "all_min": float(X[:, j].min()), "all_max": float(X[:, j].max()),
                       "all_median": float(np.median(X[:, j])), "all_cv": float(X[:, j].std() / (abs(X[:, j].mean()) + 1e-12))}
    # ---- variance decomposition over augmented variants ----
    A = np.array([[r[k] for k in FEATS] for r in aug_rows])
    asrc = np.array([r["source_id"] for r in aug_rows])
    eta2 = {}
    for j, k in enumerate(FEATS):
        v = A[:, j]; gm = v.mean(); ss_tot = ((v - gm) ** 2).sum()
        ss_b = sum(((v[asrc == s] - gm) ** 2).sum() * 0 + len(v[asrc == s]) * (v[asrc == s].mean() - gm) ** 2 for s in set(asrc))
        eta2[k] = float(ss_b / ss_tot) if ss_tot > 0 else float("nan")
    # within-source (augmentation) spread vs between-source spread, in units of the source std
    within = {k: float(np.mean([A[asrc == s, j].std() for s in set(asrc)])) for j, k in enumerate(FEATS)}
    between = {k: float(X[:, j].std()) for j, k in enumerate(FEATS)}

    # ---- PCA (on sources; project aug) ----
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = Z @ Vt[:2].T
    expl = (S ** 2 / (S ** 2).sum())[:2]
    ZA = (A - mu) / sd
    pcsA = ZA @ Vt[:2].T

    summary = {
        "n_sources": 30, "n_aug": len(aug_rows), "features": FEATS,
        "panel_size_hw": [src_rows[0]["h"], src_rows[0]["w"]],
        "per_feature": coverage,
        "nn_distance_standardized": {
            "train_to_train_median": float(np.median(nn_train_train)), "train_to_train_max": float(nn_train_train.max()),
            "test_to_train_median": float(np.median(nn_test_train)), "test_to_train_max": float(nn_test_train.max()),
            "val_to_train_median": float(np.median(nn_val_train)), "val_to_train_max": float(nn_val_train.max()),
            "test_to_train_per_source": {ids[i]: float(d) for i, d in zip(np.where(te)[0], nn_test_train)},
        },
        "eta2_between_source_over_aug": eta2,
        "within_source_aug_std": within, "between_source_std": between,
        "pca_explained": [float(e) for e in expl],
        "pca_loadings_pc1": {k: float(v) for k, v in zip(FEATS, Vt[0])},
        "pca_loadings_pc2": {k: float(v) for k, v in zip(FEATS, Vt[1])},
    }
    (OUT / "bg_diversity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- markdown summary ----
    md = ["# Background texture diversity (30 real sources, 540 augmented variants)\n",
          f"Panel size (h x w): {src_rows[0]['h']} x {src_rows[0]['w']}  |  features: {', '.join(FEATS)}\n",
          "## Per-feature spread over the 30 sources and test/val coverage by the 18 train sources\n",
          "| feature | all median | all min | all max | all CV | train min | train max | test min | test max | test inside train range | val inside |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in FEATS:
        c = coverage[k]
        md.append(f"| {k} | {c['all_median']:.4g} | {c['all_min']:.4g} | {c['all_max']:.4g} | {c['all_cv']:.3f} | {c['train_min']:.4g} | {c['train_max']:.4g} | "
                  f"{c['test_min']:.4g} | {c['test_max']:.4g} | {c['test_inside_train_range']:.2f} | {c['val_inside_train_range']:.2f} |")
    nn = summary["nn_distance_standardized"]
    md += ["\n## Nearest-neighbour distance in standardized feature space\n",
           f"- train->train: median {nn['train_to_train_median']:.2f}, max {nn['train_to_train_max']:.2f}",
           f"- test->train:  median {nn['test_to_train_median']:.2f}, max {nn['test_to_train_max']:.2f}",
           f"- val->train:   median {nn['val_to_train_median']:.2f}, max {nn['val_to_train_max']:.2f}",
           "\n## Variance decomposition over the 540 augmented variants (eta^2 = between-source share)\n",
           "| feature | eta^2 | between-source std | within-source (aug) std |", "|---|---:|---:|---:|"]
    for k in FEATS:
        md.append(f"| {k} | {eta2[k]:.2f} | {between[k]:.4g} | {within[k]:.4g} |")
    md += [f"\nPCA on sources: PC1 {expl[0]*100:.0f}%, PC2 {expl[1]*100:.0f}% of variance."]
    (OUT / "bg_diversity_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#C44E52"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), dpi=200)
    ax = axes[0]
    for r in src_rows:
        ax.plot(SCALES, [r[f"c{s}"] * 100 for s in SCALES], "-o", ms=2.5, lw=0.9, color=colors[r["split"]], alpha=0.85)
    for s_, c_ in colors.items():
        ax.plot([], [], "-o", ms=2.5, color=c_, label=f"{s_} ({len(SPLITS[s_])})")
    ax.set_xscale("log", base=2); ax.set_xticks(SCALES); ax.set_xticklabels([str(s) for s in SCALES])
    ax.set_xlabel("band-pass scale $s$ [px]"); ax.set_ylabel("RMS contrast [% of mean]")
    ax.set_title("(a) octave contrast spectra, 30 sources", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    ax.scatter(pcsA[:, 0], pcsA[:, 1], s=4, c=[colors[r["split"]] for r in aug_rows], alpha=0.18, edgecolors="none")
    ax.scatter(pcs[:, 0], pcs[:, 1], s=26, c=[colors[r["split"]] for r in src_rows], edgecolors="k", linewidths=0.4, zorder=3)
    for i, sid in enumerate(ids):
        if split[i] == "test":
            ax.annotate(sid, (pcs[i, 0], pcs[i, 1]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({expl[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({expl[1]*100:.0f}%)")
    ax.set_title("(b) texture-feature PCA: sources (solid), aug (faint)", fontsize=9); ax.grid(alpha=0.25, lw=0.5)
    ax = axes[2]
    data = [nn_train_train, nn_val_train, nn_test_train]
    ax.boxplot(data, tick_labels=["train→train", "val→train", "test→train"], widths=0.5,
               medianprops={"color": "k"})
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)), d, s=10,
                   color=list(colors.values())[i - 1], zorder=3)
    ax.set_ylabel("NN distance (standardized features)")
    ax.set_title("(c) held-out sources are not texture outliers", fontsize=9); ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig_bg_diversity.png", bbox_inches="tight", pad_inches=0.02)
    FIG_COPY.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_COPY, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT / 'fig_bg_diversity.png'} and {FIG_COPY}")


if __name__ == "__main__":
    main()
