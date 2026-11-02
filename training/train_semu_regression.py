# -*- coding: utf-8 -*-
"""Dedicated SEMU regression: defect crop (+position) (+route graph) -> SEMU severity.

Camera-ready extension of the WACV script (same model/optimizer/protocol):
  * variants   : image_only | image_position | image_position_route | image_route
                 (the route graph is encoded with the attribution model's RouteEncoder
                 and mean-pooled, exactly as in the Task-2 fusion baselines)
  * --seeds    : several seeds -> mean +/- std over seeds
  * breakdown  : per visual class / per subtype test MAE, R^2 and reject-F1 (regressed
                 SEMU thresholded at the subtype reject threshold)

The crop cache of the original run (runs_semu_reg/_cache: mat_*.pt, records_*.pkl)
is reused; route tensors and labels are materialized once next to it.

  virtualenv\\Scripts\\python.exe training\\train_semu_regression.py --seeds 7,17,27 \
      --variants image_only,image_position,image_position_route --out runs_semu_reg_cr
"""
import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mura_rca_data import MASK_TYPES, SUBTYPES, DefectRCADataset, build_defect_records, load_route
from mura_rca_models import RouteEncoder
from rca_image_encoders import build_rca_image_encoder

VARIANTS = {
    "image_only": (False, False),
    "image_position": (True, False),
    "image_position_route": (True, True),
    "image_route": (False, True),
}


def get_records(root, split, cache_dir):
    cf = Path(cache_dir) / f"records_{split}.pkl"
    if cf.exists():
        return pickle.load(open(cf, "rb"))
    recs = build_defect_records(Path(root), split=split)
    recs.sort(key=lambda r: (r.key[0], r.key[1]))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(recs, open(cf, "wb"))
    return recs


def materialize(root, split, cache_dir):
    """One pass: crop+resize each record into memory tensors, cached to .pt."""
    cf = Path(cache_dir) / f"mat_{split}.pt"
    if cf.exists():
        d = torch.load(cf)
        print(f"  [{split}] loaded {d['image'].shape[0]} materialized crops", flush=True)
        return d["image"], d["position"], d["semu"]
    ds = DefectRCADataset.__new__(DefectRCADataset)
    ds.root = Path(root); ds.image_size = 224; ds.route_shuffle = False; ds.n_time_bins = 10; ds.route_swap = False
    ds.records = get_records(root, split, cache_dir)
    imgs, poss, sems = [], [], []
    t = time.time()
    for i in range(len(ds)):
        it = ds[i]
        imgs.append(it["image"]); poss.append(it["position"]); sems.append(it["semu"])
        if (i + 1) % 1000 == 0:
            print(f"    [{split}] {i+1}/{len(ds)}", flush=True)
    image = torch.stack(imgs); position = torch.stack(poss); semu = torch.stack(sems)
    torch.save({"image": image, "position": position, "semu": semu}, cf)
    print(f"  [{split}] materialized {len(ds)} crops in {time.time()-t:.0f}s", flush=True)
    return image, position, semu


def materialize_meta(root, split, cache_dir):
    """Route tensors [N,21,4] (tool, step, cycle, pos) + labels, in the cached record order."""
    cf = Path(cache_dir) / f"meta_{split}.pt"
    if cf.exists():
        return torch.load(cf)
    recs = get_records(root, split, cache_dir)
    routes, cls, sub, rej, thr, semu, keys = [], [], [], [], [], [], []
    for r in recs:
        rt = load_route(Path(root), r.row)
        routes.append(torch.stack([rt["route_tool"], rt["route_step"], rt["route_cycle"], rt["route_pos"]], dim=1))
        cls.append(MASK_TYPES.index(str(r.box["mask_type"]))); sub.append(SUBTYPES.index(str(r.box["subtype"])))
        rej.append(float(bool(r.box["is_reject"])))
        thr.append(float(r.box.get("reject_visible_threshold", r.defect.get("reject_visible_threshold", float("nan")))))
        semu.append(float(r.box["semu"])); keys.append((r.key[0], int(r.key[1])))
    meta = {"route": torch.stack(routes), "mask_type": torch.tensor(cls), "subtype": torch.tensor(sub),
            "is_reject": torch.tensor(rej), "reject_thr": torch.tensor(thr), "semu": torch.tensor(semu), "keys": keys}
    torch.save(meta, cf)
    print(f"  [{split}] materialized route/labels for {len(recs)} records", flush=True)
    return meta


class SemuRegressor(nn.Module):
    def __init__(self, kind, embed_dim, pretrained, use_position, use_route, image_size=224):
        super().__init__()
        self.enc = build_rca_image_encoder(kind, embed_dim, pretrained=pretrained, image_size=image_size)
        self.use_position, self.use_route = use_position, use_route
        self.route_enc = RouteEncoder(embed_dim) if use_route else None
        in_dim = embed_dim + (7 if use_position else 0) + (embed_dim if use_route else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, 1))

    def forward(self, image, position, route):
        feats = [self.enc(image)]
        if self.use_position:
            feats.append(position)
        if self.use_route:
            nodes = self.route_enc(route[..., 0], route[..., 1], route[..., 2], route[..., 3])
            feats.append(nodes.mean(dim=1))
        return self.head(torch.cat(feats, dim=1)).squeeze(1)


def r2_score(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / sst if sst > 1e-12 else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval(); ys, ps = [], []
    for img, pos, rt, semu in loader:
        pred = model(img.to(device), pos.to(device), rt.to(device)).float().cpu().numpy() * std + mean
        ps.extend(pred.tolist()); ys.extend(semu.numpy().tolist())
    mae = float(np.abs(np.array(ys) - np.array(ps)).mean())
    return r2_score(ys, ps), mae, np.array(ys), np.array(ps)


def train_variant(tag, seed, dl_tr, dl_va, dl_te, mean, std, args, device):
    use_pos, use_route = VARIANTS[tag]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    model = SemuRegressor(args.encoder, 256, True, use_pos, use_route).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # the submitted supplement lists a constant LR for Task 3; the released script used cosine
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs) if args.schedule == "cosine"
             else torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0))
    best_va, best_state = -1e9, None
    accum = max(1, int(args.accum_steps))
    for ep in range(1, args.epochs + 1):
        model.train()
        for img, pos, rt, semu in dl_tr:
            # fp32 throughout (bf16 autocast was found to cost ~0.2 R^2 on this regression);
            # the nominal batch is processed as `accum` micro-batches to fit the 8 GB GPU
            tgt = (semu.to(device) - mean) / std
            opt.zero_grad()
            for i_mb in range(accum):
                sl = slice(i_mb * len(img) // accum, (i_mb + 1) * len(img) // accum)
                if sl.stop <= sl.start:
                    continue
                out = model(img[sl].to(device), pos[sl].to(device), rt[sl].to(device))
                loss = F.smooth_l1_loss(out, tgt[sl]) * (sl.stop - sl.start) / len(img)
                loss.backward()
            opt.step()
        sched.step()
        va_r2, va_mae, *_ = evaluate(model, dl_va, device, mean, std)
        if ep % 5 == 0 or ep == args.epochs:
            print(f"  [{tag} s{seed}] ep{ep:02d} val R2={va_r2:.3f} MAE={va_mae:.3f}", flush=True)
        if va_r2 > best_va:
            best_va = va_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    te_r2, te_mae, ys, ps = evaluate(model, dl_te, device, mean, std)
    print(f"=== [{tag} s{seed}] TEST R2={te_r2:.4f} MAE={te_mae:.4f} (best val R2={best_va:.3f}) ===", flush=True)
    return {"tag": tag, "seed": seed, "use_position": use_pos, "use_route": use_route,
            "test_r2": te_r2, "test_mae": te_mae, "best_val_r2": best_va, "ys": ys, "ps": ps}


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")


def breakdown(results, meta_te, out_dir):
    """Per class / subtype metrics, averaged over seeds for each variant."""
    cls = meta_te["mask_type"].numpy(); sub = meta_te["subtype"].numpy()
    rej = meta_te["is_reject"].numpy() > 0.5; thr = meta_te["reject_thr"].numpy()
    groups = [("all", "all", np.ones(len(cls), bool))] + \
             [("class", c, cls == i) for i, c in enumerate(MASK_TYPES)] + \
             [("subtype", s, sub == i) for i, s in enumerate(SUBTYPES)]
    rows = []
    tags = sorted({r["tag"] for r in results}, key=lambda t: list(VARIANTS).index(t))
    for tag in tags:
        rs = [r for r in results if r["tag"] == tag]
        for kind, g, m in groups:
            if m.sum() == 0:
                continue
            per_seed = {"mae": [], "r2": [], "reject_f1": []}
            for r in rs:
                y, p = r["ys"][m], r["ps"][m]
                per_seed["mae"].append(float(np.abs(y - p).mean())); per_seed["r2"].append(r2_score(y, p))
                pr, tr = p >= thr[m], rej[m]
                per_seed["reject_f1"].append(f1(int((pr & tr).sum()), int((pr & ~tr).sum()), int((~pr & tr).sum())))
            row = {"variant": tag, "kind": kind, "group": g, "n": int(m.sum()), "n_seeds": len(rs),
                   "semu_mean": float(r["ys"][m].mean()), "semu_std": float(r["ys"][m].std())}
            for k, v in per_seed.items():
                row[f"{k}_mean"] = float(np.nanmean(v)); row[f"{k}_std"] = float(np.nanstd(v))
            rows.append(row)
    cols = list(rows[0].keys())
    with open(out_dir / "semu_breakdown.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    md = ["# SEMU regression breakdown (test; mean +/- std over seeds)\n"]
    for tag in tags:
        md += [f"\n## {tag}\n", "| group | n | SEMU mean | MAE | R^2 | reject F1 |", "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["variant"] == tag:
                md.append(f"| {r['group']} | {r['n']} | {r['semu_mean']:.2f} | {r['mae_mean']:.3f} ± {r['mae_std']:.3f} | "
                          f"{r['r2_mean']:.3f} ± {r['r2_std']:.3f} | {r['reject_f1_mean']:.3f} ± {r['reject_f1_std']:.3f} |")
    (out_dir / "semu_breakdown.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mura_rca_wacv_realbg_visible")
    ap.add_argument("--encoder", default="smallcnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--accum_steps", type=int, default=2, help="micro-batches per optimizer step (memory only; batch_size is the nominal batch)")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "const"], help="LR schedule (submitted supplement: const, 50 epochs)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cache_dir", default="runs_semu_reg/_cache")
    ap.add_argument("--out", default="runs_semu_reg_cr")
    ap.add_argument("--seeds", default="7", help="comma-separated seeds")
    ap.add_argument("--variants", default="image_only,image_position,image_position_route")
    ap.add_argument("--fig_dir", default="tex_camera_ready/figures")
    ap.add_argument("--no_tf32", action="store_true",
                    help="disable TF32 convolutions/matmuls (the Mura contrast of ~1%% of the mean is close to TF32's 10-bit mantissa)")
    ap.add_argument("--no_cudnn", action="store_true", help="diagnostic: torch.backends.cudnn.enabled=False (native conv kernels)")
    ap.add_argument("--cudnn_benchmark", action="store_true", help="diagnostic: deterministic=False, benchmark=True")
    ap.add_argument("--device", default=None, help="diagnostic: force 'cpu' or 'cuda'")
    args = ap.parse_args()
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        print("TF32 disabled", flush=True)
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled", flush=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        assert v in VARIANTS, v

    torch.backends.cudnn.deterministic = not args.cudnn_benchmark
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} encoder={args.encoder} seeds={seeds} variants={variants} | materializing ...", flush=True)
    Xtr, Ptr, Str = materialize(args.data_root, "train", args.cache_dir)
    Xva, Pva, Sva = materialize(args.data_root, "val", args.cache_dir)
    Xte, Pte, Ste = materialize(args.data_root, "test", args.cache_dir)
    Mtr, Mva, Mte = (materialize_meta(args.data_root, s, args.cache_dir) for s in ("train", "val", "test"))
    for S, M, nm in ((Str, Mtr, "train"), (Sva, Mva, "val"), (Ste, Mte, "test")):
        assert torch.allclose(S, M["semu"], atol=1e-4), f"record order mismatch in {nm}"
    mean, std = float(Str.mean()), float(Str.std() + 1e-6)
    print(f"sizes: train={len(Str)} val={len(Sva)} test={len(Ste)} | SEMU mean={mean:.3f} std={std:.3f}", flush=True)

    dl_tr = DataLoader(TensorDataset(Xtr, Ptr, Mtr["route"], Str), batch_size=args.batch_size, shuffle=True, drop_last=True)
    # evaluation batches of 64 (not 256): the fp32 activations of the first conv block at
    # 256x224x224 alone exceed what is left of an 8 GB GPU shared with the desktop
    dl_va = DataLoader(TensorDataset(Xva, Pva, Mva["route"], Sva), batch_size=64, shuffle=False)
    dl_te = DataLoader(TensorDataset(Xte, Pte, Mte["route"], Ste), batch_size=64, shuffle=False)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for tag in variants:
        for seed in seeds:
            results.append(train_variant(tag, seed, dl_tr, dl_va, dl_te, mean, std, args, device))
            np.savez(out_dir / f"preds_{tag}_s{seed}.npz", true=results[-1]["ys"], pred=results[-1]["ps"])

    summary = {"data_root": args.data_root, "encoder": args.encoder, "epochs": args.epochs, "seeds": seeds,
               "mean": mean, "std": std, "per_run": [{k: v for k, v in r.items() if k not in ("ys", "ps")} for r in results],
               "aggregate": {}}
    for tag in variants:
        rs = [r for r in results if r["tag"] == tag]
        summary["aggregate"][tag] = {
            "test_r2_mean": float(np.mean([r["test_r2"] for r in rs])), "test_r2_std": float(np.std([r["test_r2"] for r in rs])),
            "test_mae_mean": float(np.mean([r["test_mae"] for r in rs])), "test_mae_std": float(np.std([r["test_mae"] for r in rs])),
            "n_seeds": len(rs)}
    json.dump(summary, open(out_dir / "semu_regression_result.json", "w"), indent=2)
    breakdown(results, Mte, out_dir)

    # ---- figure: pred-vs-true scatter per variant (first seed) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first = [r for r in results if r["seed"] == seeds[0]]
    label = {"image_only": "image-only", "image_position": "image+position",
             "image_position_route": "image+position+route", "image_route": "image+route"}
    fig, axes = plt.subplots(1, len(first), figsize=(3.4 * len(first), 3.0), dpi=160)
    axes = np.atleast_1d(axes)
    lim = max(6.2, max(float(r["ys"].max()) for r in first))
    for ax, r in zip(axes, first):
        agg = summary["aggregate"][r["tag"]]
        ax.scatter(r["ys"], r["ps"], s=10, color="#3b6fa0", alpha=0.25, edgecolors="none")
        ax.plot([0, lim], [0, lim], "--", color="#888", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("true SEMU"); ax.set_ylabel("predicted SEMU")
        ttl = f"{label[r['tag']]}: $R^2$={agg['test_r2_mean']:.2f}"
        if agg["n_seeds"] > 1:
            ttl += f"±{agg['test_r2_std']:.2f}"
        ttl += f", MAE={agg['test_mae_mean']:.2f}"
        ax.set_title(ttl, fontsize=9)
    fig.tight_layout(pad=0.4)
    for dst in [out_dir / "fig_semu_regression.png", Path(args.fig_dir) / "fig_semu_regression_cr.png"]:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(dst, bbox_inches="tight", pad_inches=0.02)
            print(f"wrote {dst}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"  (skip {dst}: {e})", flush=True)
    plt.close(fig)

    print("\n=== SUMMARY (mean ± std over seeds) ===", flush=True)
    for tag, a in summary["aggregate"].items():
        print(f"  {tag:22s} R2={a['test_r2_mean']:.4f}±{a['test_r2_std']:.4f} MAE={a['test_mae_mean']:.4f}±{a['test_mae_std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
