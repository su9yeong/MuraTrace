#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export weights-only checkpoints for release (drops optimizer/scheduler state).

  python tools/strip_checkpoints.py --runs runs_wacv_realbg_visible/seg_adv_swin_small runs_cr_ambiguous/rca_visionroute_s7 ... --out release/checkpoints

Each input directory must contain best.pt (a dict with the model state under 'model', 'model_state',
'state_dict' or the whole dict being a state_dict). The output keeps the state dict plus a small
metadata block (source run, epoch, validation score, architecture arguments when present).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def extract(ckpt: dict):
    for key in ("model", "model_state", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key], key
    if all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt, "<root>"
    raise KeyError(f"no state dict found; keys = {list(ckpt)[:10]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for run in args.runs:
        run = Path(run)
        src = run / "best.pt"
        try:
            ckpt = torch.load(src, map_location="cpu", weights_only=False)
            state, key = extract(ckpt)
        except Exception as exc:  # keep going, report at the end
            print(f"{run.name}: SKIPPED ({exc})")
            index.append({"file": None, "source_run": str(run), "error": str(exc)})
            continue
        meta = {"source_run": run.name, "state_key": key}
        for k in ("epoch", "best_epoch", "val_score", "best_val", "args", "config", "arch", "model_name"):
            if k in ckpt and not hasattr(ckpt[k], "shape"):
                meta[k] = ckpt[k]
        bp = run / "best_point.json"
        if bp.exists():
            meta["best_point"] = json.load(open(bp, encoding="utf-8"))
        dst = out / f"{run.name}.pt"
        torch.save({"model": state, "meta": meta}, dst)
        mb_in, mb_out = src.stat().st_size / 2**20, dst.stat().st_size / 2**20
        print(f"{run.name}: {mb_in:.0f} MB -> {mb_out:.0f} MB")
        index.append({"file": dst.name, "source_run": str(run), "size_mb": round(mb_out, 1)})
    json.dump(index, open(out / "index.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
