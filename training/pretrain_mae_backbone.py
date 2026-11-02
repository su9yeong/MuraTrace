"""Self-supervised (SimMIM-style) backbone pretraining on real backgrounds.

Option 4 in the analysis: pretrain the segmentation encoder on the unlabeled
real display backgrounds in `real_background/` before fine-tuning, to close the
synthetic->real domain gap on low-contrast mura.

The encoder is built through segmentation_models_pytorch's own encoder factory,
so the saved `state_dict` keys align exactly with the encoder inside the
advanced seg model. Feed the resulting checkpoint to
`train_segmentation_advanced.py --init_backbone <ckpt>`.

This is a pretraining *producer*: it has no test_eval pair. Its evaluation is
the downstream advanced run + advanced evaluator. We log reconstruction loss to
metrics.jsonl for sanity.

Inputs are converted to grayscale then replicated to 3 channels and kept in
[0,1] to match the segmentation model's input distribution (gray->RGB adapter,
no ImageNet normalization at fine-tune time).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from train_segmentation import write_json


class BackgroundPatches(Dataset):
    """Random grayscale crops from a directory of background images."""

    def __init__(self, image_dir: str, image_size: int = 256, samples_per_epoch: int = 2000) -> None:
        self.paths = sorted(
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not self.paths:
            raise FileNotFoundError(f"no images found in {image_dir}")
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            img = Image.open(path).convert("L")
            self._cache[path] = np.asarray(img, dtype=np.float32) / 255.0
        return self._cache[path]

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = self._load(random.choice(self.paths))
        h, w = arr.shape
        s = self.image_size
        if h < s or w < s:
            arr = np.asarray(
                Image.fromarray((arr * 255).astype(np.uint8)).resize((max(w, s), max(h, s))),
                dtype=np.float32,
            ) / 255.0
            h, w = arr.shape
        y0 = random.randint(0, h - s)
        x0 = random.randint(0, w - s)
        crop = arr[y0:y0 + s, x0:x0 + s]
        if random.random() < 0.5:
            crop = crop[:, ::-1]
        if random.random() < 0.5:
            crop = crop[::-1, :]
        crop = np.ascontiguousarray(crop)
        return torch.from_numpy(crop)[None].repeat(3, 1, 1)  # [3,H,W]


class SimMIMDecoder(nn.Module):
    """Lightweight decoder: deepest encoder feature -> reconstructed image."""

    def __init__(self, in_ch: int, out_ch: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, out_ch, 1),
        )

    def forward(self, feat: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.net(feat)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def random_mask(x: torch.Tensor, patch: int, ratio: float) -> torch.Tensor:
    """Return a [B,1,H,W] mask (1 = masked) on a patch grid."""
    b, _, h, w = x.shape
    gh, gw = h // patch, w // patch
    grid = (torch.rand(b, 1, gh, gw, device=x.device) < ratio).float()
    return F.interpolate(grid, size=(h, w), mode="nearest")


def main() -> None:
    ap = argparse.ArgumentParser(description="SimMIM backbone pretraining on real backgrounds.")
    ap.add_argument("--image_dir", default="real_background")
    ap.add_argument("--out_dir", default="runs/mae_backbone")
    ap.add_argument("--encoder_name", default="resnet34", help="must match the downstream advanced --encoder_name")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--samples_per_epoch", type=int, default=2000)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--mask_patch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    try:
        from segmentation_models_pytorch.encoders import get_encoder
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install segmentation-models-pytorch + timm to pretrain the backbone.") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(out_dir / "run_config.json", {"args": vars(args), "device": str(device)})

    encoder = get_encoder(args.encoder_name, in_channels=3, depth=5, weights=None).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        deepest = encoder(dummy)[-1]
    decoder = SimMIMDecoder(in_ch=deepest.shape[1], out_ch=3).to(device)

    ds = BackgroundPatches(args.image_dir, image_size=args.image_size, samples_per_epoch=args.samples_per_epoch)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    best = float("inf")
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            encoder.train()
            decoder.train()
            total, n = 0.0, 0
            for x in loader:
                x = x.to(device, non_blocking=True)
                mask = random_mask(x, args.mask_patch, args.mask_ratio)
                masked_input = x * (1.0 - mask)
                with autocast(enabled=args.amp):
                    feat = encoder(masked_input)[-1]
                    recon = decoder(feat, size=x.shape[-2:])
                    diff = (recon - x).abs() * mask
                    loss = diff.sum() / (mask.sum() * x.shape[1] + 1e-6)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total += float(loss.item())
                n += 1
            avg = total / max(n, 1)
            log.write(json.dumps({"epoch": epoch, "recon_l1": avg}) + "\n")
            log.flush()
            print(f"epoch {epoch} recon_l1={avg:.5f}", flush=True)
            ckpt = {"epoch": epoch, "encoder": encoder.state_dict(), "encoder_name": args.encoder_name, "args": vars(args), "recon_l1": avg}
            torch.save(ckpt, out_dir / "last.pt")
            if avg < best:
                best = avg
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", {"epoch": epoch, "recon_l1": avg, "checkpoint": "best.pt"})


if __name__ == "__main__":
    main()
