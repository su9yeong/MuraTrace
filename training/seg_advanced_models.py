"""Shared builders for the advanced segmentation methodologies.

This module centralizes model construction so that the *advanced* trainer
(`train_segmentation_advanced.py`) and its paired evaluator
(`evaluate_segmentation_advanced.py`) build byte-for-byte identical
architectures. The training checkpoint stores every architecture-determining
argument under `ckpt["args"]`; the evaluator rebuilds from those same args with
`pretrained=False` (the ImageNet weights are irrelevant at eval time because the
fine-tuned `state_dict` is loaded afterwards).

It covers the per-pixel segmentation options recommended for MuraTrace:
  - 0: ImageNet-pretrained encoders (currently the legacy runs use no pretraining)
  - 1: global-context backbones (SegFormer/MiT, ConvNeXt, Swin) and UPerNet decoder
  - 3: high-resolution HRNet encoder

The MuraTrace panel images are single channel. ImageNet weights expect 3
channels, so a thin gray->RGB adapter replicates the gray channel into 3 before
the encoder. This is the simplest faithful way to reuse RGB pretraining and is
the approach the analysis recommended.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# arch name -> segmentation_models_pytorch class attribute name
ARCH_TO_SMP_CLASS = {
    "deeplabv3plus": "DeepLabV3Plus",
    "deeplabv3": "DeepLabV3",
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "upernet": "UPerNet",
    "manet": "MAnet",
    "pan": "PAN",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "linknet": "Linknet",
    "segformer": "Segformer",
}


class GrayToRGBAdapter(nn.Module):
    """Replicate a single gray channel into 3 channels, then run the core model.

    Keeps the public forward signature identical to a plain SMP model so the
    shared `run_epoch` (which calls `model(batch["image"])` with a [B,1,H,W]
    tensor) works unchanged.
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.core(x)


def _import_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Install segmentation-models-pytorch to use the advanced segmentation "
            "models: pip install segmentation-models-pytorch timm"
        ) from exc
    return smp


# transformer encoders whose patch-embed asserts a fixed input resolution;
# they accept a timm `img_size` kwarg to be built for a different resolution.
_FIXED_RES_TOKENS = ("swin", "vit", "deit", "beit", "eva", "pit", "coat", "twins", "maxvit")


def _needs_img_size(encoder_name: str) -> bool:
    n = encoder_name.lower()
    return any(tok in n for tok in _FIXED_RES_TOKENS)


def build_advanced_segmentation_model(
    arch: str,
    encoder_name: str,
    num_classes: int = 6,
    pretrained: bool = True,
    in_channels: int = 3,
    encoder_depth: int | None = None,
    image_size: int | None = None,
) -> nn.Module:
    """Build an SMP segmentation model with optional ImageNet pretraining.

    `in_channels == 3` (default) enables the gray->RGB adapter so ImageNet
    encoder weights load cleanly. `in_channels == 1` keeps the native
    single-channel stem (no pretraining benefit, kept for ablations).
    """
    smp = _import_smp()
    arch = arch.lower()
    if arch not in ARCH_TO_SMP_CLASS:
        raise ValueError(f"unknown advanced seg arch: {arch}. choices={sorted(ARCH_TO_SMP_CLASS)}")
    cls_name = ARCH_TO_SMP_CLASS[arch]
    if not hasattr(smp, cls_name):
        raise RuntimeError(
            f"Your segmentation_models_pytorch version has no {cls_name} class. "
            f"Upgrade smp (pip install -U segmentation-models-pytorch)."
        )
    model_cls = getattr(smp, cls_name)
    encoder_weights = "imagenet" if pretrained else None
    core_in_channels = 3 if in_channels == 3 else 1

    kwargs: dict[str, Any] = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": core_in_channels,
        "classes": num_classes,
    }
    if encoder_depth is not None:
        kwargs["encoder_depth"] = encoder_depth
    # fixed-resolution transformer encoders (Swin/ViT/...) must be built for the
    # training resolution; forward `img_size` through SMP -> timm.create_model.
    if image_size is not None and _needs_img_size(encoder_name):
        kwargs["img_size"] = int(image_size)

    try:
        core = model_cls(**kwargs)
    except TypeError as exc:
        # If SMP did not accept/forward `img_size`, we cannot set the encoder
        # resolution; building without it would only fail later at forward with
        # the fixed-resolution assert. Stop now with actionable guidance.
        if "img_size" in kwargs:
            raise RuntimeError(
                f"{cls_name}/{encoder_name!r} is fixed-resolution and your "
                f"segmentation_models_pytorch did not forward `img_size` to timm "
                f"(error: {exc}). Either upgrade smp+timm, or train this encoder "
                f"with --image_size 224 to match its native resolution."
            ) from exc
        raise RuntimeError(
            f"Failed to build {cls_name} with encoder_name={encoder_name!r}, "
            f"pretrained={pretrained}. Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build {cls_name} with encoder_name={encoder_name!r}, "
            f"pretrained={pretrained}. Some encoders need specific timm versions "
            f"or image sizes. Original error: {exc}"
        ) from exc

    if core_in_channels == 3:
        return GrayToRGBAdapter(core)
    return core


def encoder_module(model: nn.Module) -> nn.Module:
    """Return the SMP encoder submodule whether or not it is wrapped."""
    core = model.core if isinstance(model, GrayToRGBAdapter) else model
    if not hasattr(core, "encoder"):
        raise AttributeError("model has no .encoder submodule")
    return core.encoder


def load_backbone_weights(model: nn.Module, backbone_ckpt_path: str) -> dict[str, Any]:
    """Load a self-supervised (MAE/SimMIM) encoder checkpoint into the SMP encoder.

    Returns a small report of how many tensors matched. Non-matching keys are
    skipped so a backbone trained on a different head still transfers cleanly.
    """
    ckpt = torch.load(backbone_ckpt_path, map_location="cpu")
    state = ckpt.get("encoder", ckpt.get("model", ckpt))
    enc = encoder_module(model)
    enc_state = enc.state_dict()
    matched = {}
    for k, v in state.items():
        key = k
        # tolerate a leading "encoder." prefix from the pretrainer
        if key not in enc_state and key.startswith("encoder."):
            key = key[len("encoder."):]
        if key in enc_state and enc_state[key].shape == v.shape:
            matched[key] = v
    # Don't rely on the return value of load_state_dict (None in some versions);
    # compute the missing set ourselves.
    enc.load_state_dict(matched, strict=False)
    missing_keys = sorted(set(enc_state.keys()) - set(matched.keys()))
    return {
        "matched_tensors": len(matched),
        "encoder_tensors": len(enc_state),
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(set(matched.keys()) - set(enc_state.keys())),
    }


def build_model_from_ckpt_args(ckpt_args: dict[str, Any], num_classes: int = 6) -> nn.Module:
    """Rebuild a segmentation model from stored training args.

    Handles both *advanced* checkpoints (have an `arch` key) and *legacy*
    checkpoints produced by `train_segmentation.py` (use `model` + the
    `build_segmentation_model` factory). Used by the ensemble evaluator, which
    must load checkpoints from either trainer.
    """
    if ckpt_args.get("arch"):
        return build_advanced_segmentation_model(
            arch=str(ckpt_args["arch"]),
            encoder_name=str(ckpt_args.get("encoder_name") or "resnet34"),
            num_classes=num_classes,
            pretrained=False,  # weights overwritten by load_state_dict
            in_channels=int(ckpt_args.get("in_channels", 3)),
            image_size=int(ckpt_args.get("image_size", 512)),
        )
    # legacy path
    from mura_rca_models import build_segmentation_model

    return build_segmentation_model(
        str(ckpt_args.get("model", "deeplabv3plus")),
        num_classes=num_classes,
        in_ch=1,
        image_size=int(ckpt_args.get("image_size", 512)),
        encoder_name=ckpt_args.get("encoder_name"),
    )
