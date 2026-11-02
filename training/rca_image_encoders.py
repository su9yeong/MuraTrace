"""Configurable image encoders for the RCA model.

The base RCA model uses a small from-scratch CNN (`ImageEncoder`). The
segmentation study showed that pretrained global-context backbones
(ConvNeXt/Swin) dramatically improve visual perception; this module lets the RCA
visual heads (subtype / reject / SEMU) reuse the same backbones.

All encoders take a [B,1,H,W] gray crop and return a [B, embed_dim] vector.

`pretrained` meanings:
  - imagenet : timm/SMP ImageNet weights (general visual prior; proven in seg).
  - init_backbone : load our OWN SimMIM/MAE backbone trained on real_background
    (domain pretraining "on our data"; currently available for resnet34).
The two can be combined: build an imagenet encoder, then overwrite with our
domain weights via init_backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# friendly name -> SMP/timm encoder name (224-native variants for RCA crops)
ENCODER_ALIAS = {
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "convnext": "tu-convnext_small",
    "convnext_tiny": "tu-convnext_tiny",
    "swin": "tu-swin_small_patch4_window7_224",
    "swin_tiny": "tu-swin_tiny_patch4_window7_224",
}


class BackbonePooledEncoder(nn.Module):
    """SMP/timm encoder -> global average pool -> linear to embed_dim."""

    def __init__(self, encoder_name: str, embed_dim: int, pretrained: bool, image_size: int = 224) -> None:
        super().__init__()
        try:
            from segmentation_models_pytorch.encoders import get_encoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install segmentation-models-pytorch + timm to use pretrained RCA encoders."
            ) from exc
        smp_name = ENCODER_ALIAS.get(encoder_name, encoder_name)
        kwargs = {"in_channels": 3, "depth": 5, "weights": "imagenet" if pretrained else None}
        # fixed-resolution transformer encoders need img_size at 224 (their native)
        if any(t in smp_name.lower() for t in ("swin", "vit", "beit", "deit")):
            kwargs["img_size"] = int(image_size)
        try:
            self.encoder = get_encoder(smp_name, **kwargs)
        except TypeError:
            kwargs.pop("img_size", None)
            self.encoder = get_encoder(smp_name, **kwargs)
        out_ch = self.encoder.out_channels[-1]
        self.proj = nn.Sequential(nn.Linear(out_ch, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.encoder(x)[-1]
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.proj(feat)


def load_domain_backbone(encoder: nn.Module, ckpt_path: str) -> dict:
    """Load our SimMIM/MAE encoder weights into a BackbonePooledEncoder.encoder."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("encoder", ckpt.get("model", ckpt))
    target = encoder.encoder if isinstance(encoder, BackbonePooledEncoder) else encoder
    tgt = target.state_dict()
    matched = {k: v for k, v in state.items() if k in tgt and tgt[k].shape == v.shape}
    target.load_state_dict(matched, strict=False)
    return {"matched": len(matched), "total": len(tgt)}


def build_rca_image_encoder(
    kind: str,
    embed_dim: int,
    pretrained: bool = True,
    init_backbone: str | None = None,
    image_size: int = 224,
) -> nn.Module:
    if kind == "smallcnn":
        from mura_rca_models import ImageEncoder
        return ImageEncoder(embed_dim)
    enc = BackbonePooledEncoder(kind, embed_dim, pretrained=pretrained, image_size=image_size)
    if init_backbone:
        rep = load_domain_backbone(enc, init_backbone)
        print(f"rca image encoder: loaded domain backbone {init_backbone} matched={rep['matched']}/{rep['total']}", flush=True)
    return enc
