from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mura_rca_data import (
    EXTENTS,
    FAULT_MODES,
    MASK_TYPES,
    POLARITIES,
    PROCESS_STEPS,
    SUBTYPES,
    TOOL_IDS,
)


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Small U-Net lower-bound baseline for panel segmentation."""

    def __init__(self, in_ch: int = 1, num_classes: int = 6, base: int = 32) -> None:
        super().__init__()
        self.e1 = conv_block(in_ch, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.e4 = conv_block(base * 4, base * 8)
        self.b = conv_block(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = conv_block(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = conv_block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


class SwinUNETRSeg(nn.Module):
    def __init__(self, num_classes: int = 6, in_ch: int = 1, image_size: int = 384) -> None:
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError("Install MONAI to use Swin UNETR: pip install monai") from exc
        kwargs = {
            "in_channels": in_ch,
            "out_channels": num_classes,
            "feature_size": 24,
            "spatial_dims": 2,
            "use_checkpoint": False,
        }
        try:
            self.model = SwinUNETR(img_size=(image_size, image_size), **kwargs)
        except TypeError:
            self.model = SwinUNETR(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TimmFeatureSeg(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_classes: int = 6,
        in_ch: int = 1,
        decoder_ch: int = 128,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use VMamba/timm segmentation: pip install timm") from exc
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            in_chans=in_ch,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = [int(info["num_chs"]) for info in self.encoder.feature_info]
        self.proj = nn.ModuleList([nn.Conv2d(ch, decoder_ch, 1) for ch in channels])
        self.head = nn.Sequential(
            conv_block(decoder_ch, decoder_ch),
            nn.Conv2d(decoder_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = self.encoder(x)
        out = None
        for feat, proj in zip(feats, self.proj):
            item = F.interpolate(proj(feat), size=size, mode="bilinear", align_corners=False)
            out = item if out is None else out + item
        return self.head(out)


def build_vmamba_model(
    num_classes: int = 6,
    in_ch: int = 1,
    encoder_name: str | None = None,
) -> nn.Module:
    candidates = []
    if encoder_name:
        candidates.append(encoder_name)
    candidates.extend(
        [
            "vmamba_tiny_s1l8",
            "vmamba_tiny_s2l15",
            "vmamba_small_s2l15",
        ]
    )
    errors = []
    for name in candidates:
        try:
            return TimmFeatureSeg(name, num_classes=num_classes, in_ch=in_ch)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "Could not create a VMamba timm feature encoder. Install a timm/VMamba build "
        "that exposes one of the tried model names, or pass --encoder_name. Tried:\n"
        + "\n".join(errors)
    )


def build_segmentation_model(
    name: str,
    num_classes: int = 6,
    in_ch: int = 1,
    image_size: int = 384,
    encoder_name: str | None = None,
) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return SmallUNet(in_ch=in_ch, num_classes=num_classes)
    if name == "swin_unetr":
        return SwinUNETRSeg(num_classes=num_classes, in_ch=in_ch, image_size=image_size)
    if name == "vmamba":
        return build_vmamba_model(num_classes=num_classes, in_ch=in_ch, encoder_name=encoder_name)
    if name == "hrnet":
        return TimmFeatureSeg(encoder_name or "hrnet_w18", num_classes=num_classes, in_ch=in_ch)
    if name in {"smp_unet", "segformer", "deeplabv3plus"}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "Install segmentation-models-pytorch to use this model: "
                "pip install segmentation-models-pytorch timm"
            ) from exc
        if name == "smp_unet":
            return smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=in_ch,
                classes=num_classes,
            )
        if name == "deeplabv3plus":
            if not hasattr(smp, "DeepLabV3Plus"):
                raise RuntimeError("Your segmentation_models_pytorch version has no DeepLabV3Plus class.")
            return smp.DeepLabV3Plus(
                encoder_name=encoder_name or "resnet34",
                encoder_weights=None,
                in_channels=in_ch,
                classes=num_classes,
            )
        if not hasattr(smp, "Segformer"):
            raise RuntimeError("Your segmentation_models_pytorch version has no Segformer class.")
        return smp.Segformer(
            encoder_name="mit_b1",
            encoder_weights=None,
            in_channels=in_ch,
            classes=num_classes,
        )
    raise ValueError(f"unknown segmentation model: {name}")


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1, 32),
            nn.MaxPool2d(2),
            conv_block(32, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            nn.MaxPool2d(2),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.proj(feat)


class RouteEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.tool_emb = nn.Embedding(len(TOOL_IDS), embed_dim)
        self.step_emb = nn.Embedding(len(PROCESS_STEPS), embed_dim)
        self.cycle_emb = nn.Embedding(4, embed_dim)
        self.pos_emb = nn.Embedding(32, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(
        self,
        route_tool: torch.Tensor,
        route_step: torch.Tensor,
        route_cycle: torch.Tensor,
        route_pos: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.tool_emb(route_tool)
            + self.step_emb(route_step)
            + self.cycle_emb(route_cycle.clamp_min(0).clamp_max(3))
            + self.pos_emb(route_pos.clamp_min(0).clamp_max(31))
        )
        return self.encoder(x)


class DenseChainGATLayer(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor, route_pos: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        pos_i = route_pos[:, :, None]
        pos_j = route_pos[:, None, :]
        chain_mask = torch.abs(pos_i - pos_j) <= 1
        scores = scores.masked_fill(~chain_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return self.out(torch.matmul(attn, v) + x)


class RouteGATEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, depth: int = 2) -> None:
        super().__init__()
        self.tool_emb = nn.Embedding(len(TOOL_IDS), embed_dim)
        self.step_emb = nn.Embedding(len(PROCESS_STEPS), embed_dim)
        self.cycle_emb = nn.Embedding(4, embed_dim)
        self.pos_emb = nn.Embedding(32, embed_dim)
        self.layers = nn.ModuleList([DenseChainGATLayer(embed_dim) for _ in range(depth)])

    def forward(
        self,
        route_tool: torch.Tensor,
        route_step: torch.Tensor,
        route_cycle: torch.Tensor,
        route_pos: torch.Tensor,
    ) -> torch.Tensor:
        route_pos_clamped = route_pos.clamp_min(0).clamp_max(31)
        x = (
            self.tool_emb(route_tool)
            + self.step_emb(route_step)
            + self.cycle_emb(route_cycle.clamp_min(0).clamp_max(3))
            + self.pos_emb(route_pos_clamped)
        )
        for layer in self.layers:
            x = layer(x, route_pos_clamped)
        return x


class PositionTimeEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        return self.net(position)


class PanelSinusoidalEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4 for row/col sinusoidal encoding")
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        pos = position[:, :2]
        n_freq = self.embed_dim // 4
        freqs = (2.0 ** torch.arange(n_freq, device=position.device, dtype=position.dtype)) * math.pi
        outs = []
        for axis in range(2):
            x = pos[:, axis : axis + 1] * freqs
            outs.append(torch.sin(x))
            outs.append(torch.cos(x))
        return self.proj(torch.cat(outs, dim=-1))


@dataclass
class RCAModelConfig:
    mode: str = "visionroute"
    embed_dim: int = 256
    n_time_bins: int = 10
    route_depth: int = 2
    route_heads: int = 4


class DefectRCAModel(nn.Module):
    """Defect-token multimodal RCA model.

    Modes:
    - image_only: crop image -> heads
    - image_position: image + position/time scalar features
    - route_only: route graph only
    - late_fusion: concat image + route + position
    - rcfilm: row/col sinusoidal + route-GAT FiLM conditioning of image feature
    - visionroute: image/position token cross-attends route nodes
    """

    def __init__(self, cfg: RCAModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.image_encoder = ImageEncoder(d)
        self.route_encoder = RouteEncoder(d, depth=cfg.route_depth, heads=cfg.route_heads)
        self.route_gat_encoder = RouteGATEncoder(d, depth=cfg.route_depth)
        self.pos_encoder = PositionTimeEncoder(d)
        self.panel_pos_encoder = PanelSinusoidalEncoder(d)
        self.rcfilm = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
        )
        self.cross = nn.MultiheadAttention(d, cfg.route_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.fuse = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleDict(
            {
                "mask_type": nn.Linear(d, len(MASK_TYPES)),
                "extent": nn.Linear(d, len(EXTENTS)),
                "subtype": nn.Linear(d, len(SUBTYPES)),
                "polarity": nn.Linear(d, len(POLARITIES)),
                "semu": nn.Linear(d, 1),
                "is_reject": nn.Linear(d, 1),
                "process_step": nn.Linear(d, len(PROCESS_STEPS)),
                "tool_id": nn.Linear(d, len(TOOL_IDS)),
                "fault_mode": nn.Linear(d, len(FAULT_MODES)),
                "time_bin": nn.Linear(d, cfg.n_time_bins + 1),
            }
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mode = self.cfg.mode
        img_feat = self.image_encoder(batch["image"])
        pos_feat = self.pos_encoder(batch["position"])
        route_nodes = self.route_encoder(
            batch["route_tool"],
            batch["route_step"],
            batch["route_cycle"],
            batch["route_pos"],
        )
        route_feat = route_nodes.mean(dim=1)

        if mode == "image_only":
            feat = img_feat
        elif mode == "image_position":
            feat = img_feat + pos_feat
        elif mode == "route_only":
            feat = route_feat
        elif mode == "late_fusion":
            feat = img_feat + pos_feat + route_feat
        elif mode == "rcfilm":
            route_gat_nodes = self.route_gat_encoder(
                batch["route_tool"],
                batch["route_step"],
                batch["route_cycle"],
                batch["route_pos"],
            )
            route_gat_feat = route_gat_nodes.mean(dim=1)
            panel_pos_feat = self.panel_pos_encoder(batch["position"])
            gamma, beta = self.rcfilm(torch.cat([panel_pos_feat, route_gat_feat], dim=1)).chunk(2, dim=1)
            feat = self.norm((1.0 + gamma) * img_feat + beta)
        elif mode == "visionroute":
            query = (img_feat + pos_feat).unsqueeze(1)
            attended, _attn = self.cross(query, route_nodes, route_nodes, need_weights=False)
            feat = self.norm(query.squeeze(1) + attended.squeeze(1))
        else:
            raise ValueError(f"unknown RCA mode: {mode}")
        feat = self.fuse(feat)
        out = {name: head(feat) for name, head in self.heads.items()}
        out["semu"] = out["semu"].squeeze(1)
        out["is_reject"] = out["is_reject"].squeeze(1)
        return out
