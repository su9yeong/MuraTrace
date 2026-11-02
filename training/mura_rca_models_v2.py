"""RCA model V2: configurable image encoder + an `image_route` (no-position) mode.

Subclasses the base DefectRCAModel, swapping only the image encoder for a
pretrained backbone (see rca_image_encoders) and adding one fusion mode:
  - image_route : feat = image + route   (no position/time)

Everything else (route encoder, heads, other modes) is inherited unchanged, so
results stay comparable to the base RCA runs.
"""
from __future__ import annotations

import torch

from mura_rca_models import DefectRCAModel, RCAModelConfig  # noqa: F401 (re-exported)
from rca_image_encoders import build_rca_image_encoder

EXTRA_MODES = ["image_route"]


class DefectRCAModelV2(DefectRCAModel):
    def __init__(
        self,
        cfg: RCAModelConfig,
        image_encoder: str = "smallcnn",
        image_pretrained: bool = True,
        image_init_backbone: str | None = None,
        image_size: int = 224,
    ) -> None:
        super().__init__(cfg)
        if image_encoder != "smallcnn":
            self.image_encoder = build_rca_image_encoder(
                image_encoder, cfg.embed_dim, pretrained=image_pretrained,
                init_backbone=image_init_backbone, image_size=image_size,
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mode = self.cfg.mode
        img_feat = self.image_encoder(batch["image"])
        pos_feat = self.pos_encoder(batch["position"])
        route_nodes = self.route_encoder(
            batch["route_tool"], batch["route_step"], batch["route_cycle"], batch["route_pos"],
        )
        route_feat = route_nodes.mean(dim=1)

        if mode == "image_only":
            feat = img_feat
        elif mode == "image_position":
            feat = img_feat + pos_feat
        elif mode == "route_only":
            feat = route_feat
        elif mode == "image_route":  # new: image + route, no position/time
            feat = img_feat + route_feat
        elif mode == "late_fusion":
            feat = img_feat + pos_feat + route_feat
        elif mode == "rcfilm":
            route_gat_nodes = self.route_gat_encoder(
                batch["route_tool"], batch["route_step"], batch["route_cycle"], batch["route_pos"],
            )
            route_gat_feat = route_gat_nodes.mean(dim=1)
            panel_pos_feat = self.panel_pos_encoder(batch["position"])
            gamma, beta = self.rcfilm(torch.cat([panel_pos_feat, route_gat_feat], dim=1)).chunk(2, dim=1)
            feat = self.norm((1.0 + gamma) * img_feat + beta)
        elif mode == "visionroute":
            query = (img_feat + pos_feat).unsqueeze(1)
            attended, _ = self.cross(query, route_nodes, route_nodes, need_weights=False)
            feat = self.norm(query.squeeze(1) + attended.squeeze(1))
        else:
            raise ValueError(f"unknown RCA mode: {mode}")

        feat = self.fuse(feat)
        out = {name: head(feat) for name, head in self.heads.items()}
        out["semu"] = out["semu"].squeeze(1)
        out["is_reject"] = out["is_reject"].squeeze(1)
        return out
