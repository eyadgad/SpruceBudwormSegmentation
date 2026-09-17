"""Model factory — the common interface.

``create_model(cfg)`` returns an ``nn.Module`` mapping (B, C, H, W) inputs to
(B, 1, H, W) logits for every architecture in the Phase 2 shortlist:

  unet            baseline U-Net
  attention_unet  Attention U-Net (notebook's best deep model)
  nnunet          nnU-Net-style PlainConvUNet
  smp_unetpp      U-Net++ (SMP, pretrained encoder)
  smp_deeplabv3p  DeepLabV3+ (SMP, pretrained encoder)
  smp_segformer   SegFormer (SMP, MiT encoder)
"""
from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .unet import SimpleUNet
from .attention_unet import AttentionUNet
from .nnunet import NNUNet


def create_model(cfg: Dict) -> nn.Module:
    m = cfg["model"]
    name = m["name"]
    in_channels = len(cfg["channels"])
    base_filters = int(m.get("base_filters", 32))

    # Temporal settings live in one top-level block (see dataset.temporal_cfg),
    # so the dataset and the model can never disagree about the input layout.
    tc = cfg.get("temporal") or {}
    temporal_enabled = bool(tc.get("enabled", False))
    stacked = temporal_enabled and str(tc.get("mode", "attention")) == "stack"
    if stacked:
        # The 2.5D control concatenates neighbour frames onto the channel axis,
        # so the first conv is wider. Attention fusion keeps the time axis
        # separate and leaves the channel count alone.
        in_channels *= 2 * int(tc.get("radius", 1)) + 1

    if cfg.get("task") == "classify" or name in (
        "convnext_tiny", "efficientnetv2_s", "swin_tiny", "resnext50",
    ):
        from .classifier import build_classifier
        return build_classifier(cfg)

    if name in ("gated_swin_attn", "swin_unet", "transunet"):
        from .gated_hybrids import GatedSwinAttnUNet, SwinUNet, TransUNet
        img_size = int(m.get("img_size", 256))
        if name == "gated_swin_attn":
            return GatedSwinAttnUNet(in_channels=in_channels, img_size=img_size)
        if name == "swin_unet":
            return SwinUNet(in_channels=in_channels, img_size=img_size)
        return TransUNet(in_channels=in_channels)

    if name == "unet":
        return SimpleUNet(in_channels=in_channels, base_filters=base_filters)
    if name == "attention_unet":
        return AttentionUNet(
            in_channels=in_channels, base_filters=base_filters,
            cls_head=bool(m.get("cls_head", False)),
            # Only attention fusion changes the network; the stack control is an
            # ordinary 2D model that happens to have more input channels.
            temporal=temporal_enabled and not stacked,
            temporal_heads=int(tc.get("n_head", 8)),
            temporal_dk=int(tc.get("d_k", 8)),
            cls_img_size=int(m.get("img_size", 384)),
        )
    if name == "nnunet":
        return NNUNet(in_channels=in_channels, base_filters=base_filters,
                      num_stages=int(m.get("num_stages", 5)))
    if name.startswith("smp_"):
        from .smp_wrap import build_smp
        return build_smp(name, cfg, in_channels)
    raise ValueError(f"Unknown model '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def seg_logits(out):
    """Segmentation logits from a model output.

    Models with an auxiliary classification head return ``(seg, cls)``; every
    other architecture returns a bare tensor. Inference paths only ever want the
    segmentation map, so they funnel through here rather than each learning the
    multi-task calling convention.
    """
    return out[0] if isinstance(out, tuple) else out
