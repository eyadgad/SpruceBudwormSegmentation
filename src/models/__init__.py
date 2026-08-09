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

    if name == "unet":
        return SimpleUNet(in_channels=in_channels, base_filters=base_filters)
    if name == "attention_unet":
        return AttentionUNet(in_channels=in_channels, base_filters=base_filters)
    if name == "nnunet":
        return NNUNet(in_channels=in_channels, base_filters=base_filters,
                      num_stages=int(m.get("num_stages", 5)))
    if name.startswith("smp_"):
        from .smp_wrap import build_smp
        return build_smp(name, cfg, in_channels)
    raise ValueError(f"Unknown model '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
