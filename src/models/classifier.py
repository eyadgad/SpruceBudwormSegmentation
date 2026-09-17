"""ImageNet-pretrained scan classifiers via timm.

Selected for a single-GPU, ~2k-scan radar transfer setting (see
configs/experiments_cls.yaml). Each model maps (B, C, H, W) -> (B, 1) logits.
The first conv is re-initialized when C != 3.
"""
from __future__ import annotations

from typing import Dict

import torch.nn as nn

# Short name -> timm id. Kept here so configs stay readable.
TIMM_IDS = {
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "resnext50": "resnext50_32x4d",
}


def build_classifier(cfg: Dict) -> nn.Module:
    try:
        import timm
    except ImportError as e:
        raise ImportError("timm is required for scan classifiers") from e

    name = cfg["model"]["name"]
    if name not in TIMM_IDS:
        raise ValueError(f"Unknown classifier '{name}'. Known: {sorted(TIMM_IDS)}")
    img_size = int(cfg["model"].get("img_size", 384))
    kwargs = dict(
        pretrained=bool(cfg["model"].get("pretrained", True)),
        in_chans=len(cfg["channels"]),
        num_classes=1,
    )
    # Swin position embeddings are resolution-specific.
    if name == "swin_tiny":
        kwargs["img_size"] = img_size
    return timm.create_model(TIMM_IDS[name], **kwargs)
