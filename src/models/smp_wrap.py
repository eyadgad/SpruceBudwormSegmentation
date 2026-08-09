"""U-Net++, DeepLabV3+, SegFormer via segmentation_models_pytorch (SMP).

Imported lazily inside the factory so the custom models (unet, attention_unet,
nnunet) work even if SMP is not installed. All return single-channel logits at
input resolution.
"""
from __future__ import annotations

from typing import Dict


def _require_smp():
    try:
        import segmentation_models_pytorch as smp  # noqa
    except ImportError as e:  # real dependency boundary
        raise ImportError(
            "segmentation_models_pytorch is required for smp_* models "
            "(pip install segmentation-models-pytorch). Custom models "
            "(unet, attention_unet, nnunet) do not need it."
        ) from e
    return smp


def build_smp(name: str, cfg: Dict, in_channels: int):
    smp = _require_smp()
    m = cfg["model"]
    encoder = m.get("encoder", "efficientnet-b2")
    weights = "imagenet" if m.get("pretrained", True) else None
    common = dict(in_channels=in_channels, classes=1)

    if name == "smp_unetpp":
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=weights, **common)
    if name == "smp_deeplabv3p":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=weights, **common)
    if name == "smp_segformer":
        # SegFormer uses MixVisionTransformer encoders ("mit_b0".."mit_b5").
        seg_encoder = m.get("encoder_segformer", "mit_b0")
        try:
            return smp.Segformer(encoder_name=seg_encoder, encoder_weights=weights, **common)
        except (KeyError, ValueError, AttributeError) as e:
            raise ValueError(
                f"Could not build SegFormer with encoder '{seg_encoder}' "
                f"(weights={weights}). Set model.encoder_segformer to a valid "
                f"mit_b0..mit_b5 and/or model.pretrained=false. Original: {e}"
            ) from e
    raise ValueError(f"Unknown SMP model: {name}")
