"""Single-model cascade-like segmenters, adapted from public implementations.

- Swin-UNet (Cao et al., ECCVW 2022; github.com/HuCaoFighting/Swin-Unet):
  hierarchical Swin encoder + skip decoder. Official code uses a Swin
  patch-expanding decoder; here the encoder is the published Swin-T via
  timm and the decoder is a conv CUP (same skip/upsample idea, 10-ch radar).
- TransUNet (Chen et al.; github.com/Beckschen/TransUNet):
  CNN encoder (ResNet-34) + Transformer on the bottleneck + CUP decoder.
- Gated Swin-AttnUNet: Swin-T encoder + Attention U-Net decoder (Oktay).

All three expose ``classify(x)`` (scan-level logit from the bottleneck) and
return ``(seg_logits, cls_logits)`` from ``forward``. The scan-level gate
``p = σ(z_cls) * σ(z_seg)`` is applied at evaluation, not inside the decoder
loss, so the pixel head can still learn plumes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_unet import AttentionGate
from .unet import ConvBlock


def _nchw(t: torch.Tensor) -> torch.Tensor:
    """timm Swin features_only is NHWC; ResNet is NCHW."""
    if t.dim() == 4 and t.shape[1] == t.shape[2] and t.shape[-1] >= 16:
        return t.permute(0, 3, 1, 2).contiguous()
    return t


class _ClsHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(_nchw(feat)).flatten(1))


class _ConvDecoder(nn.Module):
    """CUP-style conv decoder with skip concatenations (TransUNet / Swin-UNet ports)."""

    def __init__(self, skip_chs, bot_ch, out_ch=32):
        super().__init__()
        chs = [bot_ch] + list(skip_chs[::-1])
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        prev = bot_ch
        for skip in skip_chs[::-1]:
            self.ups.append(nn.ConvTranspose2d(prev, skip, 2, stride=2))
            self.decs.append(ConvBlock(skip + skip, skip))
            prev = skip
        # 64 -> 128 -> 256 when the last skip is 64px (Swin / ResNet stage-1).
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(prev, out_ch, 2, stride=2),
            ConvBlock(out_ch, out_ch),
            nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2),
            ConvBlock(out_ch, out_ch),
            nn.Conv2d(out_ch, 1, 1),
        )

    def forward(self, bot, skips):
        x = _nchw(bot)
        for up, dec, skip in zip(self.ups, self.decs, skips[::-1]):
            x = up(x)
            s = _nchw(skip)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, s], dim=1))
        return self.tail(x)


class _AttnDecoder(nn.Module):
    """Attention-gated decoder (Oktay) on a Swin feature pyramid."""

    def __init__(self, chs):
        super().__init__()
        c0, c1, c2, c3 = chs
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.att3 = AttentionGate(c2, c2, c2 // 2)
        self.dec3 = ConvBlock(c2 + c2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.att2 = AttentionGate(c1, c1, c1 // 2)
        self.dec2 = ConvBlock(c1 + c1, c1)
        self.up1 = nn.ConvTranspose2d(c1, c0, 2, stride=2)
        self.att1 = AttentionGate(c0, c0, c0 // 2)
        self.dec1 = ConvBlock(c0 + c0, c0)
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(c0, 32, 2, stride=2),
            ConvBlock(32, 32),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            ConvBlock(32, 32),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, bot, skips):
        e0, e1, e2 = (_nchw(s) for s in skips)
        b = _nchw(bot)
        x = self.up3(b)
        if x.shape[-2:] != e2.shape[-2:]:
            x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, self.att3(x, e2)], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, self.att2(x, e1)], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, self.att1(x, e0)], dim=1))
        return self.tail(x)


class _BottleneckViT(nn.Module):
    """Tiny Transformer on a CNN feature map (TransUNet encoder-only idea)."""

    def __init__(self, dim: int, depth: int = 4, heads: int = 8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tok = x.flatten(2).transpose(1, 2)
        tok = self.norm(self.enc(tok))
        return tok.transpose(1, 2).reshape(b, c, h, w)


def _make_swin(in_channels: int, img_size: int = 256):
    import timm
    return timm.create_model(
        "swin_tiny_patch4_window7_224", pretrained=True,
        features_only=True, in_chans=in_channels, img_size=img_size,
    )


class GatedSwinAttnUNet(nn.Module):
    """Swin-T encoder + Attention U-Net decoder + scan-level cls head."""

    def __init__(self, in_channels: int = 10, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.encoder = _make_swin(in_channels, img_size)
        chs = [96, 192, 384, 768]
        self.decoder = _AttnDecoder(chs)
        self.cls = _ClsHead(chs[-1])

    def _feats(self, x):
        return [_nchw(f) for f in self.encoder(x)]

    def classify(self, x):
        return self.cls(self._feats(x)[-1])

    def forward(self, x, pad_mask=None):
        feats = self._feats(x)
        seg = self.decoder(feats[-1], feats[:-1])
        if seg.shape[-2:] != x.shape[-2:]:
            seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return seg, self.cls(feats[-1])


class SwinUNet(nn.Module):
    """Swin-UNet-style: Swin-T encoder + conv CUP decoder (official skip idea)."""

    def __init__(self, in_channels: int = 10, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.encoder = _make_swin(in_channels, img_size)
        chs = [96, 192, 384, 768]
        self.decoder = _ConvDecoder(chs[:-1], chs[-1])
        self.cls = _ClsHead(chs[-1])

    def _feats(self, x):
        return [_nchw(f) for f in self.encoder(x)]

    def classify(self, x):
        return self.cls(self._feats(x)[-1])

    def forward(self, x, pad_mask=None):
        feats = self._feats(x)
        seg = self.decoder(feats[-1], feats[:-1])
        if seg.shape[-2:] != x.shape[-2:]:
            seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return seg, self.cls(feats[-1])


class TransUNet(nn.Module):
    """TransUNet hybrid: ResNet-34 skips + bottleneck Transformer + CUP decoder."""

    def __init__(self, in_channels: int = 10, vit_dim: int = 256, vit_depth: int = 4):
        super().__init__()
        self.img_size = 256
        import timm
        self.encoder = timm.create_model(
            "resnet34", pretrained=True, features_only=True, in_chans=in_channels,
        )
        self.proj = nn.Conv2d(512, vit_dim, 1)
        self.vit = _BottleneckViT(vit_dim, depth=vit_depth, heads=8)
        self.unproj = nn.Conv2d(vit_dim, 512, 1)
        # skips: 64@64, 128@32, 256@16  (drop 128px stem)
        self.decoder = _ConvDecoder([64, 128, 256], 512)
        self.cls = _ClsHead(512)

    def _feats(self, x):
        raw = self.encoder(x)
        # raw: stem 64@128, l1 64@64, l2 128@32, l3 256@16, l4 512@8
        skips = raw[1:4]
        bot = self.unproj(self.vit(self.proj(raw[-1])))
        return skips, bot

    def classify(self, x):
        _, bot = self._feats(x)
        return self.cls(bot)

    def forward(self, x, pad_mask=None):
        skips, bot = self._feats(x)
        seg = self.decoder(bot, list(skips))
        if seg.shape[-2:] != x.shape[-2:]:
            seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return seg, self.cls(bot)
