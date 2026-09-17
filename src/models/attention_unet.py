"""Attention U-Net (Oktay et al., 2018, arXiv:1804.03999).

Adapted from the reference notebook — its strongest deep model and the one the
new architectures must beat. Attention gates re-weight skip connections to focus
on sparse targets and suppress clutter. Single-channel logit output.

Two optional heads extend it without altering the segmentation path:

``cls_head``  an auxiliary presence classifier tapped off the bottleneck
              (multi-task hard parameter sharing). ``forward`` then returns
              ``(seg_logits, cls_logits)``.
``temporal``  U-TAE-style temporal attention over a short sequence of scans. The
              encoder runs on every frame with shared weights, L-TAE aggregates
              at the bottleneck, and its attention masks collapse the skips. The
              decoder and attention gates below are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .temporal import LTAE2d, TemporalAggregator
from .unet import ConvBlock


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 6, base_filters: int = 32,
                 cls_head: bool = False, temporal: bool = False,
                 temporal_heads: int = 8, temporal_dk: int = 8,
                 cls_img_size: int = 384):
        super().__init__()
        f = base_filters
        self.cls_head_enabled = cls_head
        self.temporal = temporal
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.att4 = AttentionGate(f * 8, f * 8, f * 4)
        self.dec4 = ConvBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.att3 = AttentionGate(f * 4, f * 4, f * 2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.att2 = AttentionGate(f * 2, f * 2, f)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.att1 = AttentionGate(f, f, f // 2)
        self.dec1 = ConvBlock(f * 2, f)
        self.out = nn.Conv2d(f, 1, 1)

        # Auxiliary presence classifier on the shared bottleneck. AdaptiveAvgPool
        # keeps it resolution-agnostic, so the same weights work on 256x256
        # training patches and on 960x960 full scenes.
        if cls_head:
            self.cls_pool = nn.AdaptiveAvgPool2d(1)
            self.cls_fc = nn.Linear(f * 16, 1)
            # Full-scan classify() resizes to this; conv encoder is size-agnostic.
            self.cls_img_size = int(cls_img_size)

        if temporal:
            self.ltae = LTAE2d(f * 16, n_head=temporal_heads, d_k=temporal_dk)
            self.temporal_agg = TemporalAggregator(n_head=temporal_heads)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def classify(self, x):
        """Scan-level logit from a full (possibly resized) scene — not a patch crop."""
        if not self.cls_head_enabled:
            raise RuntimeError("classify() requires cls_head=True")
        _e1, _e2, _e3, _e4, b = self._encode(x)
        return self.cls_fc(self.cls_pool(b).flatten(1))

    def forward(self, x, pad_mask=None):
        if self.temporal:
            if x.dim() != 5:
                raise ValueError(f"temporal model expects (B,T,C,H,W), got {tuple(x.shape)}")
            bs, t = x.shape[:2]
            # Shared spatial encoder over every frame: fold time into the batch.
            e1, e2, e3, e4, b = self._encode(x.reshape(bs * t, *x.shape[2:]))
            b = b.view(bs, t, *b.shape[1:])
            b, attention = self.ltae(b, pad_mask)
            # Reuse the bottleneck's temporal weighting on each skip.
            e1, e2, e3, e4 = (
                self.temporal_agg(e.view(bs, t, *e.shape[1:]), attention)
                for e in (e1, e2, e3, e4)
            )
        else:
            if x.dim() != 4:
                raise ValueError(f"non-temporal model expects (B,C,H,W), got {tuple(x.shape)}")
            e1, e2, e3, e4, b = self._encode(x)

        cls_logits = self.cls_fc(self.cls_pool(b).flatten(1)) if self.cls_head_enabled else None

        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, self.att4(g4, e4)], dim=1))
        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))
        seg_logits = self.out(d1)
        return (seg_logits, cls_logits) if self.cls_head_enabled else seg_logits
