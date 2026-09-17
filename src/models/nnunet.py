"""nnU-Net-style U-Net.

This is a self-contained reproduction of nnU-Net's default 2D network recipe
(Isensee et al., 2021, Nature Methods) — a "PlainConvUNet": per-stage double
convolutions with InstanceNorm + LeakyReLU(0.01), strided-conv downsampling (no
max-pool), and transposed-conv upsampling. It is trained inside THIS framework's
shared loop, so it is the nnU-Net *architecture*, not the full self-configuring
nnU-Net autoML pipeline (which runs its own preprocessing/CV/inference and does
not fit a shared common interface). Documented as such in the README.

Output: single-channel logits at input resolution. Requires H,W divisible by
2**(num_stages-1).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _norm_act(ch):
    return nn.Sequential(nn.InstanceNorm2d(ch, affine=True), nn.LeakyReLU(0.01, inplace=True))


class StageBlock(nn.Module):
    """Two convs; the first optionally strides to downsample."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _norm_act(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm_act(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class NNUNet(nn.Module):
    def __init__(self, in_channels: int = 6, base_filters: int = 32,
                 num_stages: int = 5, max_filters: int = 320):
        super().__init__()
        feats: List[int] = [min(base_filters * (2 ** i), max_filters) for i in range(num_stages)]
        self.num_stages = num_stages

        # Encoder: stage 0 keeps resolution, stages 1..n-1 downsample via stride-2 conv.
        self.enc = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(feats):
            self.enc.append(StageBlock(prev, ch, stride=1 if i == 0 else 2))
            prev = ch

        # Decoder: transposed conv upsample + concat skip + double conv.
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(num_stages - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(feats[i], feats[i - 1], 2, stride=2))
            self.dec.append(StageBlock(feats[i - 1] * 2, feats[i - 1], stride=1))
        self.out = nn.Conv2d(feats[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, stage in enumerate(self.enc):
            x = stage(x)
            if i < self.num_stages - 1:
                skips.append(x)
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            skip = skips[-(j + 1)]
            x = dec(torch.cat([x, skip], dim=1))
        return self.out(x)
