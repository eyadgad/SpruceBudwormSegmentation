"""Temporal attention over successive radar scans (U-TAE / L-TAE style).

Follows Garnot & Landrieu:
  * "Panoptic Segmentation of Satellite Image Time Series with Convolutional
    Temporal Attention Networks", ICCV 2021 (arXiv:2107.07933) -- U-TAE.
  * "Lightweight Temporal Self-Attention for Classifying Satellite Image Time
    Series", ICPR-W 2020 (arXiv:2007.00586) -- L-TAE.

Why this and not ConvLSTM / 3D CNN. U-TAE reports ~63.1 mIoU on PASTIS, roughly
+5 mIoU over UNet-3d, UNet+ConvLSTM and FPN+ConvLSTM, and L-TAE matches temporal
attention encoders with ~10x its parameters and recurrent units with >300x. Just
as importantly for this repository, the shape of the method leaves the 2D
segmentation network intact: the spatial encoder is applied to every frame with
shared weights, temporal attention runs only at the lowest-resolution
bottleneck, and the resulting attention masks are reused to collapse the skip
connections. The Attention U-Net encoder, attention gates and decoder are
untouched.

Missing neighbours. On this corpus only ~54% of scans have both +/-1 in-night
neighbours, ~29% have one and ~17% have none, so absent frames are the common
case rather than an edge case. Two things happen for an absent slot:

  * its tensor content is a **copy of the centre frame**, so the shared encoder's
    BatchNorm never sees an all-zero image and its running statistics stay
    representative; and
  * it is flagged in ``pad_mask`` and removed from the attention softmax, so it
    contributes nothing to the aggregate.

A scan with no neighbours at all therefore degrades exactly to the single-frame
model rather than to a model fed two blank images.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTAE2d(nn.Module):
    """Lightweight temporal attention over a (B, T, C, H, W) feature sequence.

    Attention is computed independently per spatial location. Each head owns a
    *learned* master query (rather than one projected from the input, as in the
    original TAE) -- the parameter saving that makes L-TAE lightweight -- and the
    d_model channels are split across heads so each head aggregates its own
    channel group.

    Returns ``(aggregated, attention)`` with shapes ``(B, C, H, W)`` and
    ``(B, n_head, T, H, W)``. The attention is returned so the decoder can reuse
    the same temporal weighting on the higher-resolution skips.
    """

    def __init__(self, in_channels: int, n_head: int = 8, d_k: int = 8):
        super().__init__()
        if in_channels % n_head != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by n_head ({n_head})")
        self.in_channels = in_channels
        self.n_head = n_head
        self.d_k = d_k
        self.d_head = in_channels // n_head

        # One master query per head, learned rather than projected (L-TAE).
        self.query = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.query, std=1.0 / math.sqrt(d_k))
        self.key = nn.Linear(in_channels, n_head * d_k)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=in_channels)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        b, t, c, h, w = x.shape

        # GroupNorm per frame: normalizing each frame independently keeps a
        # replicated padding frame from shifting the statistics of real ones.
        normed = self.in_norm(x.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)

        # (B,T,C,H,W) -> (B*H*W, T, C): one independent sequence per pixel.
        seq = normed.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)

        k = self.key(seq).view(b * h * w, t, self.n_head, self.d_k)
        # Broadcast the master query across batch/time and reduce over d_k.
        scores = (k * self.query.view(1, 1, self.n_head, self.d_k)).sum(-1)
        scores = scores / math.sqrt(self.d_k)                      # (B*H*W, T, n_head)

        if pad_mask is not None:
            # pad_mask: (B,T) True = padded. Expand to every pixel of that frame.
            m = pad_mask.view(b, 1, 1, t).expand(b, h, w, t).reshape(b * h * w, t, 1)
            scores = scores.masked_fill(m, float("-inf"))

        attn = torch.softmax(scores, dim=1)                        # over T

        # Split the channels into per-head groups and weight each by its head.
        v = seq.view(b * h * w, t, self.n_head, self.d_head)
        out = (attn.unsqueeze(-1) * v).sum(dim=1)                  # (B*H*W, n_head, d_head)
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        out = self.out_norm(out)

        attention = attn.view(b, h, w, t, self.n_head).permute(0, 4, 3, 1, 2).contiguous()
        return out, attention


class TemporalAggregator(nn.Module):
    """Collapse a (B,T,C,H,W) skip connection using L-TAE attention masks.

    The masks come from the bottleneck, so they are bilinearly resampled to the
    skip's resolution first. Channels are chunked into ``n_head`` groups and each
    group is weighted by its own head's mask -- U-TAE's ``att_group`` mode -- so
    different heads can attend to different frames for different features.
    """

    def __init__(self, n_head: int = 8):
        super().__init__()
        self.n_head = n_head

    def forward(self, x: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        n_head = attention.shape[1]
        if c % n_head != 0:
            raise ValueError(f"skip channels ({c}) must be divisible by n_head ({n_head})")

        attn = attention
        if attn.shape[-2:] != (h, w):
            # (B,n_head,T,h0,w0) -> fold heads+time into batch for interpolation.
            attn = F.interpolate(
                attn.reshape(b, n_head * t, *attn.shape[-2:]),
                size=(h, w), mode="bilinear", align_corners=False,
            ).view(b, n_head, t, h, w)

        # (B,T,C,H,W) -> (B,n_head,T,C/n_head,H,W)
        grouped = x.view(b, t, n_head, c // n_head, h, w).permute(0, 2, 1, 3, 4, 5)
        weighted = (grouped * attn.unsqueeze(3)).sum(dim=2)        # sum over T
        return weighted.reshape(b, c, h, w)
