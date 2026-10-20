"""Pure PyTorch StyleGAN operators for UniFace inference.

The filtering sequence follows UniFace training/op/upfirdn2d.py's native
reference (itself attributed to rosinality/stylegan2-pytorch). Bias activation
implements the upstream CUDA operation: sqrt(2) * leaky_relu(x + bias, .2).
No extension compilation, toolchain, downloads, or CUDA-specific imports.
See NOTICE and PROVENANCE.json for upstream attribution.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        input = input + bias.view(1, -1, *([1] * (input.ndim - 2)))
    return F.leaky_relu(input, negative_slope=negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Insert zeros, pad/crop, convolve with flipped FIR, then decimate.

    The zero insertion includes trailing zeros, matching the upstream native
    reference and CUDA op (not interpolate or nearest-neighbor upsampling).
    """
    if input.ndim != 4 or kernel.ndim != 2:
        raise ValueError("upfirdn2d requires NCHW input and a 2D kernel.")
    if (isinstance(up, bool) or isinstance(down, bool)
            or not isinstance(up, int) or not isinstance(down, int)
            or up < 1 or down < 1):
        raise ValueError("Upsampling and downsampling factors must be positive integers.")
    if len(pad) != 2:
        raise ValueError("UniFace filtering expects a two-element padding tuple.")
    batch, channels, height, width = input.shape
    pad0, pad1 = pad
    out = input.reshape(-1, height, 1, width, 1)
    out = F.pad(out, (0, up - 1, 0, 0, 0, up - 1))
    out = out.reshape(-1, 1, height * up, width * up)
    out = F.pad(out, (max(pad0, 0), max(pad1, 0),
                      max(pad0, 0), max(pad1, 0)))
    out = out[:, :, max(-pad0, 0):out.shape[2] - max(-pad1, 0),
              max(-pad0, 0):out.shape[3] - max(-pad1, 0)]
    weight = kernel.to(device=input.device, dtype=input.dtype).flip((0, 1))[None, None]
    out = F.conv2d(out, weight)
    out = out[:, :, ::down, ::down]
    return out.reshape(batch, channels, out.shape[2], out.shape[3])
