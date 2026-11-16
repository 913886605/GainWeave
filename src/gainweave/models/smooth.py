"""The bounded, RGB-shared C_SMOOTH highlight correction."""
from __future__ import annotations

from contextlib import AbstractContextManager
import torch
from torch import nn

HEAD_PARAMETERS = 2_769


def luminance(rgb):
    return rgb[:, :1] * .2126 + rgb[:, 1:2] * .7152 + rgb[:, 2:3] * .0722


def highlight_weight(s_linear):
    u = ((luminance(s_linear) - .5) / .4).clamp(0, 1)
    return u.square() * (3 - 2 * u)


class SmoothHighlightHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(19, 16, 3, padding=1, bias=True)
        self.activation = nn.SiLU()
        self.output = nn.Conv2d(16, 1, 1, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values):
        return self.output(self.activation(self.conv(values)))


def build_head():
    head = SmoothHighlightHead().float()
    if sum(p.numel() for p in head.parameters()) != HEAD_PARAMETERS:
        raise RuntimeError("Incorrect SmoothHead architecture")
    return head


def apply_head(head, features, s_linear, anchor):
    if features.dtype != torch.float32 or s_linear.dtype != torch.float32 or anchor.dtype != torch.float32:
        raise ValueError("HDR recovery requires float32 features, RGB and anchor")
    if s_linear.shape != anchor.shape or features.shape != (s_linear.shape[0], 16, *s_linear.shape[-2:]):
        raise ValueError("SmoothHead feature and RGB geometry mismatch")
    normalized = features / torch.sqrt(features.square().mean(dim=1, keepdim=True) + 1e-6)
    z = head(torch.cat((normalized, s_linear), dim=1))
    w = highlight_weight(s_linear)
    delta = .5 * w * torch.tanh(z)
    gain = torch.exp2(delta)
    active = luminance(s_linear) > .5
    hdr = torch.where(active.expand_as(anchor), anchor * gain, anchor)
    if not bool(torch.isfinite(hdr).all()) or not bool((hdr >= 0).all()):
        raise ValueError("HDR recovery produced nonfinite or negative values")
    return dict(H0=anchor, H_new=hdr, reconstruction=hdr, delta=delta, w=w, z=z)


class HeadStemCapture(AbstractContextManager):
    def __init__(self, bridge, shape):
        self.bridge, self.shape = bridge, shape
        self.value = None
        self.firings = 0
        self.handle = None

    def __enter__(self):
        def capture(_module, _args, value):
            self.firings += 1
            if self.firings != 1 or value.shape != (self.shape[0], 16, *self.shape[-2:]):
                raise ValueError("Incompatible R3 head_stem feature map")
            self.value = value.detach().clone()
        self.handle = self.bridge.head_stem.register_forward_hook(capture)
        return self

    def __exit__(self, kind, value, traceback):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if kind is None and (self.firings != 1 or self.value is None):
            raise RuntimeError("R3 head_stem did not fire exactly once")
        return False
