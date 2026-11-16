"""The original R3-B bridge and its five frozen decoder feature sites."""
from __future__ import annotations

from contextlib import AbstractContextManager
import torch
from torch import nn
from torch.nn import functional as F

HOOKS = (
    "decoder.mid_block", "decoder.up_blocks.0", "decoder.up_blocks.1",
    "decoder.up_blocks.2", "decoder.up_blocks.3",
)
HOOK_CHANNELS = (512, 512, 512, 256, 128)
BRIDGE_PARAMETERS = 42_788


class Block(nn.Module):
    def __init__(self, in_channels: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel, padding=1 if kernel == 3 else 0, bias=True)
        self.norm = nn.GroupNorm(4, 16)
        self.act = nn.SiLU()

    def forward(self, value):
        return self.act(self.norm(self.conv(value)))


class R3Bridge(nn.Module):
    """42,788-parameter R3-B; state keys match accepted B/update8192."""
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([Block(channels, 1) for channels in HOOK_CHANNELS])
        self.fusions = nn.ModuleList([Block(16, 3) for _ in range(4)])
        self.head_stem = Block(16, 3)
        self.output = nn.Conv2d(16, 4, 1, bias=True)
        if sum(p.numel() for p in self.parameters()) != BRIDGE_PARAMETERS:
            raise RuntimeError("Incorrect R3-B architecture")

    def fuse(self, laterals):
        if len(laterals) != 5:
            raise ValueError("R3-B requires five decoder feature maps")
        current = laterals[0]
        for index, lateral in enumerate(laterals[1:]):
            if current.shape[-2:] != lateral.shape[-2:]:
                current = F.interpolate(current, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            current = self.fusions[index](current + lateral)
        return current

    def forward_laterals(self, laterals, s_linear):
        raw = self.output(self.head_stem(self.fuse(laterals)))
        if raw.shape[-2:] != s_linear.shape[-2:]:
            raise ValueError("Decoder features and linear RGB have different resolutions")
        gate = torch.sigmoid(raw[:, :1])
        gain = F.softplus(raw[:, 1:4])
        reconstruction = s_linear * torch.exp2(gate * gain)
        return dict(raw=raw, gate=gate, gain=gain, reconstruction=reconstruction)


def _resolve(root, name):
    for part in name.split("."):
        root = root[int(part)] if part.isdecimal() else getattr(root, part)
    return root


class DecoderFeatureCapture(AbstractContextManager):
    """Apply each bridge adapter at the original decoder hook time."""
    def __init__(self, vae, bridge, shape):
        self.vae, self.bridge, self.shape = vae, bridge, shape
        self.handles = []
        self.laterals = [None] * 5
        self.firings = [0] * 5

    def __enter__(self):
        n, _, h, w = self.shape
        sizes = ((h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w), (h, w))
        try:
            for i, name in enumerate(HOOKS):
                def capture(_module, _args, value, *, index=i):
                    self.firings[index] += 1
                    if self.firings[index] != 1:
                        raise RuntimeError("A decoder feature hook fired more than once")
                    if value.shape != (n, HOOK_CHANNELS[index], *sizes[index]) or value.dtype != torch.float32:
                        raise ValueError("Incompatible VAE decoder feature geometry or dtype")
                    self.laterals[index] = self.bridge.adapters[index](value.detach())
                self.handles.append(_resolve(self.vae, name).register_forward_hook(capture))
        except BaseException:
            self._remove()
            raise
        return self

    def _remove(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def __exit__(self, kind, value, traceback):
        self._remove()
        if kind is None and (self.firings != [1] * 5 or any(v is None for v in self.laterals)):
            raise RuntimeError("The VAE did not produce all five decoder features")
        return False


def build_bridge():
    return R3Bridge()
