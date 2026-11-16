from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gainweave.models import R3Bridge, build_head, apply_head, highlight_weight
from gainweave.pipeline import GainWeave

torch.set_num_threads(1)


class Up(nn.Module):
    def __init__(self, ci, co, up=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 1)
        self.up = up
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest") if self.up else x)


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(shift_factor=.1159, scaling_factor=.3611)
        self.decoder = nn.Module()
        self.decoder.mid_block = nn.Conv2d(16, 512, 1)
        self.decoder.up_blocks = nn.ModuleList([Up(512, 512), Up(512, 512), Up(512, 256), Up(256, 128, False)])
    def encode(self, image):
        value = F.avg_pool2d(image.mean(1, keepdim=True), 8).repeat(1, 16, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))
    def decode(self, z, return_dict=False):
        x = self.decoder.mid_block(z)
        for block in self.decoder.up_blocks:
            x = block(x)
        return (x,)


def test_parameter_keys_and_zero_identity():
    bridge, head = R3Bridge(), build_head()
    assert len(bridge.state_dict()) == 42
    assert set(head.state_dict()) == {"conv.weight", "conv.bias", "output.weight", "output.bias"}
    assert sum(p.numel() for p in bridge.parameters()) == 42788
    assert sum(p.numel() for p in head.parameters()) == 2769
    base = torch.rand(1, 3, 8, 8)
    anchor = base * 3
    feature = torch.randn(1, 16, 8, 8)
    result = apply_head(head, feature, base, anchor)
    assert torch.equal(result["H_new"], anchor)
    result["H_new"].sum().backward()
    assert head.output.weight.grad.abs().sum() > 0
    assert head.conv.weight.grad.count_nonzero() == 0


def test_budget_shared_rgb_and_low_copy():
    head = build_head()
    with torch.no_grad():
        head.output.bias.fill_(20)
    base = torch.tensor([.25, .5, .7, 1.]).view(1, 1, 1, 4).expand(1, 3, 8, 4).clone()
    anchor = torch.rand_like(base) + .1
    result = apply_head(head, torch.zeros(1, 16, 8, 4), base, anchor)
    assert (result["delta"].abs() <= .5 * result["w"]).all()
    assert torch.equal(result["H_new"][..., :2], anchor[..., :2])
    ratio = result["H_new"] / anchor
    torch.testing.assert_close(ratio[:, 0], ratio[:, 1])
    assert highlight_weight(torch.full((1, 3, 2, 2), .5)).count_nonzero() == 0


def test_frozen_forward_hooks_training_and_padding():
    import numpy as np
    model = GainWeave(ToyVAE(), R3Bridge(), build_head())
    model.head.requires_grad_(True)
    model.train()
    base = torch.full((1, 3, 16, 16), .8)
    result = model.forward_tensors(torch.full_like(base, .9), base)
    result["reconstruction"].sum().backward()
    assert model.head.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vae.parameters())
    assert all(p.grad is None for p in model.bridge.parameters())
    assert not model.vae.training and not model.bridge.training
    assert all(not m._forward_hooks for m in model.modules())
    recovered = model.recover(np.full((17, 19, 3), 200, np.uint8))
    assert recovered["hdr_linear_rgb"].shape == (17, 19, 3)
    assert recovered["weight"] is recovered["w"]
    assert recovered["metadata"]["padded_size"] == [24, 24]


def test_hooks_removed_when_decoder_raises():
    class Broken(nn.Module):
        def forward(self, x):
            raise RuntimeError("synthetic decoder failure")
    model = GainWeave(ToyVAE(), R3Bridge(), build_head())
    model.vae.decoder.up_blocks[1] = Broken()
    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        model.forward_tensors(torch.ones(1, 3, 16, 16), torch.ones(1, 3, 16, 16))
    assert all(not m._forward_hooks for m in model.modules())


def test_frozen_hooks_and_trainable_head():
    model = GainWeave(ToyVAE(), R3Bridge(), build_head())
    model.head.requires_grad_(True)
    s = torch.rand(1, 3, 16, 16)
    result = model.forward_tensors(s, s)
    assert torch.equal(result["H_new"], result["H0"])
    result["H_new"].sum().backward()
    assert model.head.output.bias.grad is not None
    assert all(p.grad is None for p in model.vae.parameters())
    assert all(p.grad is None for p in model.bridge.parameters())
    assert all(not m._forward_hooks for m in model.vae.modules())
    assert not model.bridge.head_stem._forward_hooks


def test_hooks_removed_on_exception():
    model = GainWeave(ToyVAE(), R3Bridge(), build_head())
    def fail(_module, _args):
        raise RuntimeError("synthetic decoder failure")
    handle = model.vae.decoder.up_blocks[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        model.forward_tensors(torch.ones(1, 3, 16, 16), torch.ones(1, 3, 16, 16))
    handle.remove()
    assert all(not m._forward_hooks for m in model.vae.modules())
    assert not model.bridge.head_stem._forward_hooks


def test_recover_padding_and_native_ev():
    import numpy as np
    model = GainWeave(ToyVAE(), R3Bridge(), build_head())
    result = model.recover(np.full((13, 19, 3), 180, np.uint8))
    assert result["hdr_linear_rgb"].shape == (13, 19, 3)
    assert result["weight"] is result["w"]
    assert result["metadata"]["padded_size"] == [16, 24]
    assert result["metadata"]["native_exposure_ev"] == 0
    np.testing.assert_array_equal(result["hdr_linear_rgb"], result["anchor"])
