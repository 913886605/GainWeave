import types

import numpy as np
import pytest
import torch
from torch import nn

from gainweave.models.smooth import build_head
from gainweave.paper_controls import PaperControl, _build_bridge


def test_rgb_source_and_padding():
    bridge = _build_bridge("rgb")
    assert sum(p.numel() for p in bridge.parameters()) == 42_852
    model = PaperControl("rgb", bridge, build_head())
    rgb = np.random.default_rng(19).integers(0, 256, size=(17, 27, 3), dtype=np.uint8)
    result = model.recover(rgb)
    assert model.vae is None
    assert result["H0"].shape == rgb.shape
    # The untrained head is initialized as the identity.
    np.testing.assert_array_equal(result["H0"], result["H_new"])


def test_one_feature_and_cleanup_on_decode_failure():
    class ToyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.up_blocks = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.config = types.SimpleNamespace(scaling_factor=1, shift_factor=0)
            self.fail = False

        def encode(self, image):
            zero = torch.zeros((len(image), 128, *image.shape[-2:]))
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: zero))

        def decode(self, latent, return_dict):
            self.decoder.up_blocks[3](latent)
            if self.fail:
                raise RuntimeError("injected")

    vae = ToyVAE()
    bridge = _build_bridge("one")
    assert sum(p.numel() for p in bridge.parameters()) == 42_917
    model = PaperControl("one", bridge, build_head(), vae)
    t = torch.full((1, 3, 16, 16), .7)
    assert model.forward_tensors(t, t)["H_new"].shape == t.shape
    assert not vae.decoder.up_blocks[3]._forward_hooks
    vae.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        model.forward_tensors(t, t)
    assert not vae.decoder.up_blocks[3]._forward_hooks


def test_control_loader_rejects_corrupted_manifest(tmp_path):
    (tmp_path / "controls").mkdir()
    (tmp_path / "controls/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        PaperControl.from_pretrained("rgb", model=tmp_path, local_files_only=True, device="cpu")
