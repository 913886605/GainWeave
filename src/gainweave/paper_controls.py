"""Inference with the trained One and RGB-only controls from Tables III and IV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import io
from .hub import (
    DEFAULT_BASE_MODEL, DEFAULT_BASE_REVISION, DEFAULT_MODEL, DEFAULT_VAE_CONFIG_SHA256,
    DEFAULT_VAE_WEIGHTS_SHA256, verify_file,
)
from .models.bridge import R3Bridge
from .models.smooth import HeadStemCapture, apply_head, build_head
from .pipeline import configure_recovery_precision

PAPER_REVISION = "b972bb16640226b2fd910d2e6556b0544ba95849"  # v0.2.0
MANIFEST_SHA256 = "ddcb3caf610a1312869e9f2bfaae8b15ddac3cd4f63967831543356fd3b71a1b"
FRONTENDS = {"one": (128, (30, 30, 30, 15, 8)), "rgb": (3, (55, 55, 55, 28, 15))}


def _get_control_files(model, architecture, revision, cache_dir, local_files_only):
    root = Path(model).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(
            repo_id=str(model), revision=revision, cache_dir=cache_dir, local_files_only=local_files_only,
            allow_patterns=["controls/manifest.json", f"controls/{architecture}/*.safetensors"],
        ))
    manifest = verify_file(root / "controls/manifest.json", MANIFEST_SHA256)
    config = json.loads(manifest.read_text())["architecture"][architecture]
    paths = {key: verify_file(root / config[key]["file"], config[key]["sha256"])
             for key in ("bridge", "highlight")}
    return paths


def _load_vae(vae_source, cache_dir, local_files_only):
    from diffusers import AutoencoderKL
    from huggingface_hub import snapshot_download

    if vae_source is None:
        root = Path(snapshot_download(
            DEFAULT_BASE_MODEL, revision=DEFAULT_BASE_REVISION, allow_patterns=["vae/*"],
            cache_dir=cache_dir, local_files_only=local_files_only,
        )) / "vae"
    else:
        root = Path(vae_source).expanduser()
        if (root / "vae").is_dir():
            root = root / "vae"
    verify_file(root / "config.json", DEFAULT_VAE_CONFIG_SHA256)
    verify_file(root / "diffusion_pytorch_model.safetensors", DEFAULT_VAE_WEIGHTS_SHA256)
    vae = AutoencoderKL.from_pretrained(
        str(root), torch_dtype=torch.float32, local_files_only=True, use_safetensors=True,
        low_cpu_mem_usage=False,
    )
    vae.disable_tiling()
    vae.disable_slicing()
    return vae


def _build_bridge(architecture):
    channels, widths = FRONTENDS[architecture]
    bridge = R3Bridge()
    bridge.adapters = nn.ModuleList(
        nn.Sequential(nn.Conv2d(channels, width, 1), nn.SiLU(),
                      nn.Conv2d(width, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        for width in widths
    )
    return bridge.float()


class PaperControl(nn.Module):
    """Each endpoint uses its own trained bridge and highlight head."""

    def __init__(self, architecture, bridge, head, vae=None):
        super().__init__()
        if architecture not in FRONTENDS or (architecture == "one") != (vae is not None):
            raise ValueError("One requires a VAE; RGB-only takes linear RGB directly.")
        self.architecture, self.bridge, self.head, self.vae = architecture, bridge, head, vae

    @classmethod
    def from_pretrained(cls, architecture, model=DEFAULT_MODEL, *, revision=PAPER_REVISION,
                        device="cuda", cache_dir=None, local_files_only=False, vae_source=None):
        from safetensors.torch import load_file

        architecture = architecture.lower()
        if architecture not in FRONTENDS:
            raise ValueError("architecture must be 'one' or 'rgb'")
        configure_recovery_precision()
        paths = _get_control_files(model, architecture, revision, cache_dir, local_files_only)
        bridge, head = _build_bridge(architecture), build_head()
        for module, path in ((bridge, paths["bridge"]), (head, paths["highlight"])):
            module.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        vae = _load_vae(vae_source, cache_dir, local_files_only) if architecture == "one" else None
        return cls(architecture, bridge, head, vae).eval().requires_grad_(False).to(device)

    def _decoder_feature(self, encoded):
        latent = self.vae.encode(encoded * 2 - 1).latent_dist.mode()
        scale, shift = float(self.vae.config.scaling_factor), float(self.vae.config.shift_factor)
        canonical = (((latent - shift) * scale) / scale + shift).float().contiguous()
        features = []
        handle = self.vae.decoder.up_blocks[3].register_forward_hook(
            lambda _module, _args, output: features.append(output.detach())
        )
        try:
            self.vae.decode(canonical, return_dict=False)
        finally:
            handle.remove()
        if len(features) != 1 or features[0].shape != (encoded.shape[0], 128, *encoded.shape[-2:]):
            raise RuntimeError("Expected one full-resolution 128-channel decoder feature map.")
        return features[0]

    @torch.no_grad()
    def forward_tensors(self, s_srgb, s_linear):
        """Paper inputs: matching contiguous NCHW float32 encoded and linear SDR tensors."""
        if s_linear.ndim != 4 or s_linear.shape[1] != 3 or s_srgb.shape != s_linear.shape:
            raise ValueError("Expected matching NCHW RGB tensors.")
        height, width = s_linear.shape[-2:]
        if min(height, width) < 8 or height % 8 or width % 8:
            raise ValueError("Spatial dimensions must be positive multiples of eight.")
        device = next(self.bridge.parameters()).device
        for t in (s_srgb, s_linear):
            if (t.device != device or t.dtype != torch.float32 or not torch.isfinite(t).all()
                    or torch.any(t < 0) or torch.any(t > 1)):
                raise ValueError("Expected finite FP32 inputs in [0,1] on the model device.")
        # Layout matters for exact agreement with the reference FP32 convolution path.
        s_linear = s_linear.contiguous()
        feature = self._decoder_feature(s_srgb.contiguous()) if self.architecture == "one" else s_linear
        laterals = []
        for adapter, divisor in zip(self.bridge.adapters, (8, 4, 2, 1, 1), strict=True):
            value = F.interpolate(feature, size=(height // divisor, width // divisor), mode="area") if divisor > 1 else feature
            laterals.append(adapter(value))
        with HeadStemCapture(self.bridge, s_linear.shape) as stem:
            anchor = self.bridge.forward_laterals(laterals, s_linear)["reconstruction"]
        return apply_head(self.head, stem.value, s_linear, anchor)

    forward = forward_tensors

    def recover(self, image):
        """Return the native linear HDR outputs of both stages for one RGB8 image."""
        encoded = io.load_srgb(image).astype(np.float32) / np.float32(255.)
        linear = io.srgb_to_linear(encoded)
        height, width = encoded.shape[:2]
        def tensor(value):
            chw = np.ascontiguousarray(value.transpose(2, 0, 1)[None])
            result = torch.from_numpy(chw).to(next(self.bridge.parameters()).device)
            if height % 8 or width % 8:
                result = F.pad(result, (0, -width % 8, 0, -height % 8), mode="reflect")
            return result
        outputs = self.forward_tensors(tensor(encoded), tensor(linear))
        return {key: outputs[key][0, :, :height, :width].permute(1, 2, 0).cpu().contiguous().numpy()
                for key in ("H0", "H_new")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(FRONTENDS), required=True)
    parser.add_argument("--input", required=True, type=Path, help="RGB8 image")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub repo ID or local snapshot root")
    parser.add_argument("--revision", default=PAPER_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--vae-source", type=Path, help="Optional local VAE for One")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists.")
    model = PaperControl.from_pretrained(
        args.architecture, args.model, revision=args.revision, device=args.device,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only, vae_source=args.vae_source,
    )
    outputs = model.recover(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    io.write_exr(args.output / "anchor.exr", outputs["H0"])
    io.write_exr(args.output / "native.exr", outputs["H_new"])


if __name__ == "__main__":
    main()
