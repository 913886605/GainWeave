"""Portable SDR-to-relative-HDR recovery with the released frozen components."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import numpy as np

# Configure cuBLAS before any caller in this module initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

from . import io
from .hub import DEFAULT_MODEL, DEFAULT_REVISION, DEFAULT_BASE_MODEL, DEFAULT_BASE_REVISION, download_assets, file_sha256
from .hub import DEFAULT_VAE_CONFIG_SHA256, DEFAULT_VAE_WEIGHTS_SHA256
from .models import DecoderFeatureCapture, HeadStemCapture, R3Bridge, apply_head, build_head


def configure_recovery_precision():
    """Match the original FP32 recovery arithmetic and deterministic settings."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class GainWeave(torch.nn.Module):
    def __init__(self, vae, bridge, head, config=None):
        super().__init__()
        self.vae = vae.float().eval().requires_grad_(False)
        self.bridge = bridge.float().eval().requires_grad_(False)
        self.head = head.float().eval().requires_grad_(False)
        self.config = {} if config is None else dict(config)

    @property
    def device(self):
        return next(self.bridge.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        self.bridge.eval()
        return self

    @classmethod
    def from_pretrained(cls, model=DEFAULT_MODEL, *, revision=DEFAULT_REVISION, device="cuda",
                        cache_dir=None, local_files_only=False, vae_source=None):
        configure_recovery_precision()
        assets = download_assets(task="recover", model=model, revision=revision, cache_dir=cache_dir,
                                 local_files_only=local_files_only, vae_source=vae_source)
        from safetensors.torch import load_file
        from diffusers import AutoencoderKL

        config = assets["config"]
        bridge, head = R3Bridge().float(), build_head()
        for module, field, count in ((bridge, "r3_weights", 42_788), (head, "smooth_weights", 2_769)):
            state = load_file(str(assets["model_dir"] / config[field]), device="cpu")
            if sum(v.numel() for v in state.values()) != count or any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
                raise ValueError(f"Invalid float32 parameter state: {config[field]}")
            module.load_state_dict(state, strict=True)
        vae = AutoencoderKL.from_pretrained(str(assets["vae_dir"]), local_files_only=True,
                         torch_dtype=torch.float32, use_safetensors=True, low_cpu_mem_usage=False)
        vae.disable_tiling()
        vae.disable_slicing()
        result = cls(vae, bridge, head, config).to(device).eval()
        result.source_revision = revision
        result.model_source = str(model)
        return result

    def save_pretrained(self, output_dir):
        """Save only the two custom components and their public configuration."""
        from safetensors.torch import save_file
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        names = ("bridge.safetensors", "highlight.safetensors")
        for module, name, count in ((self.bridge, names[0], 42_788), (self.head, names[1], 2_769)):
            state = {key: value.detach().cpu().contiguous().clone() for key, value in module.state_dict().items()}
            if sum(t.numel() for t in state.values()) != count or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()):
                raise ValueError("Only finite float32 GainWeave component states can be exported")
            save_file(state, str(target / name))
        # Deliberately enumerate public fields; training paths, VAE tensors,
        # optimizer, RNG, data sequence and audit metadata are never serialized.
        config = dict(schema_version="gainweave-inference-v1", version="0.1.0",
            architecture="R3BWithSmoothHighlight", r3_weights=names[0], smooth_weights=names[1],
            sha256={name: file_sha256(target / name) for name in names},
            base_model=self.config.get("base_model", DEFAULT_BASE_MODEL),
            base_revision=self.config.get("base_revision", DEFAULT_BASE_REVISION),
            vae_config_sha256=self.config.get("vae_config_sha256", DEFAULT_VAE_CONFIG_SHA256),
            vae_weights_sha256=self.config.get("vae_weights_sha256", DEFAULT_VAE_WEIGHTS_SHA256),
            parameters=dict(bridge=42_788, highlight=2_769, total=45_557), weights_dtype="float32",
            recovery_dtype="float32", highlight=dict(low=.5, high=.9, max_residual_ev=.5, fixed_residual_ev=.25),
            default_presentation_ev=1)
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return target

    def forward_tensors(self, s_srgb, S):
        """NCHW float32 -> original anchor plus the bounded learned correction.

        Frozen outputs are ordinary detached tensors. Gradients can flow through
        the head after a trainer enables `head.requires_grad_(True)`.
        """
        if s_srgb.ndim != 4 or s_srgb.shape != S.shape or S.shape[1] != 3 or S.dtype != torch.float32 or s_srgb.dtype != torch.float32:
            raise ValueError("Expected matching NCHW float32 encoded sRGB and linear RGB")
        if min(S.shape[-2:]) < 8 or any(n % 8 for n in S.shape[-2:]):
            raise ValueError("Tensor input dimensions must be at least 8 and divisible by 8")
        if any(not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any() for t in (s_srgb, S)):
            raise ValueError("Encoded sRGB and linear RGB must be finite in [0, 1]")
        if s_srgb.device != self.device or S.device != self.device:
            raise ValueError("Inputs and GainWeave must be on the same device")
        with torch.inference_mode(False), torch.no_grad():
            image, base = s_srgb.detach().clone().contiguous(), S.detach().clone().contiguous()
            raw = self.vae.encode(image * 2 - 1).latent_dist.mode()
            diffusion = (raw - float(self.vae.config.shift_factor)) * float(self.vae.config.scaling_factor)
            canonical = (diffusion / float(self.vae.config.scaling_factor) + float(self.vae.config.shift_factor)).float().contiguous()
            with HeadStemCapture(self.bridge, base.shape) as stem:
                with DecoderFeatureCapture(self.vae, self.bridge, base.shape) as capture:
                    self.vae.decode(canonical, return_dict=False)[0]
                anchor = self.bridge.forward_laterals(capture.laterals, base)["reconstruction"].detach().clone()
            features = stem.value.detach().clone()
        # Exiting inference mode also matters when the caller uses it globally:
        # trainable convolutions must receive ordinary tensors they can save.
        with torch.inference_mode(False):
            return apply_head(self.head, features, base, anchor)

    def forward(self, s_srgb, S):
        return self.forward_tensors(s_srgb, S)

    def recover(self, image, *, variant="gainweave"):
        """Recover one RGB8 input. No exposure, peak normalization or resizing.

        Nonmultiples of eight receive reflection padding on the bottom/right;
        all returned arrays are cropped back to the original dimensions.
        """
        if variant not in ("gainweave", "anchor", "fixed"):
            raise ValueError("variant must be 'gainweave', 'anchor' or 'fixed'")
        rgb = io.load_srgb(image)
        height, width = rgb.shape[:2]
        srgb = rgb.astype(np.float32) / np.float32(255.)
        linear = io.srgb_to_linear(srgb)
        s = torch.from_numpy(srgb.transpose(2, 0, 1).copy()[None]).to(self.device)
        base = torch.from_numpy(linear.transpose(2, 0, 1).copy()[None]).to(self.device)
        pad_h, pad_w = (-height) % 8, (-width) % 8
        if pad_h or pad_w:
            s = F.pad(s, (0, pad_w, 0, pad_h), mode="reflect")
            base = F.pad(base, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            values = self.forward_tensors(s, base)
            if variant == "anchor":
                hdr, delta = values["H0"], torch.zeros_like(values["delta"])
            elif variant == "fixed":
                delta = .25 * values["w"]
                hdr = values["H0"] * torch.exp2(delta)
            else:
                hdr, delta = values["H_new"], values["delta"]
        def rgb_array(t):
            return np.ascontiguousarray(t[0, :, :height, :width].detach().cpu().numpy().transpose(1, 2, 0), dtype=np.float32)
        def scalar_array(t):
            return np.ascontiguousarray(t[0, 0, :height, :width].detach().cpu().numpy(), dtype=np.float32)
        hdr, anchor, delta, weight = rgb_array(hdr), rgb_array(values["H0"]), scalar_array(delta), scalar_array(values["w"])
        return dict(hdr_linear_rgb=hdr, hdr=hdr, anchor=anchor, delta_ev=delta, weight=weight, w=weight,
            metadata=dict(variant=variant, original_size=[height, width], padded_size=[height + pad_h, width + pad_w],
                          padding="reflect_bottom_right", native_exposure_ev=0, native_units="relative_linear_RGB",
                          model_source=getattr(self, "model_source", None),
                          model_revision=getattr(self, "source_revision", None),
                          source_weights_sha256=dict(self.config.get("sha256", {})),
                          base_model=self.config.get("base_model", DEFAULT_BASE_MODEL),
                          base_revision=self.config.get("base_revision", DEFAULT_BASE_REVISION),
                          physical_nits=None, recovery_dtype="float32", tested_size=[1024, 1024],
                          nonstandard_size=height != 1024 or width != 1024,
                          weights_sha256=dict(self.config.get("sha256", {})),
                          vae_weights_sha256=self.config.get("vae_weights_sha256", DEFAULT_VAE_WEIGHTS_SHA256)))

    def close(self):
        """Release this model's device tensors without changing cached weights."""
        self.to("cpu")
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
