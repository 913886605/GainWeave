"""Download only the assets required by each public inference task."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

DEFAULT_MODEL = "Noel2211/GainWeave"
DEFAULT_REVISION = "v0.1.0"
DEFAULT_BASE_MODEL = "Alpha-VLLM/Lumina-Image-2.0"
DEFAULT_BASE_REVISION = "53504abd8178b30685b6c4c7a4cd181ff78b73e9"
DEFAULT_VAE_CONFIG_SHA256 = "80c5ed836c7171f6817a8addb910cdd32d30ca65549ba13b75e097c0c30fb326"
DEFAULT_VAE_WEIGHTS_SHA256 = "8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94"


def _filename(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError("Asset names must be relative paths without parent traversal")
    return name


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("A lowercase SHA256 identity is required for every model asset")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required model asset is missing: {path}")
    if file_sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}; obtain the matching published assets")
    return path


def read_config(model_dir):
    config = json.loads((Path(model_dir) / "config.json").read_text())
    if config.get("schema_version") not in ("gainweave-v1", "gainweave-inference-v1"):
        raise ValueError("Unsupported GainWeave config schema")
    if config.get("architecture") != "R3BWithSmoothHighlight":
        raise ValueError("This checkpoint does not describe the GainWeave R3-B + SmoothHead architecture")
    for field in ("r3_weights", "smooth_weights"):
        _filename(config[field])
        if config[field] not in config.get("sha256", {}):
            raise ValueError(f"config.json has no SHA256 identity for {config[field]}")
    if config.get("weights_dtype") != "float32" or config.get("recovery_dtype") != "float32":
        raise ValueError("This release requires float32 recovery weights and computation")
    if config.get("vae_config_sha256") != DEFAULT_VAE_CONFIG_SHA256 or config.get("vae_weights_sha256") != DEFAULT_VAE_WEIGHTS_SHA256:
        raise ValueError("This release requires the original Lumina FP32 recovery VAE")
    highlight = config.get("highlight", {})
    required = dict(low=.5, high=.9, max_residual_ev=.5, fixed_residual_ev=.25)
    if any(highlight.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported highlight thresholds or residual EV budget")
    return config


def _snapshot(repo_id, revision, patterns, cache_dir, local_files_only):
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=patterns,
                    cache_dir=cache_dir, local_files_only=local_files_only))
    except Exception as error:
        # Do not echo exception text: URLs can contain credentials or request details.
        raise RuntimeError(
            f"Cannot obtain {repo_id}@{revision}. Check the model/revision, network or local cache. "
            "If access is restricted, sign in with `hf auth login` and accept the upstream model terms."
        ) from error


def download_assets(task="recover", model=DEFAULT_MODEL, revision=DEFAULT_REVISION, cache_dir=None,
                    local_files_only=False, *, model_id=None, vae_source=None):
    """Return model_dir, weights_dir, vae_dir and (for generate) base_dir.

    `recover` requests only the upstream `vae/` subtree. A local VAE directory
    can be supplied through `vae_source`, without downloading the full base.
    """
    if task not in ("recover", "generate"):
        raise ValueError("task must be 'recover' or 'generate'")
    model = model_id if model_id is not None else model
    local = Path(model).expanduser()
    if local.is_dir():
        model_dir = local.resolve()
    else:
        model_dir = _snapshot(str(model), revision, ["config.json"], cache_dir, local_files_only)
    config = read_config(model_dir)
    names = [config["r3_weights"], config["smooth_weights"]]
    if not local.is_dir():
        model_dir = _snapshot(str(model), revision, ["config.json", *names], cache_dir, local_files_only)
        if read_config(model_dir) != config:
            raise ValueError("Model configuration changed between download requests; use a commit revision")
    for name in names:
        verify_file(model_dir / name, config["sha256"][name])

    base_dir = None
    if vae_source is not None:
        source = Path(vae_source).expanduser()
        if source.is_dir():
            vae_dir = source / "vae" if (source / "vae" / "config.json").is_file() else source
        else:
            source = _snapshot(str(vae_source), config.get("base_revision", DEFAULT_BASE_REVISION), ["vae/*"], cache_dir, local_files_only)
            vae_dir = source / "vae"
    else:
        base_dir = _snapshot(config.get("base_model", DEFAULT_BASE_MODEL), config.get("base_revision", DEFAULT_BASE_REVISION),
                             None if task == "generate" else ["vae/*"], cache_dir, local_files_only)
        vae_dir = base_dir / "vae"
    if task == "generate" and vae_source is not None:
        base_dir = _snapshot(config.get("base_model", DEFAULT_BASE_MODEL), config.get("base_revision", DEFAULT_BASE_REVISION),
                             None, cache_dir, local_files_only)
    verify_file(vae_dir / "config.json", config["vae_config_sha256"])
    verify_file(vae_dir / "diffusion_pytorch_model.safetensors", config["vae_weights_sha256"])
    return dict(model_dir=model_dir, weights_dir=model_dir, vae_dir=vae_dir,
                base_dir=base_dir if task == "generate" else None, config=config)
