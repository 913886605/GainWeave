import hashlib
import json
import sys
from types import SimpleNamespace
import pytest

from gainweave import hub
from gainweave.pipeline import GainWeave
from gainweave.models import R3Bridge, build_head
from test_models import ToyVAE


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture_assets(tmp_path, monkeypatch):
    model_dir, base_dir = tmp_path / "model", tmp_path / "base"
    model_dir.mkdir(); (base_dir / "vae").mkdir(parents=True)
    cfg_raw, weights_raw = b"{}", b"test-only-VAE-fixture"
    (base_dir / "vae" / "config.json").write_bytes(cfg_raw)
    (base_dir / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(weights_raw)
    monkeypatch.setattr(hub, "DEFAULT_VAE_CONFIG_SHA256", digest(cfg_raw))
    monkeypatch.setattr(hub, "DEFAULT_VAE_WEIGHTS_SHA256", digest(weights_raw))
    model = GainWeave(ToyVAE(), R3Bridge(), build_head(), config={"private_path": "/secret/training"})
    model.save_pretrained(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    cfg.update(vae_config_sha256=digest(cfg_raw), vae_weights_sha256=digest(weights_raw))
    (model_dir / "config.json").write_text(json.dumps(cfg))
    calls = []
    def snapshot(repo_id, revision, patterns, cache_dir, local_files_only):
        calls.append((repo_id, revision, patterns, local_files_only))
        return model_dir if repo_id == "test/model" else base_dir
    monkeypatch.setattr(hub, "_snapshot", snapshot)
    return model, model_dir, base_dir, calls


def test_minimal_recover_and_complete_generate_download(tmp_path, monkeypatch):
    _, model_dir, base_dir, calls = fixture_assets(tmp_path, monkeypatch)
    assets = hub.download_assets("recover", model="test/model", revision="fixed", local_files_only=True)
    assert assets["model_dir"] == model_dir
    assert assets["vae_dir"] == base_dir / "vae"
    assert calls[-1][2] == ["vae/*"] and calls[-1][3]
    hub.download_assets("generate", model="test/model", revision="fixed")
    assert calls[-1][2] is None
    assert "private_path" not in json.loads((model_dir / "config.json").read_text())
    assert set(p.name for p in model_dir.iterdir()) == {"config.json", "bridge.safetensors", "highlight.safetensors"}


def test_corruption_fails_and_local_vae_avoids_base_download(tmp_path, monkeypatch):
    _, model_dir, base_dir, calls = fixture_assets(tmp_path, monkeypatch)
    hub.download_assets(model=str(model_dir), vae_source=base_dir / "vae")
    assert calls == []
    (model_dir / "highlight.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        hub.download_assets(model=str(model_dir), vae_source=base_dir / "vae")


def test_from_pretrained_state_roundtrip(tmp_path, monkeypatch):
    import torch
    original, model_dir, base_dir, _ = fixture_assets(tmp_path, monkeypatch)
    class Autoencoder:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            vae = ToyVAE()
            vae.disable_tiling = lambda: None
            vae.disable_slicing = lambda: None
            return vae
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(AutoencoderKL=Autoencoder))
    model = GainWeave.from_pretrained(str(model_dir), device="cpu", vae_source=base_dir / "vae")
    assert all(torch.equal(value, model.bridge.state_dict()[key]) for key, value in original.bridge.state_dict().items())
    assert all(torch.equal(value, model.head.state_dict()[key]) for key, value in original.head.state_dict().items())
    assert all(not p.requires_grad for p in model.parameters())
