"""Portable head-only training with explicit input, output, and restart state."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import (EVS, create_file, file_sha256, json_bytes, load_dataset,
                   load_prepared_sample, prepare_sample)
from .losses import FixedScheduler, LOSS_COEFFICIENTS, optimizer_for, update_once

CHECKPOINT_SCHEMA = "gainweave-training-checkpoint-v1"


def read_config(path, schema):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"Expected {schema} configuration")
    return path, value


def config_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def seed_all(seed, include_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)


def capture_rng(include_cuda=False):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=state[0], keys=state[1].tolist(),
                position=state[2], has_gauss=state[3], cached_gaussian=state[4]),
                torch_cpu=torch.random.get_rng_state().clone(),
                torch_cuda=[item.cpu().clone() for item in torch.cuda.get_rng_state_all()] if include_cuda else [],
                include_cuda=include_cuda)


def restore_rng(state, include_cuda=False):
    if state["include_cuda"] != include_cuda:
        raise ValueError("Resume requires the same CPU/CUDA RNG scope")
    random.setstate(state["python"])
    saved = state["numpy"]
    np.random.set_state((saved["name"], np.asarray(saved["keys"], dtype=np.uint32), saved["position"],
                         saved["has_gauss"], saved["cached_gaussian"]))
    torch.random.set_rng_state(state["torch_cpu"])
    if include_cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def tensor_digest(state):
    """Hash dtype, shape, names and raw tensor bytes, including BF16 tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json_bytes(dict(name=name, dtype=str(value.dtype), shape=list(value.shape))))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_sequence(rows, updates, seed):
    """Uniform source, then uniform view and EV; camera alternates within batch."""
    by_source = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            by_source[row["source_id"]].append(index)
    sources = sorted(by_source)
    if not sources:
        raise ValueError("No training sources in prepared manifest")
    generator = random.Random(seed)
    sequence = []
    for index in range(8 * updates):
        source = generator.choice(sources)
        sequence.append(dict(row=generator.choice(by_source[source]), EV=generator.choice(EVS),
                             camera="hard_clip" if index % 2 == 0 else "soft_compress"))
    return sequence


def save_checkpoint(path, model, optimizer, scheduler, *, bindings, sequence, include_cuda=False):
    """Create an immutable CPU checkpoint and SHA sidecar with a safe load schema."""
    state = dict(schema=CHECKPOINT_SCHEMA, head=_cpu_clone(model.head.state_dict()),
                 optimizer=_cpu_clone(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                 completed_update=scheduler.completed_update, next_sample=8 * scheduler.completed_update,
                 rng=capture_rng(include_cuda), bindings=copy.deepcopy(bindings), sequence=copy.deepcopy(sequence))
    stream = io.BytesIO()
    torch.save(state, stream)
    raw = stream.getvalue()
    create_file(path, raw)
    certificate = dict(schema=CHECKPOINT_SCHEMA, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                       completed_update=scheduler.completed_update, head_sha256=tensor_digest(state["head"]))
    create_file(str(path) + ".json", json_bytes(certificate))
    return certificate


def load_checkpoint(path, model, optimizer, scheduler, *, bindings, sequence, include_cuda=False):
    path = Path(path)
    certificate = json.loads(Path(str(path) + ".json").read_text())
    raw = path.read_bytes()
    if certificate.get("schema") != CHECKPOINT_SCHEMA or certificate["bytes"] != len(raw) or certificate["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Checkpoint SHA256, size, or schema does not match its sidecar")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA or state["bindings"] != bindings or state["sequence"] != sequence:
        raise ValueError("Checkpoint recipe, frozen model, dataset or sample sequence differs")
    if state["completed_update"] != certificate["completed_update"] or state["next_sample"] != 8 * state["completed_update"]:
        raise ValueError("Checkpoint position is invalid")
    if tensor_digest(state["head"]) != certificate["head_sha256"]:
        raise ValueError("Checkpoint head identity differs")
    model.head.load_state_dict(state["head"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scheduler.completed_update != state["completed_update"]:
        raise ValueError("Checkpoint scheduler and model positions disagree")
    restore_rng(state["rng"], include_cuda)
    return state["completed_update"]


def _model_source(base, source):
    if source is None:
        return None
    # Hub repository IDs remain strings. Explicit/existing local paths resolve
    # relative to the config so moving the project does not depend on cwd.
    if str(source).startswith((".", "/", "~")) or (base / source).exists():
        return str(config_path(base, source))
    return source


def export_model(model, staging: Path, destination: Path):
    """Validate a complete staged export before atomically making it final.

    Failed staging directories are retained in their original attempt. Resume
    uses a new attempt and never mistakes a partial export for a finished run.
    """
    if staging.exists() or destination.exists():
        raise FileExistsError("Model export staging/final directory already exists")
    model.save_pretrained(staging)
    config = json.loads((staging / "config.json").read_text())
    for field, expected in (("r3_weights", "bridge.safetensors"), ("smooth_weights", "highlight.safetensors")):
        if config.get(field) != expected:
            raise ValueError(f"Export configuration is missing {field}={expected}")
        weight = staging / expected
        if not weight.is_file() or not weight.stat().st_size:
            raise ValueError(f"Export is missing nonempty {expected}")
        if file_sha256(weight) != config.get("sha256", {}).get(expected):
            raise ValueError(f"Export SHA256 does not match {expected}")
    if destination.exists():
        raise FileExistsError(f"A finished model already exists: {destination}")
    # Both directories are under one output root, so rename is atomic and
    # readers never observe only one component of the final model.
    staging.rename(destination)
    return destination


def run_training(config_path_value, data_root, output_dir=None, device="cuda", updates=None):
    """Train only a fresh smooth head; defaults reproduce the 2048-update recipe.

    ``updates`` (or the config field) may shorten engineering checks; these are
    labelled modified recipes. Dataset sampling is rebuilt for the user's own
    sources and does not claim to reproduce the private paper data sequence.
    Resume is always an explicit checkpoint path, never an inferred restart.
    """
    config_file, config = read_config(config_path_value, "gainweave-train-v1")
    base = config_file.parent
    data_root = Path(data_root).resolve()
    manifest_name = config.get("dataset", "manifest.json")
    _, rows = load_dataset(data_root, manifest_name)
    for row in rows:  # Verify even dev data identities without exposing it to training forwards.
        if row["split"] == "train":
            load_prepared_sample(data_root, row)
    total = int(config.get("updates", 2048) if updates is None else updates)
    interval = int(config.get("checkpoint_every", 128))
    seed = int(config.get("seed", 0))
    if total < 1 or interval < 1:
        raise ValueError("updates and checkpoint_every must be positive")
    target = Path(output_dir).resolve() if output_dir is not None else config_path(base, config["output_dir"])
    resume = config_path(base, config["resume"]) if config.get("resume") else None
    if target == data_root or target.is_relative_to(data_root):
        raise ValueError("Training output must be outside prepared data")
    if target.exists() and any(target.iterdir()) and resume is None:
        raise FileExistsError(f"Training output is not empty: {target}; set resume explicitly")
    if (target / "final").exists():
        raise FileExistsError(f"A finished model already exists: {target / 'final'}")
    use_cuda = torch.device(device).type == "cuda"
    seed_all(seed, use_cuda)
    torch.use_deterministic_algorithms(True)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    from gainweave import GainWeave
    from gainweave.models import build_head

    options = dict(config.get("model", {}))
    pretrained = _model_source(base, options.pop("pretrained", "Noel2211/GainWeave"))
    if "vae_source" in options:
        options["vae_source"] = _model_source(base, options["vae_source"])
    model = GainWeave.from_pretrained(pretrained, device=device, **options)
    model.vae.eval().requires_grad_(False)
    model.bridge.eval().requires_grad_(False)
    # Recreate the entire head: keeping a trained stem and resetting only its
    # last layer would not reproduce fresh initialization.
    seed_all(seed, use_cuda)
    model.head = build_head().float().to(device).train().requires_grad_(True)
    optimizer = optimizer_for(model.head)
    scheduler = FixedScheduler(optimizer, total_updates=total)
    sequence = make_sequence(rows, total, seed)
    foundation = dict(vae=tensor_digest(model.vae.state_dict()), bridge=tensor_digest(model.bridge.state_dict()))
    bindings = dict(manifest_sha256=file_sha256(data_root / manifest_name), foundation=foundation,
                    seed=seed, updates=total, microbatch=1, accumulation=8, loss=LOSS_COEFFICIENTS,
                    recipe="smooth_highlight_v1", dtype="float32")
    completed = load_checkpoint(resume, model, optimizer, scheduler, bindings=bindings,
                                sequence=sequence, include_cuda=use_cuda) if resume else 0
    # Failed dependency/model admission leaves no output run to block a retry.
    # No optimization has occurred before the first recoverable checkpoint.
    target.mkdir(parents=True, exist_ok=True)
    attempts = target / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt_index = 1
    while (attempts / f"attempt_{attempt_index:04d}").exists():
        attempt_index += 1
    attempt = attempts / f"attempt_{attempt_index:04d}"
    attempt.mkdir()
    create_file(attempt / "config.json", json_bytes(dict(config, runtime_device=str(device),
                actual_updates=total, reference_recipe=total == 2048 and seed == 0)))
    create_file(attempt / "sequence.json", json_bytes(sequence))
    create_file(attempt / "start.json", json_bytes(dict(completed_update=completed, bindings=bindings,
                resumed=resume is not None, reference_recipe=total == 2048 and seed == 0,
                sampling="uniform_source_then_view_then_EV; user_owned_data")))
    checkpoint = resume
    if resume is None:
        checkpoint = target / "checkpoints" / f"update_000000_{attempt.name}.pt"
        save_checkpoint(checkpoint, model, optimizer, scheduler, bindings=bindings,
                        sequence=sequence, include_cuda=use_cuda)
    try:
        for update in range(completed + 1, total + 1):
            samples = []
            for record in sequence[8 * (update - 1):8 * update]:
                row = rows[record["row"]]
                hdr, valid = load_prepared_sample(data_root, row)
                samples.append(prepare_sample(hdr, row["a0"], record["EV"], record["camera"], valid))
            result = update_once(model, optimizer, scheduler, update, samples, device)
            create_file(attempt / f"update_{update:06d}.json", json_bytes(result))
            if update % interval == 0 or update == total:
                checkpoint = target / "checkpoints" / f"update_{update:06d}_{attempt.name}.pt"
                save_checkpoint(checkpoint, model, optimizer, scheduler, bindings=bindings,
                                sequence=sequence, include_cuda=use_cuda)
            print(json.dumps(dict(update=update, updates=total, loss=result["loss"])), flush=True)
        after = dict(vae=tensor_digest(model.vae.state_dict()), bridge=tensor_digest(model.bridge.state_dict()))
        if after != foundation:
            raise RuntimeError("Frozen foundation changed during training")
        from safetensors.torch import save

        head_file = attempt / "head.safetensors"
        create_file(head_file, save(_cpu_clone(model.head.state_dict())))
        exported = target / "final"
        export_model(model, attempt / "final_export", exported)
        summary = dict(schema="gainweave-training-result-v1", completed_update=total,
                       checkpoint=None if checkpoint is None else str(checkpoint.relative_to(target)) if checkpoint.is_relative_to(target) else str(checkpoint),
                       head=str(head_file.relative_to(target)), head_sha256=file_sha256(head_file),
                       model_directory=str(exported.relative_to(target)),
                       bindings=bindings, frozen_foundation_unchanged=True,
                       reference_recipe=total == 2048 and seed == 0,
                       scope="User-owned data; not the original private paper sequence")
        create_file(attempt / "result.json", json_bytes(summary))
        return summary
    except BaseException as exc:
        create_file(attempt / "failure.json", json_bytes(dict(error_type=type(exc).__name__,
                    error=str(exc), last_completed_update=scheduler.completed_update,
                    latest_checkpoint=None if checkpoint is None else str(checkpoint))))
        raise
