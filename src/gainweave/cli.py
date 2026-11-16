"""Installed command line interface; model downloads are explicit."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

DEFAULT_MODEL = "Noel2211/GainWeave"
DEFAULT_REVISION = "v0.1.0"


def _model_arguments(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo ID or local model directory")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--vae-source", help="Local Lumina model root or VAE directory")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda")


def build_parser():
    parser = argparse.ArgumentParser(description="GainWeave HDR generation and recovery")
    parser.add_argument("--version", action="version", version="GainWeave 0.1.0")
    subs = parser.add_subparsers(dest="command", required=True)
    download = subs.add_parser("download", help="Download own weights and the required official base")
    _model_arguments(download)
    download.add_argument("--task", choices=("recover", "generate"), default="recover")
    for command in ("recover", "generate"):
        p = subs.add_parser(command)
        _model_arguments(p)
        p.add_argument("--output", type=Path, required=True, help="New output directory")
        p.add_argument("--variant", choices=("gainweave", "anchor", "fixed"), default="gainweave")
        p.add_argument("--presentation-ev", type=float, default=1.0)
        p.add_argument("--presentation-exr", action=argparse.BooleanOptionalAction, default=True)
        if command == "recover":
            p.add_argument("--input", type=Path, required=True)
        else:
            p.add_argument("--prompt", required=True)
            p.add_argument("--negative-prompt", default="")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--width", type=int, default=1024)
            p.add_argument("--height", type=int, default=1024)
            p.add_argument("--steps", type=int, default=50)
            p.add_argument("--guidance-scale", type=float, default=4.0)
            p.add_argument("--cfg-trunc-ratio", type=float, default=0.25)
            p.add_argument("--base-model", help="Local complete Lumina directory or HF repo ID")
            p.add_argument("--offload", choices=("auto", "none", "model", "sequential"), default="auto")
    p = subs.add_parser("prepare-data", help="Prepare a local explicitly linear HDR dataset")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p = subs.add_parser("train", help="Train the final highlight head; frozen released bridge")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--updates", type=int, help="Override update count; labels training as a modified recipe")
    p = subs.add_parser("evaluate", help="Score user-supplied paired relative HDR predictions")
    p.add_argument("--config", type=Path, required=True)
    return parser


def save_outputs(output, result, *, presentation_ev=1.0, presentation_exr=True, metadata=None):
    import math
    import numpy as np
    from PIL import Image
    from .io import write_exr, preview_rgb8
    if not math.isfinite(presentation_ev) or abs(presentation_ev) > 32:
        raise ValueError("presentation EV must be finite and between -32 and 32")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hdr = result["hdr_linear_rgb"]
    write_exr(output / "native.exr", hdr)
    Image.fromarray(preview_rgb8(hdr, presentation_ev)).save(output / "preview.png")
    if presentation_exr:
        scaled = hdr * np.float32(2.0 ** presentation_ev)
        name = "presentation_plus1ev.exr" if presentation_ev == 1.0 else f"presentation_EV{presentation_ev:+g}.exr"
        write_exr(output / name, scaled)
    np.savez_compressed(output / "gain.npz", delta_ev=result["delta_ev"], weight=result["weight"])
    description = dict(result.get("metadata", {}))
    description.update(metadata or {})
    description.update(native_exposure_ev=0, presentation_ev=presentation_ev,
                       native_units="relative_linear_rgb", physical_nits=None,
                       presentation_already_exposed=True,
                       preview="native -> exposure -> RGB Reinhard -> exact sRGB -> uint8")
    (output / "metadata.json").write_text(json.dumps(description, indent=2, allow_nan=False) + "\n")
    return output


def _generate(args):
    import torch
    from diffusers import Lumina2Pipeline
    from .hub import DEFAULT_BASE_MODEL, DEFAULT_BASE_REVISION
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Text generation requires a CUDA GPU; use recover for image recovery")
    if any(v <= 0 or v % 16 for v in (args.width, args.height)) or args.steps < 1:
        raise ValueError("width/height must be positive multiples of 16; steps must be positive")
    source = args.base_model or DEFAULT_BASE_MODEL
    local_base = Path(source).is_dir()
    if not local_base and source != DEFAULT_BASE_MODEL:
        raise ValueError("--base-model must be the pinned official Lumina repository or a local model directory")
    kwargs = dict(torch_dtype=torch.bfloat16, local_files_only=args.local_files_only,
                  cache_dir=args.cache_dir, low_cpu_mem_usage=True)
    if not local_base:
        kwargs["revision"] = DEFAULT_BASE_REVISION
    device = torch.device(args.device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    offload = args.offload
    if offload == "auto":
        offload = "none" if torch.cuda.get_device_properties(device_index).total_memory >= 40 * 1024**3 else "model"
    pipeline = Lumina2Pipeline.from_pretrained(source, **kwargs)
    if offload == "none":
        pipeline.to(device)
    elif offload == "model":
        pipeline.enable_model_cpu_offload(gpu_id=device_index)
    else:
        pipeline.enable_sequential_cpu_offload(gpu_id=device_index)
    start = time.monotonic()
    with torch.inference_mode():
        image = pipeline(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            width=args.width, height=args.height, num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale, cfg_trunc_ratio=args.cfg_trunc_ratio,
            cfg_normalization=True, max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0].convert("RGB")
    details = dict(prompt=args.prompt, negative_prompt=args.negative_prompt, seed=args.seed,
                   steps=args.steps, guidance_scale=args.guidance_scale,
                   cfg_trunc_ratio=args.cfg_trunc_ratio, cfg_normalization=True,
                   max_sequence_length=256, generation_dtype="bfloat16", offload=offload,
                   base_model="local" if local_base else DEFAULT_BASE_MODEL,
                   base_model_source=str(source),
                   base_revision=None if local_base else DEFAULT_BASE_REVISION,
                   generation_seconds=time.monotonic() - start)
    if hasattr(pipeline, "maybe_free_model_hooks"):
        pipeline.maybe_free_model_hooks()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return image, details


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        if args.command == "download":
            from .hub import download_assets
            result = download_assets(task=args.task, model=args.model, revision=args.revision,
                                     cache_dir=args.cache_dir, local_files_only=args.local_files_only)
            print(json.dumps(result, default=str, indent=2))
        elif args.command in ("recover", "generate"):
            from . import GainWeave
            if args.output.exists():
                raise FileExistsError(f"Output exists; choose a new directory: {args.output}")
            if args.command == "generate":
                image, details = _generate(args)
            else:
                image, details = args.input, dict(input=str(args.input), seed=None)
            model = GainWeave.from_pretrained(args.model, revision=args.revision,
                        vae_source=args.vae_source, device=args.device,
                        cache_dir=args.cache_dir, local_files_only=args.local_files_only)
            start = time.monotonic()
            result = model.recover(image, variant=args.variant)
            details.update(recovery_seconds=time.monotonic() - start, variant=args.variant)
            save_outputs(args.output, result, presentation_ev=args.presentation_ev,
                         presentation_exr=args.presentation_exr, metadata=details)
            if args.command == "generate":
                image.save(args.output / "input_sdr.png")
            print(f"Saved native HDR and presentation outputs to {args.output}")
        elif args.command == "prepare-data":
            from .training import prepare_dataset
            print(prepare_dataset(args.input_root, args.output_root, seed=args.seed))
        elif args.command == "train":
            from .training import run_training
            print(run_training(args.config, args.data_root, args.output, device=args.device,
                               updates=args.updates))
        else:
            from .evaluation import run_evaluation
            print(run_evaluation(args.config))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"GainWeave: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
