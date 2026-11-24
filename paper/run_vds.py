"""Regenerate VDS Table III using a prepared official VDS dataset and published weights."""
import argparse
import csv
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gainweave import GainWeave
from gainweave.paper_controls import PAPER_REVISION, PaperControl
from reproduce_tables import DIGITAL, PAPER, VDS, read_rows, verify_archived_scores
from score_predictions import vds_rows

EXTERNAL = ("gmnet", "vid2hdrimg", "x2hdr_ldr", "lediff")


def snapshot_input(path, expected):
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB8: {path}")
        codes = np.asarray(image)
    if codes.dtype != np.uint8 or hashlib.sha256(codes.tobytes()).hexdigest() != expected:
        raise ValueError(f"Input does not match official common crop: {path}")
    return codes


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with (PAPER / "manifests/vds_scenes.csv").open() as f:
        scene_ids = list(csv.DictReader(f))
    if len(scene_ids) != 68 or len({r["scene"] for r in scene_ids}) != 68:
        raise ValueError("Expected the full disjoint published CAL/TEST manifest")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as d:
        tmp = Path(d)
        rows = []
        for row in scene_ids:
            case = args.prepared.resolve() / row["scene"]
            ref = case / "linear_hdr.npy"
            if hashlib.sha256(ref.read_bytes()).hexdigest() != row["reference_npy_512_sha256"]:
                raise ValueError(f"Reference does not match registered VDS file: {ref}")
            rows.append(dict(input=case/"sdr.png", reference=ref, scene=row["scene"], split=row["split"],
                             predictions={}, rgb_sha=row["rgb8_512_sha256"]))
        for arm in ("five", "one", "rgb"):
            kwargs = dict(model=args.model, revision=args.revision, device=args.device,
                          local_files_only=args.local_files_only, vae_source=args.vae_source)
            pipeline = GainWeave.from_pretrained(**kwargs) if arm == "five" else PaperControl.from_pretrained(arm, **kwargs)
            for r in rows:
                scene = r["scene"]
                rgb8 = snapshot_input(r["input"], r["rgb_sha"])
                output = pipeline.recover(rgb8)
                arrays = {arm: output["hdr_linear_rgb"] if arm=="five" else output["H_new"]}
                if arm == "five":
                    arrays["anchor"] = output["anchor"]
                    r["weight"] = tmp / (scene + "-w.npy")
                    np.save(r["weight"],output["weight"], allow_pickle=False)
                for m, arr in arrays.items():
                    dest = tmp / f"{scene}-{m}.npy"
                    np.save(dest,arr,allow_pickle=False)
                    r["predictions"][m] = dest
            del pipeline
            gc.collect()
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        scales = {}
        methods = list(VDS)
        if args.external_predictions:
            for r in rows:
                for m in EXTERNAL:
                    r["predictions"][m] = args.external_predictions.resolve() / m / (r["scene"]+".npy")
        else:
            methods = [m for m in VDS if m not in EXTERNAL]
            verify_archived_scores(PAPER/"scores")
        out = tmp / "final"
        out.mkdir()
        with (out/"vds.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, ["case","source","ev","method",*[f"n_{d}" for d in DIGITAL],*DIGITAL], lineterminator="\n")
            writer.writeheader()
            writer.writerows(vds_rows(rows, tmp, scales, methods=methods))
            if not args.external_predictions:
                writer.writerows(r for r in read_rows(PAPER/"scores","vds") if r["method"] in EXTERNAL)
        report = {"new_scales": scales, "new_pixel_scores": methods,
                  "external_methods_using_released_scores": [] if args.external_predictions else list(EXTERNAL)}
        (out / "vds_provenance.json").write_text(json.dumps(report,indent=2)+"\n")
        os.rename(out, args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True, help="Output from manifests/prepare_vds.py")
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument("--external-predictions", type=Path, help="Native arrays <method>/<scene>.npy for 20 CAL and 48 TEST scenes; otherwise use archived scene scores for the four external methods")
    p.add_argument("--model", default="Noel2211/GainWeave")
    p.add_argument("--revision", default=PAPER_REVISION)
    p.add_argument("--vae-source", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    run(p.parse_args())
