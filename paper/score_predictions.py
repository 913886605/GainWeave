"""Score registered native predictions and references; see SCORING.md for the input manifest."""
import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from reproduce_tables import ARCH, DIGITAL, DOMAINS, PAPER, STRENGTHS, VDS, strength_id


def check_rgb(x):
    if (x.dtype != np.float32 or x.ndim != 3 or x.shape[2] != 3
            or not np.isfinite(x).all() or np.min(x) < 0):
        raise ValueError("Expected nonnegative finite float32 RGB, shape H x W x 3")
    return x


def native(path):
    if path.suffix.lower() == ".exr":
        from gainweave.io import read_exr
        return read_exr(path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Use native .exr or float32 RGB .npy")
    return check_rgb(np.load(path, allow_pickle=False))


def inverse_srgb(rgb8):
    if rgb8.dtype != np.uint8 or rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError("Expected uint8 RGB")
    u = rgb8.astype(np.float32) / np.float32(255.)
    return np.where(u <= np.float32(.04045), u / np.float32(12.92),
                    ((u + np.float32(.055)) / np.float32(1.055)) ** np.float32(2.4))


def luminance(x):
    return x[..., 0] * np.float32(.2126) + x[..., 1] * np.float32(.7152) + x[..., 2] * np.float32(.0722)


def exposed_target(hdr, ev):
    check_rgb(hdr)
    y = luminance(hdr)
    positive = y[y > 0].astype(np.float64)
    if not positive.size:
        raise ValueError("Cannot expose an all-black view")
    a0 = np.float32(.18 / np.exp(np.mean(np.log(positive), dtype=np.float64)))
    return check_rgb(hdr * a0 * np.float32(2.**ev))


def fixed(anchor, weight, a):
    if weight.dtype != np.float32 or weight.shape != anchor.shape[:2]:
        raise ValueError("Saved weight must be a float32 H x W array on the anchor grid")
    if not np.isfinite(weight).all() or weight.min() < 0 or weight.max() > 1:
        raise ValueError("Invalid gain weight")
    return anchor * np.exp2(np.float32(a) * weight[..., None])


def fit_scale_terms(target, pred):
    if check_rgb(target).shape != check_rgb(pred).shape:
        raise ValueError("Calibration geometry mismatch")
    p = pred.astype(np.float64).ravel()
    t = target.astype(np.float64).ravel()
    return float(p @ t), float(p @ p)


def masks_for(target, camera, rgb8=None):
    check_rgb(target)
    if camera == "vds":
        if rgb8 is None or rgb8.shape != target.shape or rgb8.dtype != np.uint8:
            raise ValueError("VDS requires the paired RGB8 input")
        y = np.einsum("hwc,c->hw", target.astype(np.float64), np.array([.2126, .7152, .0722]))
    elif camera in ("hard_clip", "soft_compression"):
        y = luminance(target).astype(np.float64)
    else:
        raise ValueError(f"Unknown camera {camera}")
    masks = {"full": None, "bright": y >= np.quantile(y, .99, method="linear")}
    if camera == "hard_clip":
        masks["lost"] = target > np.float32(1.)
    if camera == "vds":
        masks["digital255"] = rgb8 == 255
    return masks


def measure(pred, log_target, masks, k=1.):
    if check_rgb(pred).shape != log_target.shape or not math.isfinite(k) or k < 0:
        raise ValueError("Invalid prediction geometry or calibration")
    error = np.abs(np.log1p(pred.astype(np.float64) * k) - log_target)
    result = {}
    for name, mask in masks.items():
        selected = error if mask is None else error[mask]
        result[f"n_{name}"] = selected.size
        result[name] = float(selected.mean(dtype=np.float64)) if selected.size else ""
    return result


def score(target, pred, camera, rgb8=None, k=1.):
    return measure(pred, np.log1p(target.astype(np.float64)), masks_for(target, camera, rgb8), k)


def load_weight(path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            # gain.npz from gainweave recover stores a 2-D input-derived weight.
            return data["weight"]
    return np.load(path, allow_pickle=False)


def laval_rows(records, base, group):
    for r in records:
        hard = group != "features_soft_compression"
        camera = "hard_clip" if hard else "soft_compression"
        ev = int(r["ev"])
        target = exposed_target(native(base / r["raw_hdr"]), ev)
        masks = masks_for(target, camera)
        log_target = np.log1p(target.astype(np.float64))
        key = {"case": r["case"], "source": r["source"], "ev": ev}
        if target.shape != (1024, 1024, 3):
            raise ValueError("Expected 1024 x 1024 Laval paper grid")
        if group.startswith("features_"):
            for m in ARCH:
                yield dict(**key, method=m, **measure(native(base / r["predictions"][m]), log_target, masks))
            continue
        s = np.minimum(target, np.float32(1))
        learned = native(base / r["learned"])
        if s.shape != learned.shape:
            raise ValueError("Laval geometry mismatch")
        def copied(p):
            return np.where(s < 1, s, p)
        if group == "laval_q80":
            # The two native Q80 values use the original pipeline outputs, including native Fixed.
            f = native(base / r["fixed_native"])
            methods = (("fixed_native", f), ("learned_native", learned),
                       ("fixed_copy", copied(f)), ("learned_copy", copied(learned)))
            for m, pred in methods:
                yield dict(**key, method=m, **measure(pred, log_target, masks))
        else:
            anchor, weight = native(base / r["anchor"]), load_weight(base / r["weight"])
            for a in STRENGTHS:
                p = copied(fixed(anchor, weight, a))
                yield dict(**key, method=strength_id(a), **measure(p, log_target, masks))
            yield dict(**key, method="learned_copy", **measure(copied(learned), log_target, masks))


def load_case(r, base):
    from PIL import Image
    with Image.open(base / r["input"]) as image:
        if image.mode != "RGB":
            raise ValueError("VDS requires the frozen common RGB8 inputs")
        rgb8 = np.array(image)
    target = native(base / r["reference"])
    if rgb8.shape != target.shape or rgb8.shape != (512, 512, 3):
        raise ValueError("Expected registered 512 x 512 VDS inputs")
    return target, rgb8, inverse_srgb(rgb8)


def vds_rows(records, base, scales_out, methods=VDS):
    terms = defaultdict(list)
    native_methods = [m for m in methods if not m.endswith("_copy")]
    for r in records:
        if r["split"] != "CAL":
            continue
        target, _, linear = load_case(r, base)
        for m in native_methods:
            pred = linear if m == "linear_sdr" else native(base / r["predictions"][m])
            terms[m].append(fit_scale_terms(target, pred))
    scales = {}
    for m in native_methods:
        denominator = math.fsum(d for n, d in terms[m])
        if denominator <= 0 or len(terms[m]) != 20:
            raise ValueError(f"Missing calibration or zero energy for {m}")
        scales[m] = max(0., math.fsum(n for n, d in terms[m]) / denominator)
    for m in ("learned_copy", "fixed_copy"):
        scales[m] = scales["anchor"]
    for r in records:
        if r["split"] != "TEST":
            continue
        target, rgb8, linear = load_case(r, base)
        masks, log_target = masks_for(target, "vds", rgb8), np.log1p(target.astype(np.float64))
        for m in methods:
            if m == "linear_sdr":
                p = linear
            elif m in ("learned_copy", "fixed_copy"):
                if m == "learned_copy":
                    p = native(base / r["predictions"]["five"])
                else:
                    anchor = native(base / r["predictions"]["anchor"])
                    p = fixed(anchor, load_weight(base / r["weight"]), .125)
                p = np.where(rgb8 < 255, linear, p)
            else:
                p = native(base / r["predictions"][m])
            yield dict(case=r["scene"], source=r["scene"], ev="", method=m,
                       **measure(p, log_target, masks, scales[m]))
    scales_out.update(scales)


def validate_ids(records, group, manifests):
    if group == "vds":
        with (manifests / "vds_scenes.csv").open() as f:
            expected = {(r["scene"], r["split"]) for r in csv.DictReader(f)}
        observed = {(r["scene"], r["split"]) for r in records}
        if (len(expected) != 68 or sum(s=="CAL" for _,s in expected) != 20 or
                len({scene for scene,_ in expected}) != 68):
            raise ValueError("Invalid VDS sample manifest")
    else:
        cam = "soft_compression" if group.endswith("soft_compression") else "hard_clip"
        with (manifests / "laval_dev_cases.csv").open() as f:
            expected = {(r["case_id"], r["source_id"], int(r["ev"])) for r in csv.DictReader(f)
                        if r["case_id"].endswith("_" + cam)
                        and (group != "laval_q80" or r["q80_id"])}
        observed = {(r["case"], r["source"], int(r["ev"])) for r in records}
        if len(expected) != (80 if group=="laval_q80" else 4335):
            raise ValueError("Invalid Laval sample manifest")
    if len(records) != len(observed) or observed != expected:
        raise ValueError("Case, EV, source or split differs from the paper manifest")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("group", choices=["laval_q80","laval_dev","features_hard_clip","features_soft_compression","vds"])
    cli.add_argument("manifest", type=Path, help="Local JSONL of paths relative to the manifest")
    cli.add_argument("--output", type=Path, required=True, help="Directory for regenerated case CSVs")
    args = cli.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    validate_ids(records, args.group, PAPER / "manifests")
    base = args.manifest.parent
    scales = {}
    ds = DIGITAL if args.group == "vds" else (("full","bright") if args.group.endswith("soft_compression") else DOMAINS)
    rows = vds_rows(records, base, scales) if args.group == "vds" else laval_rows(records, base, args.group)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.group}.csv"
    if path.with_suffix(".csv.gz").exists():
        raise ValueError("Choose a new output directory; compressed scores take precedence")
    with tempfile.TemporaryDirectory(dir=args.output) as d:
        target = Path(d) / path.name
        with target.open("w", newline="") as f:
            w = csv.DictWriter(f, ["case","source","ev","method",*[f"n_{k}" for k in ds],*ds], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        if scales:
            (args.output / "vds_scales.json").write_text(json.dumps(scales, indent=2) + "\n")
        os.replace(target,path)


if __name__ == "__main__":
    main()
