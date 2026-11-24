"""Stream Table II and IV measurements from licensed projected Laval views and public weights."""
import argparse
import csv
import gc
import os
import tempfile
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from gainweave import GainWeave
from gainweave.io import linear_to_srgb
from gainweave.paper_controls import PAPER_REVISION, PaperControl
from reproduce_tables import DOMAINS, PAPER, STRENGTHS, strength_id
from score_predictions import check_rgb, exposed_target, masks_for, measure


def tensor(x, device):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(device)


def image(value):
    return np.ascontiguousarray(value[0].detach().cpu().permute(1, 2, 0).numpy())


def writer(stack, directory, name, domains):
    f = stack.enter_context((directory / f"{name}.csv").open("w", newline=""))
    w = csv.DictWriter(f, ["case", "source", "ev", "method", *[f"n_{k}" for k in domains], *domains],
                       lineterminator="\n")
    w.writeheader()
    return w


def copy_scores(anchor, learned, weight, s, target, masks):
    """No error outside the touched components after copying the hard-clipped linear input."""
    touched = s == np.float32(1)
    fields = np.broadcast_to(weight, s.shape)
    h0, h, w = anchor[touched], learned[touched], fields[touched]
    log_t = np.log1p(target[touched].astype(np.float64))
    inside = {"full": slice(None), "lost": masks["lost"][touched],
              "bright": np.broadcast_to(masks["bright"][..., None], s.shape)[touched]}
    counts = {"full": s.size, "lost": int(masks["lost"].sum()), "bright": int(masks["bright"].sum()) * 3}
    def score(p):
        error = np.abs(np.log1p(p.astype(np.float64)) - log_t)
        return {**{"n_" + k: n for k, n in counts.items()},
                **{k: float(error[inside[k]].sum(dtype=np.float64)) / n if n else ""
                   for k, n in counts.items()}}
    yield "learned_copy", score(h)
    for a in STRENGTHS:
        p = h0 * np.exp2(np.float32(a) * w)
        yield strength_id(a), score(p)


def case_scores(r, hdr, arm, pipeline, writers, device):
    cam, ev = r["camera"], int(r["ev"])
    t = exposed_target(hdr, ev)
    s = np.minimum(t, np.float32(1)) if cam == "hard_clip" else t / (np.float32(1) + t)
    outputs = pipeline.forward_tensors(tensor(linear_to_srgb(s), device), tensor(s, device))
    h0, h = image(outputs["H0"]), image(outputs["H_new"])
    masks, log_t = masks_for(t, cam), np.log1p(t.astype(np.float64))
    key = dict(case=r["case_id"], source=r["source_id"], ev=ev)
    for stage, pred in (("H0", h0), ("H", h)):
        writers[f"features_{cam}"].writerow(dict(**key, method=f"{arm}_{stage}", **measure(pred, log_t, masks)))
    if arm != "five" or cam != "hard_clip":
        return
    for m, stats in copy_scores(h0, h, image(outputs["w"]), s, t, masks):
        writers["laval_dev"].writerow(dict(**key, method=m, **stats))
    if r["q80_id"]:
        f = image(outputs["H0"] * torch.exp2(.25 * outputs["w"]))
        for m, pred in (("fixed_native", f), ("learned_native", h),
                        ("fixed_copy", np.where(s < 1, s, f)), ("learned_copy", np.where(s < 1, s, h))):
            writers["laval_q80"].writerow(dict(**key, method=m, **measure(pred, log_t, masks)))


def run(args):
    with (PAPER / "manifests/laval_dev_cases.csv").open() as f:
        records = list(csv.DictReader(f))
    records.sort(key=lambda r: (r["view_id"], int(r["ev"]), r["camera"]))
    if len(records) != 8670 or len({r["case_id"] for r in records}) != 8670:
        raise ValueError("The published manifest must cover both cameras and 4,335 cases per camera")
    if args.q80_only:
        records = [r for r in records if r["camera"] == "hard_clip" and r["q80_id"]]
        if len(records) != 80:
            raise ValueError("Expected the 80 registered Q cases")
    views = sorted({r["view_id"] for r in records})
    if args.limit_views is not None:
        if not 1 <= args.limit_views <= len(views):
            raise ValueError("--limit-views is out of range")
        selected = set(views[:args.limit_views])
        records = [r for r in records if r["view_id"] in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        destination = Path(temp) / "scores"
        destination.mkdir()
        with ExitStack() as stack:
            writers = {k: writer(stack, destination, k, DOMAINS)
                       for k in ("laval_dev", "laval_q80", "features_hard_clip")}
            writers["features_soft_compression"] = writer(stack, destination, "features_soft_compression",
                                                         ("full", "bright"))
            for arm in args.arms:
                kwargs = dict(model=args.model, device=args.device, local_files_only=args.local_files_only,
                              revision=args.revision, vae_source=args.vae_source)
                model = GainWeave.from_pretrained(**kwargs) if arm == "five" else PaperControl.from_pretrained(arm, **kwargs)
                model.eval().requires_grad_(False)
                with torch.no_grad():
                    for i, (vid, group) in enumerate(groupby(records, key=lambda r: r["view_id"]), 1):
                        with np.load(args.views / f"{vid}.npz", allow_pickle=False) as pair:
                            hdr = check_rgb(pair["hdr_linear_rgb"])
                            if hdr.shape != (1024, 1024, 3):
                                raise ValueError(f"Expected paper geometry on {vid}")
                            if "valid_mask" in pair and not np.all(pair["valid_mask"]):
                                raise ValueError(f"Paper population uses full valid support: {vid}")
                        for row in group:
                            case_scores(row, hdr, arm, model, writers, args.device)
                        if i % 20 == 0:
                            print(f"{arm}: {i} views", flush=True)
                del model
                gc.collect()
                if torch.device(args.device).type == "cuda":
                    torch.cuda.empty_cache()
        os.rename(destination, args.output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--views", type=Path, required=True, help="Directory of <view_id>.npz projections")
    cli.add_argument("--output", type=Path, required=True, help="New directory of paper score CSVs")
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--model", default="Noel2211/GainWeave")
    cli.add_argument("--revision", default=PAPER_REVISION)
    cli.add_argument("--vae-source", type=Path)
    cli.add_argument("--local-files-only", action="store_true")
    cli.add_argument("--limit-views", type=int, help="Smoke test on the first N view IDs")
    cli.add_argument("--arms", nargs="+", default=["five","one","rgb"], choices=["five","one","rgb"])
    cli.add_argument("--q80-only", action="store_true", help="Use the 16 Q80 views and their five hard-clipped exposures")
    args = cli.parse_args()
    if len(set(args.arms)) != len(args.arms):
        cli.error("--arms must contain distinct names")
    run(args)
