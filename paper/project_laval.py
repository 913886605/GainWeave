"""Project a licensed Laval EXR into the retained views used in the paper."""

import argparse
import csv
import hashlib
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

MANIFEST = Path(__file__).resolve().parent / "manifests" / "laval_views.csv"


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project(rgb, directions, device="cuda", resolution=1024, tile=256):
    """FP32, square 90-degree FOV; pixel-centred 2x raster followed by 2x2 averaging."""
    size = 2 * resolution
    if tile <= 0 or tile % 2:
        raise ValueError("tile must be a positive even number")
    source = torch.as_tensor(rgb, dtype=torch.float32, device=device)
    height, width, _ = rgb.shape
    basis = []
    for yaw, pitch in directions:
        y, p = math.radians(yaw), math.radians(pitch)
        basis.append(((math.cos(p)*math.cos(y), math.sin(p), math.cos(p)*math.sin(y)),
                      (-math.sin(y), 0., math.cos(y)),
                      (-math.sin(p)*math.cos(y), math.cos(p), -math.sin(p)*math.sin(y))))
    forward, right, up = torch.as_tensor(basis, dtype=torch.float32, device=device).unbind(1)
    forward, right, up = (v[:, None, None, :] for v in (forward, right, up))
    axis = (np.arange(size, dtype=np.float32) + np.float32(0.5)) / np.float32(size)
    x = torch.from_numpy(2 * axis - 1).to(device)[None, None, :, None]
    ys = torch.from_numpy(1 - 2 * axis).to(device)
    out = np.empty((len(directions), resolution, resolution, 3), dtype=np.float32)

    pi = torch.tensor(np.float32(math.pi), device=device)
    two_pi = torch.tensor(np.float32(2 * math.pi), device=device)
    for start in range(0, size, tile):
        stop = min(start + tile, size)
        rays = forward + x * right + ys[None, start:stop, None, None] * up
        rays = rays / torch.linalg.vector_norm(rays, dim=3, keepdim=True)
        lon = torch.remainder(torch.atan2(rays[..., 2], rays[..., 0]), two_pi)
        lat = torch.asin(torch.clamp(rays[..., 1], -1., 1.))
        sx = lon / two_pi * width - np.float32(0.5)
        sy = ((np.float32(0.5) - lat / pi) * height - np.float32(0.5)).clamp(0, height - 1)
        x0, y0 = sx.floor().long(), sy.floor().long()
        wx, wy = (sx - x0.float()).unsqueeze(-1), (sy - y0.float()).unsqueeze(-1)
        x1, x0 = (x0 + 1) % width, x0 % width
        y1 = (y0 + 1).clamp(max=height - 1)
        tl, tr = source[y0, x0], source[y0, x1]
        bl, br = source[y1, x0], source[y1, x1]
        top, bottom = tl + (tr - tl) * wx, bl + (br - bl) * wx
        patch = top + (bottom - top) * wy
        averaged = patch.reshape(len(directions), (stop-start)//2, 2, resolution, 2, 3).mean((2, 4))
        out[:, start//2:stop//2] = averaged.cpu().numpy()
    return out


def render_one(panorama, output, manifest=MANIFEST, device="cuda", verify_archive=False, expected_source=None):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    digest = file_sha256(panorama)
    with manifest.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["source_sha256"] == digest]
    if len(rows) != 4 or len({r["source_id"] for r in rows}) != 1:
        raise ValueError("The source EXR does not match a four-view paper source")
    if expected_source is not None and rows[0]["source_id"] != expected_source:
        raise ValueError("The local source map points to a different panorama")
    bgr = cv2.imread(str(panorama), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (3884, 7768, 3) or bgr.dtype != np.float32:
        raise ValueError("Expected the 7768 x 3884 floating-point Laval RGB panorama")
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    del bgr
    if not np.isfinite(rgb).all() or (rgb < 0).any():
        raise ValueError("Invalid source RGB geometry or radiance")
    directions = [(float(r["yaw_degrees"]), float(r["pitch_degrees"])) for r in rows]
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        crops = project(rgb, directions, device)
    output.mkdir(parents=True, exist_ok=True)
    for row, crop in zip(rows, crops, strict=True):
        if row["eligible"].lower() not in ("true", "1", "yes"):
            continue
        dest = output / (row["view_id"] + ".npz")
        expected = row["crop_npz_sha256"]
        if dest.exists():
            if verify_archive and file_sha256(dest) != expected:
                raise ValueError(f"Existing NPZ differs from archive: {dest.name}")
            with np.load(dest, allow_pickle=False) as data:
                if not np.array_equal(data["hdr_linear_rgb"], crop):
                    raise ValueError(f"Existing view differs from regenerated data: {dest.name}")
            continue
        with tempfile.NamedTemporaryFile(dir=output, suffix=".npz") as tmp:
            np.savez_compressed(tmp, hdr_linear_rgb=crop)
            tmp.flush()
            actual = file_sha256(Path(tmp.name))
            if verify_archive and actual != expected:
                raise ValueError(f"NPZ differs from archive for {dest.name}: {actual} vs {expected}")
            os.link(tmp.name, dest)
        print(dest.name, actual, "archive-match" if actual == expected else "different NPZ digest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--panorama", type=Path, help="One original EXR")
    mode.add_argument("--source-map", type=Path, help="CSV generated by manifests/match_laval.py")
    parser.add_argument("--root", type=Path, help="Unpacked Laval root for relative paths in --source-map")
    parser.add_argument("--verify-archive", action="store_true",
                        help="Require exact archived NPZ identity, including compression and serialization")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", help="CUDA for the paper projection; CPU for portability")
    args = parser.parse_args()
    if args.panorama is not None:
        render_one(args.panorama, args.output, device=args.device, verify_archive=args.verify_archive)
        return
    if args.root is None:
        parser.error("--source-map requires --root")
    with args.source_map.open(newline="") as fp:
        for row in csv.DictReader(fp):
            render_one(args.root / row["local_path"], args.output, device=args.device,
                       verify_archive=args.verify_archive, expected_source=row["source_id"])


if __name__ == "__main__":
    main()
