"""Rebuild the paper's registered 512-square inputs and references from the author VDS release."""
import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read_verified(root, member, expected_sha):
    data = (root / member).read_bytes()
    if digest(data) != expected_sha:
        raise ValueError(f"Source identity mismatch: {member}")
    return data


def rebuild(root, row):
    raw_input = read_verified(root, row["sdr_member"], row["sdr_sha256"])
    raw_reference = read_verified(root, row["reference_member"], row["reference_sha256"])
    with Image.open(BytesIO(raw_input)) as img:
        if img.mode != "RGB" or img.size != (912, 608):
            raise ValueError(f"Unexpected input grid or channels: {row['scene']}")
        codes = np.array(img)
    bgr = cv2.imdecode(np.frombuffer(raw_reference, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (608, 912, 3) or bgr.dtype != np.float32:
        raise ValueError(f"Unexpected reference grid or channels: {row['scene']}")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    codes512 = cv2.resize(codes[:, 152:760], (512, 512), interpolation=cv2.INTER_AREA)
    hdr512 = cv2.resize(rgb[:, 152:760], (512, 512), interpolation=cv2.INTER_AREA)
    if digest(codes512.tobytes()) != row["rgb8_512_sha256"]:
        raise ValueError(f"Prepared RGB8 identity mismatch: {row['scene']}")
    reference_file = BytesIO()
    np.save(reference_file, hdr512.astype("<f4", copy=False), allow_pickle=False)
    if digest(reference_file.getvalue()) != row["reference_npy_512_sha256"]:
        raise ValueError(f"Prepared reference identity mismatch: {row['scene']}")
    return codes512, reference_file.getvalue()


def store(path, blob):
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError(f"An existing output differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(blob)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path, help="Directory containing the unpacked official *_set folders")
    p.add_argument("--split", choices=("CAL", "TEST", "all"), default="all")
    opts = p.add_mutually_exclusive_group(required=True)
    opts.add_argument("--verify-only", action="store_true", help="Check source and prepared identities without writing")
    opts.add_argument("--output", type=Path, help="Save <scene>/sdr.png and <scene>/linear_hdr.npy")
    args = p.parse_args()
    cv2.setNumThreads(1)
    count = 0
    with (HERE / "vds_scenes.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if args.split not in ("all", row["split"]):
                continue
            codes512, reference_npy = rebuild(args.root, row)
            if args.output:
                scene_dir = args.output / row["scene"]
                scene_dir.mkdir(parents=True, exist_ok=True)
                png = BytesIO()
                Image.fromarray(codes512).save(png, format="PNG", optimize=False, compress_level=6)
                store(scene_dir / "sdr.png", png.getvalue())
                store(scene_dir / "linear_hdr.npy", reference_npy)
            count += 1
    print(f"{count} {args.split} scene pairs matched both source and prepared identities.")


if __name__ == "__main__":
    main()
