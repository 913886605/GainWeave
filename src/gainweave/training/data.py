"""User-owned, scene-grouped linear RGB data; no bundled research dataset."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np

INPUT_SCHEMA = "gainweave-input-dataset-v1"
DATASET_SCHEMA = "gainweave-dataset-v1"
EVS = (-2, -1, 0, 1, 2)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def create_file(path: str | Path, content: bytes) -> None:
    """Durably create a file without replacing an earlier artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def relative_file(root: Path, value: str) -> Path:
    """Resolve only explicitly declared files inside a dataset directory."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Dataset paths must be nonempty relative paths")
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError(f"Missing file or path outside dataset: {value}")
    return result


def validate_rgb(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("Expected nonempty HWC RGB array with three channels")
    if value.dtype.kind != "f":
        raise ValueError("Linear HDR arrays must have a floating-point dtype")
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Linear RGB must be finite and nonnegative")
    return value


def read_linear_rgb(path: str | Path) -> np.ndarray:
    """Read EXR/HDR, declared linear NPY, or NPZ['hdr_linear_rgb']; no pickle."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "hdr_linear_rgb" not in archive.files:
                raise ValueError("HDR NPZ requires the named array 'hdr_linear_rgb'")
            value = archive["hdr_linear_rgb"]
    elif suffix == ".exr":
        from gainweave.io import read_exr

        value = read_exr(path)
    elif suffix == ".hdr":
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("RGBE .hdr input requires opencv-python") from exc
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"Cannot read RGBE image: {path}")
        value = value[..., ::-1]  # OpenCV BGR -> declared RGB.
    else:
        raise ValueError("Linear HDR inputs must be .exr, .hdr, .npy, or .npz")
    return validate_rgb(value)


def validate_mask(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """0 excludes, 1 includes, 255 ignores; do not treat 255 as foreground."""
    if value.shape != shape or value.dtype.kind not in "bui":
        raise ValueError("Masks must match H,W and use bool/integer 0,1,255 labels")
    if not np.isin(value, (0, 1, 255)).all():
        raise ValueError("Allowed mask labels are 0 (outside), 1 (inside), 255 (ignore)")
    return np.ascontiguousarray(value == 1)


def read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "mask" not in archive.files:
                raise ValueError("Standalone mask NPZ requires the named array 'mask'")
            value = archive["mask"]
    else:
        from PIL import Image

        with Image.open(path) as image:
            value = np.asarray(image)
    return validate_mask(value, shape)


def luminance(rgb: np.ndarray) -> np.ndarray:
    work = rgb.astype(np.float32, copy=False)
    return (work[..., 0] * np.float32(.2126) + work[..., 1] * np.float32(.7152)
            + work[..., 2] * np.float32(.0722))


def positive_log_average_scale(rgb: np.ndarray) -> tuple[np.float32, float, int]:
    """a0 = float32(.18 / exp(mean(log(Y[Y>0])))), without epsilon."""
    y = luminance(validate_rgb(rgb))
    selected = y[y > 0]
    count = int(selected.size)
    if not count:
        raise ValueError("Positive log-average is undefined for an all-black source")
    statistic = float(np.exp(np.log(selected.astype(np.float64)).sum(dtype=np.float64) / count))
    scale = np.float32(.18 / statistic)
    if not math.isfinite(statistic) or statistic <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive log-average scale must be finite and positive")
    return scale, statistic, count


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(str(value.shape).encode() + value.tobytes(order="C")).hexdigest()


def validate_source_splits(rows: list[dict]) -> None:
    """Multiple views are allowed; a scene or identical image cannot cross splits."""
    sources, paths, arrays, views = {}, set(), {}, set()
    for row in rows:
        source = row.get("source_id")
        split = row.get("split")
        if not isinstance(source, str) or not source.strip() or split not in ("train", "dev"):
            raise ValueError("Each row requires a nonempty source_id and train/dev split")
        if source in sources and sources[source] != split:
            raise ValueError(f"Source leakage between train/dev: {source}")
        sources[source] = split
        if row["path"] in paths:
            raise ValueError("Duplicate dataset path")
        paths.add(row["path"])
        view = row.get("view_id", row["path"])
        if (source, view) in views:
            raise ValueError("Duplicate source_id/view_id")
        views.add((source, view))
        digest = row.get("array_sha256")
        if digest:
            previous = arrays.get(digest)
            if previous is not None:
                raise ValueError("Duplicate RGB content: keep one image with its original source_id")
            arrays[digest] = (source, split)


def prepare_dataset(input_root, output_root, seed=0) -> Path:
    """Prepare dataset.json's user-owned linear RGB into source-disjoint NPZs.

    Input: {schema: gainweave-input-dataset-v1, color_space: linear_rgb,
    rows: [{source_id, path, split?: train|dev, valid_mask?: relative_path}]}.
    Declare all splits or none. Automatic splitting shuffles sorted source IDs
    with the supplied seed, assigns 20% (at least one) to dev, and never divides
    views of one source. At least two sources are required in automatic mode.
    The output directory must be absent or empty. No image resizing is applied.
    """
    input_root, output_root = Path(input_root).resolve(), Path(output_root).resolve()
    if input_root == output_root or output_root.is_relative_to(input_root):
        raise ValueError("Prepared output must be outside the input dataset directory")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Prepared output is not empty: {output_root}")
    declaration = json.loads((input_root / "dataset.json").read_text())
    if declaration.get("schema") != INPUT_SCHEMA or declaration.get("color_space") != "linear_rgb":
        raise ValueError("dataset.json must explicitly declare schema and color_space='linear_rgb'")
    rows = declaration.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset.json requires a nonempty rows list")
    rows = [dict(row) for row in rows]
    supplied = ["split" in row for row in rows]
    if any(supplied) and not all(supplied):
        raise ValueError("Declare a split for every row, or leave all splits automatic")
    if not any(supplied):
        sources = sorted({row.get("source_id", "") for row in rows})
        if len(sources) < 2:
            raise ValueError("Automatic source-based train/dev splitting needs at least two sources")
        random.Random(seed).shuffle(sources)
        dev = set(sources[:max(1, int(round(.2 * len(sources))))])
        for row in rows:
            row["split"] = "dev" if row.get("source_id") in dev else "train"
    # Validate every source before writing any output.
    for row in rows:
        path = relative_file(input_root, row["path"])
        rgb = read_linear_rgb(path)
        if any(side % 8 for side in rgb.shape[:2]):
            raise ValueError("Training H,W must be multiples of 8; prepare your own crop explicitly")
        a0, statistic, count = positive_log_average_scale(rgb)
        row.update(array_sha256=_array_sha(rgb), input_sha256=file_sha256(path),
                   shape=list(rgb.shape), a0=float(a0), positive_log_average=statistic,
                   positive_luma_count=count)
        if row.get("valid_mask"):
            mask_path = relative_file(input_root, row["valid_mask"])
            read_mask(mask_path, rgb.shape[:2])
            row["mask_sha256"] = file_sha256(mask_path)
    validate_source_splits(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, row in enumerate(sorted(rows, key=lambda x: (x["source_id"], x["path"]))):
        input_path = relative_file(input_root, row["path"])
        if file_sha256(input_path) != row["input_sha256"]:
            raise ValueError("Input changed during dataset preparation")
        rgb = read_linear_rgb(input_path)
        valid = np.ones(rgb.shape[:2], dtype=np.bool_)
        if row.get("valid_mask"):
            mask_path = relative_file(input_root, row["valid_mask"])
            if file_sha256(mask_path) != row["mask_sha256"]:
                raise ValueError("Mask changed during dataset preparation")
            valid = read_mask(mask_path, rgb.shape[:2])
        relative = f"samples/{index:06d}.npz"
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation; no silent replacement of prior samples.
        with path.open("xb") as stream:
            np.savez_compressed(stream, hdr_linear_rgb=rgb, valid_mask=valid)
        prepared.append(dict(source_id=row["source_id"], view_id=row.get("view_id", f"view_{index:06d}"),
                             path=relative, split=row["split"], sha256=file_sha256(path),
                             array_sha256=row["array_sha256"], shape=row["shape"],
                             a0=row["a0"], positive_log_average=row["positive_log_average"],
                             positive_luma_count=row["positive_luma_count"]))
    result = dict(schema=DATASET_SCHEMA, color_space="linear_rgb", seed=int(seed), rows=prepared,
                  exposure=dict(EVs=list(EVS), scale="positive_log_average_.18_no_epsilon"),
                  split_rule="explicit_source_groups" if any(supplied) else "seeded_source_groups_80_20")
    manifest_path = output_root / "manifest.json"
    create_file(manifest_path, json_bytes(result))
    return manifest_path


def load_dataset(root: str | Path, manifest="manifest.json") -> tuple[dict, list[dict]]:
    root = Path(root).resolve()
    path = relative_file(root, manifest)
    value = json.loads(path.read_text())
    if value.get("schema") != DATASET_SCHEMA or value.get("color_space") != "linear_rgb":
        raise ValueError("Prepared dataset schema or linear RGB declaration is missing")
    rows = value.get("rows", [])
    if not rows:
        raise ValueError("Prepared dataset has no rows")
    validate_source_splits(rows)
    return value, rows


def load_prepared_sample(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    path = relative_file(root, row["path"])
    if file_sha256(path) != row["sha256"]:
        raise ValueError(f"Prepared sample SHA256 mismatch: {row['path']}")
    rgb = read_linear_rgb(path)
    with np.load(path, allow_pickle=False) as archive:
        if "valid_mask" not in archive.files:
            raise ValueError("Prepared sample is missing valid_mask")
        valid = validate_mask(archive["valid_mask"], rgb.shape[:2])
    if list(rgb.shape) != row["shape"] or _array_sha(rgb) != row["array_sha256"]:
        raise ValueError("Prepared RGB content/shape does not match its manifest")
    scale, statistic, count = positive_log_average_scale(rgb)
    if float(scale) != row["a0"] or statistic != row["positive_log_average"] or count != row["positive_luma_count"]:
        raise ValueError("Prepared exposure metadata does not match RGB data")
    return rgb, valid


def exact_srgb_oetf(linear: np.ndarray) -> np.ndarray:
    if linear.dtype != np.float32 or not np.isfinite(linear).all() or (linear < 0).any() or (linear > 1).any():
        raise ValueError("sRGB OETF input must be finite float32 linear RGB in [0,1]")
    output = np.empty_like(linear)
    low = linear <= np.float32(.0031308)
    output[low] = np.float32(12.92) * linear[low]
    output[~low] = np.float32(1.055) * np.power(linear[~low], np.float32(1 / 2.4)) - np.float32(.055)
    return output


def prepare_sample(hdr, a0, ev, camera, valid=None) -> dict:
    hdr = validate_rgb(hdr)
    if ev not in EVS or camera not in ("hard_clip", "soft_compress") or not math.isfinite(float(a0)) or a0 <= 0:
        raise ValueError("Expected EV -2..2, positive a0, and hard_clip/soft_compress camera")
    target = (hdr * np.float32(a0) * np.float32(2. ** ev)).astype(np.float32)
    if not np.isfinite(target).all():
        raise ValueError("Exposure overflow")
    base = (np.clip(target, 0, 1) if camera == "hard_clip" else target / (np.float32(1) + target)).astype(np.float32)
    valid = np.ones(hdr.shape[:2], dtype=np.bool_) if valid is None else validate_mask(valid, hdr.shape[:2])
    arrays = dict(s_srgb=exact_srgb_oetf(base), base=base, target=target)
    result = {key: np.ascontiguousarray(value.transpose(2, 0, 1)) for key, value in arrays.items()}
    return dict(result, valid=valid[None], camera=camera, EV=int(ev), a0=float(np.float32(a0)))
