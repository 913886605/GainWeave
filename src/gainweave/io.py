"""Explicit sRGB input, relative-linear EXR output and display-only previews."""
from __future__ import annotations

import math
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def load_srgb(image):
    """Read an RGB8 image; return a contiguous, unnormalized uint8 HWC array."""
    if isinstance(image, np.ndarray):
        value = image
    elif isinstance(image, Image.Image):
        value = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    elif isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            value = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    else:
        raise TypeError("Input must be an image path, a PIL image, or a uint8 RGB array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Input must be uint8 HWC RGB; normalized floats belong in forward_tensors")
    if min(value.shape[:2]) < 8:
        raise ValueError("Input images must be at least 8 pixels in each dimension")
    return np.array(value, dtype=np.uint8, order="C", copy=True)


def _unit_rgb(value, name):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim < 1 or value.shape[-1] != 3 or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} must be finite RGB in [0, 1]")
    return value


def srgb_to_linear(array):
    value = np.asarray(array)
    value = value.astype(np.float32) / np.float32(255.) if value.dtype == np.uint8 else value
    value = _unit_rgb(value, "sRGB")
    return np.where(value <= np.float32(.04045), value / np.float32(12.92),
                    ((value + np.float32(.055)) / np.float32(1.055)) ** np.float32(2.4)).astype(np.float32)


def linear_to_srgb(array):
    value = _unit_rgb(array, "Linear RGB")
    result = np.empty_like(value)
    low = value <= np.float32(.0031308)
    result[low] = np.float32(12.92) * value[low]
    result[~low] = np.float32(1.055) * np.power(value[~low], np.float32(1 / 2.4)) - np.float32(.055)
    return result


def _hdr(array):
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.issubdtype(value.dtype, np.floating):
        raise ValueError("HDR must be a floating-point HWC RGB array")
    if not np.isfinite(value).all() or (value < 0).any() or (value > np.finfo(np.float32).max).any():
        raise ValueError("HDR must be finite, nonnegative and representable in float32")
    return np.ascontiguousarray(value, dtype=np.float32)


def _cv2():
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    return cv2


def write_exr(path, hdr):
    """Write unscaled float32 RGB EXR, ZIP compressed, with an exact round trip."""
    value = _hdr(hdr)
    cv2 = _cv2()
    ok, encoded = cv2.imencode(".exr", np.ascontiguousarray(value[..., ::-1]),
        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT, cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
    if not ok:
        raise RuntimeError("OpenCV could not encode EXR; install the documented OpenCV build")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None or not np.array_equal(decoded[..., ::-1], value):
        raise RuntimeError("EXR encoding did not preserve native float32 RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path


def read_exr(path):
    cv2 = _cv2()
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    value = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if value is None or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Cannot decode three-channel EXR: {path}")
    return _hdr(value[..., ::-1])


def preview_rgb8(hdr, ev=1):
    """Shared EV -> RGB Reinhard -> exact sRGB -> RGB8. Never changes native HDR."""
    value = _hdr(hdr).astype(np.float64)
    if not isinstance(ev, (int, float)) or not math.isfinite(ev):
        raise ValueError("Preview EV must be finite")
    try:
        scale = 2. ** ev
    except OverflowError as error:
        raise ValueError("Preview EV is too large") from error
    with np.errstate(over="raise", invalid="raise"):
        try:
            value = value * scale
            mapped = value / (1. + value)
        except FloatingPointError as error:
            raise ValueError("Preview exposure overflows") from error
    srgb = np.where(mapped <= .0031308, mapped * 12.92, 1.055 * np.power(mapped, 1 / 2.4) - .055)
    return np.ascontiguousarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8))
