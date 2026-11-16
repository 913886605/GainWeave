import numpy as np
import pytest
from PIL import Image
from gainweave import io


def test_srgb_roundtrip_and_endpoints():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, 2)
    result = io.linear_to_srgb(io.srgb_to_linear(image))
    np.testing.assert_array_equal(np.rint(result * 255).astype(np.uint8), image)
    assert io.srgb_to_linear(np.ones((1, 1, 3), np.float32)).min() == 1


def test_input_validation_and_copy():
    image = np.ones((8, 9, 3), np.uint8)
    copied = io.load_srgb(image)
    copied[:] = 2
    assert image.max() == 1
    with pytest.raises(ValueError):
        io.load_srgb(image.astype(np.float32))
    with pytest.raises(ValueError):
        io.load_srgb(np.zeros((7, 8, 3), np.uint8))
    assert io.load_srgb(Image.new("RGBA", (8, 8))).shape == (8, 8, 3)


def test_preview_exposure_applied_once():
    hdr = np.full((8, 8, 3), 1., np.float32)
    before = hdr.copy()
    np.testing.assert_array_equal(io.preview_rgb8(hdr, 1), io.preview_rgb8(hdr * 2, 0))
    np.testing.assert_array_equal(hdr, before)
    assert io.preview_rgb8(hdr, 2).min() > io.preview_rgb8(hdr, 1).max()
    with pytest.raises(ValueError):
        io.preview_rgb8(hdr * np.nan)


def test_exr_roundtrip(tmp_path):
    hdr = np.random.default_rng(0).uniform(0, 10, (8, 8, 3)).astype(np.float32)
    path = tmp_path / "native.exr"
    io.write_exr(path, hdr)
    np.testing.assert_array_equal(io.read_exr(path), hdr)
