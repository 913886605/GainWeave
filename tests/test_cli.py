import json

import numpy as np
import pytest

from gainweave.cli import build_parser, save_outputs
from gainweave.io import read_exr


def test_commands_parse_without_model_downloads():
    parser = build_parser()
    args = parser.parse_args(["recover", "--input", "a.png", "--output", "out"])
    assert args.variant == "gainweave" and args.presentation_ev == 1
    assert args.revision == "v0.1.0"
    args = parser.parse_args(["generate", "--prompt", "lobby", "--output", "out"])
    assert (args.width, args.height, args.steps, args.offload) == (1024, 1024, 50, "auto")


def test_native_and_presentation_exposure_are_separate(tmp_path):
    hdr = np.array([[[0., .25, 12.], [3., .7, .1]]], dtype=np.float32)
    result = dict(hdr_linear_rgb=hdr, delta_ev=np.zeros((1, 2), np.float32),
                  weight=np.ones((1, 2), np.float32), metadata={})
    directory = save_outputs(tmp_path / "out", result)
    assert np.array_equal(read_exr(directory / "native.exr"), hdr)
    assert np.array_equal(read_exr(directory / "presentation_plus1ev.exr"), hdr * np.float32(2))
    meta = json.loads((directory / "metadata.json").read_text())
    assert meta["native_exposure_ev"] == 0 and meta["presentation_ev"] == 1
    with pytest.raises(FileExistsError):
        save_outputs(directory, result)


def test_invalid_exposure_rejected_before_output(tmp_path):
    for ev in (float("nan"), float("inf"), 100):
        with pytest.raises(ValueError):
            save_outputs(tmp_path / "out", {}, presentation_ev=ev)
    assert not (tmp_path / "out").exists()
