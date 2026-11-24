"""Projection orientation, 90-degree frustum and even-tile equivalence."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("paper_project_laval", Path(__file__).parents[1] / "paper/project_laval.py")
projector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projector)


def test_longitude_wrap_and_area_average():
    nx = 128
    x = np.arange(nx)
    stripe = np.stack([np.cos(2*np.pi*(x+.5)/nx), np.sin(2*np.pi*(x+.5)/nx), np.ones(nx)], axis=1)
    hdr = np.tile(stripe, (64, 1, 1)).astype("float32")
    directions = [(0., 0.), (90., 0.)]
    p = projector.project(hdr, directions, "cpu", resolution=8, tile=4)
    np.testing.assert_array_equal(p, projector.project(hdr, directions, "cpu", resolution=8, tile=8))
    np.testing.assert_allclose(p[0, :, :, 0], p[1, :, :, 1], atol=3e-7)
    np.testing.assert_allclose(p[..., 2], 1., atol=0)
    assert p[0, ..., 1].min() < 0 < p[0, ..., 1].max()
