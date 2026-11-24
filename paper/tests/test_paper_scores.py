import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
from reproduce_tables import aggregate, tables
from score_predictions import fit_scale_terms, inverse_srgb, load_weight, masks_for, score


class PaperScoringTests(unittest.TestCase):
    def test_binary_support_and_threshold_equality(self):
        t = np.array([[[1,2,0],[1,2,0]]], dtype=np.float32)
        p = np.zeros_like(t)
        val = score(t,p,"hard_clip")
        self.assertEqual((val["n_full"],val["n_lost"],val["n_bright"]),(6,2,6))
        self.assertAlmostEqual(val["lost"],np.log(3))
        self.assertEqual(score(p,p,"hard_clip")["lost"],"")

    def test_digital_is_channelwise(self):
        t = np.ones((1,2,3),dtype=np.float32)
        rgb8 = np.array([[[255,254,0],[0,255,0]]],dtype=np.uint8)
        self.assertEqual(np.count_nonzero(masks_for(t,"vds",rgb8)["digital255"]),2)
        self.assertEqual(inverse_srgb(rgb8)[0,0,0],1)
        self.assertEqual(fit_scale_terms(t,np.full_like(t,2)),(12.,24.))

    def test_equal_ev_then_source(self):
        rows = [dict(case=str(i),source=s,ev=e,method="a",n_full=1,full=v)
                for i,(s,e,v) in enumerate([("p","0",0),("p","0",0),("p","1",6),("q","1",9)])]
        self.assertEqual(aggregate(rows,["a"],["full"],4)["a","full"],6)
        with self.assertRaises(ValueError):
            aggregate(rows + [rows[0]],["a"],["full"],4)

    def test_cli_weight_key(self):
        with TemporaryDirectory() as td:
            f = Path(td) / "gain.npz"
            a = np.array([[.75]], dtype=np.float32)
            np.savez(f,weight=a,delta_ev=np.zeros_like(a))
            np.testing.assert_array_equal(load_weight(f),a)

    def test_release_endpoints_and_gain(self):
        with TemporaryDirectory() as td:
            results=tables(PAPER/"scores", Path(td))
            self.assertEqual(results["selected_gain_ev"],.125)
            self.assertAlmostEqual(results["five_reduction_vs_rgb_percent"]["H0"],13.47,places=2)
            self.assertAlmostEqual(results["five_reduction_vs_rgb_percent"]["H"],14.30,places=2)


if __name__ == "__main__":
    unittest.main()
