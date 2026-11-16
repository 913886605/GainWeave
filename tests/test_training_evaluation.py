"""CPU-only synthetic recipe, restart, support, and exposure checks."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gainweave.models import build_head
from gainweave.models.smooth import apply_head
from gainweave.training.data import (load_dataset, positive_log_average_scale,
                                    file_sha256, prepare_dataset, prepare_sample)
from gainweave.training.engine import (capture_rng, load_checkpoint, make_sequence,
                                       run_training, save_checkpoint, seed_all, tensor_digest)
from gainweave.training.losses import (FixedScheduler, loss_denominators,
                                       optimizer_for, smooth_loss, tensors_for_sample,
                                       update_once)
from gainweave.evaluation import aggregate_cases, evaluate_case, run_evaluation


class ToyFrozenModel:
    """Tiny frozen features exercise real head/loss without a VAE checkpoint."""
    def __init__(self):
        self.vae = torch.nn.Conv2d(3, 3, 1).eval().requires_grad_(False)
        self.bridge = torch.nn.Conv2d(3, 16, 1).eval().requires_grad_(False)
        self.head = build_head().train().requires_grad_(True)

    def forward_tensors(self, srgb, base):
        with torch.no_grad():
            features = self.bridge(self.vae(srgb)).detach().clone()
            anchor = (base * 1.5 + .1).detach().clone()
        return apply_head(self.head, features, base, anchor)

    def save_pretrained(self, directory):
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir()
        save_file(self.bridge.state_dict(), str(directory / "bridge.safetensors"))
        save_file(self.head.state_dict(), str(directory / "highlight.safetensors"))
        (directory / "config.json").write_text(json.dumps(dict(
            r3_weights="bridge.safetensors", smooth_weights="highlight.safetensors",
            sha256={name: file_sha256(directory / name) for name in ("bridge.safetensors", "highlight.safetensors")})))


def synthetic_samples():
    generator = np.random.default_rng(2)
    return [prepare_sample(generator.uniform(8, 12, (8, 8, 3)).astype(np.float32), 1., 0,
                           "hard_clip" if index % 2 == 0 else "soft_compress") for index in range(8)]


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_all(0)

    def test_exact_five_term_global_batch_normalization(self):
        samples = [prepare_sample(np.full((2, 2, 3), 3, np.float32), 1., 0, "hard_clip") for _ in range(8)]
        denominators = loss_denominators(samples, "cpu")
        summed = {key: 0. for key in ("HDR", "bright", "grad", "smooth", "over")}
        total = 0.
        for sample in samples:
            tensors = tensors_for_sample(sample, "cpu")
            zero = torch.zeros((1, 1, 2, 2), requires_grad=True)
            result = dict(H_new=torch.full((1, 3, 2, 2), 5.), delta=zero, z=zero)
            loss, terms, undefined = smooth_loss(result, tensors, denominators)
            self.assertFalse(undefined)
            loss.backward()
            total += float(loss.detach())
            for key, value in terms.items():
                summed[key] += value
        expected = math.log(6) - math.log(4)
        self.assertAlmostEqual(summed["HDR"], expected, places=6)
        self.assertAlmostEqual(summed["bright"], expected, places=6)
        self.assertAlmostEqual(summed["over"], expected, places=6)
        self.assertEqual(summed["grad"], 0.)
        self.assertEqual(summed["smooth"], 0.)
        self.assertAlmostEqual(total, 2.25 * expected, places=6)

    def test_head_gradient_boundary_and_frozen_features(self):
        model = ToyFrozenModel()
        before = {name: tensor_digest(getattr(model, name).state_dict()) for name in ("vae", "bridge")}
        optimizer = optimizer_for(model.head)
        scheduler = FixedScheduler(optimizer, total_updates=4)
        samples = synthetic_samples()
        first = update_once(model, optimizer, scheduler, 1, samples, "cpu")
        self.assertEqual(first["parameter_gradient_norms"]["conv.weight"], 0.)
        self.assertGreater(first["parameter_gradient_norms"]["output.weight"], 0.)
        second = update_once(model, optimizer, scheduler, 2, samples, "cpu")
        self.assertGreater(second["parameter_gradient_norms"]["conv.weight"], 0.)
        self.assertGreater(second["parameter_gradient_norms"]["conv.bias"], 0.)
        for name in before:
            module = getattr(model, name)
            self.assertEqual(before[name], tensor_digest(module.state_dict()))
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in module.parameters()))

    def test_safe_full_state_checkpoint_replays_next_update(self):
        model = ToyFrozenModel()
        optimizer = optimizer_for(model.head)
        scheduler = FixedScheduler(optimizer, total_updates=4)
        samples = synthetic_samples()
        update_once(model, optimizer, scheduler, 1, samples, "cpu")
        sequence = make_sequence([dict(source_id="synthetic", split="train")], 4, 0)
        bindings = dict(test="synthetic", updates=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update.pt"
            save_checkpoint(path, model, optimizer, scheduler, bindings=bindings, sequence=sequence)
            expected_random = (random.random(), float(np.random.random()), torch.rand(2))
            expected = update_once(model, optimizer, scheduler, 2, samples, "cpu")
            expected_head = tensor_digest(model.head.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            load_checkpoint(path, model, optimizer, scheduler, bindings=bindings, sequence=sequence)
            actual_random = (random.random(), float(np.random.random()), torch.rand(2))
            self.assertEqual(expected_random[:2], actual_random[:2])
            self.assertTrue(torch.equal(expected_random[2], actual_random[2]))
            actual = update_once(model, optimizer, scheduler, 2, samples, "cpu")
            self.assertEqual(expected, actual)
            self.assertEqual(expected_head, tensor_digest(model.head.state_dict()))
            for key, state in expected_optimizer["state"].items():
                for field, value in state.items():
                    restored = optimizer.state_dict()["state"][key][field]
                    self.assertTrue(torch.equal(value, restored) if torch.is_tensor(value) else value == restored)
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_checkpoint(path, model, optimizer, scheduler, bindings=bindings, sequence=sequence)

    def test_a0_uses_only_strictly_positive_luminance_and_exposure(self):
        rgb = np.array([[[0, 0, 0], [1, 1, 1]], [[4, 4, 4], [0, 0, 0]]], np.float32)
        scale, statistic, count = positive_log_average_scale(rgb)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(statistic, 2., places=6)
        self.assertEqual(scale, np.float32(.09))
        sample = prepare_sample(rgb, scale, 2, "soft_compress")
        target = rgb * scale * np.float32(4)
        np.testing.assert_array_equal(sample["target"], target.transpose(2, 0, 1))
        np.testing.assert_array_equal(sample["base"], (target / (1 + target)).transpose(2, 0, 1))
        with self.assertRaisesRegex(ValueError, "all-black"):
            positive_log_average_scale(np.zeros((8, 8, 3), np.float32))

    def test_training_entrypoint_resumes_interruption_and_exports_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            rows = []
            for index, split in enumerate(("train", "dev")):
                np.save(inputs / f"user_{index}.npy", np.full((8, 8, 3), 1 + index, np.float32))
                rows.append(dict(source_id=f"user_source_{index}", path=f"user_{index}.npy", split=split))
            (inputs / "dataset.json").write_text(json.dumps(dict(schema="gainweave-input-dataset-v1", color_space="linear_rgb", rows=rows)))
            prepare_dataset(inputs, root / "data")
            config = root / "train.yaml"
            text = "schema: gainweave-train-v1\ndataset: manifest.json\nmodel: {pretrained: synthetic}\nseed: 0\nupdates: 2\ncheckpoint_every: 1\noutput_dir: unused\n"
            config.write_text(text)
            with patch("gainweave.GainWeave.from_pretrained", side_effect=lambda *args, **kwargs: ToyFrozenModel()):
                expected = run_training(config, root / "data", root / "uninterrupted", "cpu")
                actual_update = update_once

                def interrupt_second(model, optimizer, scheduler, update, samples, device):
                    if update == 2:
                        raise RuntimeError("synthetic interruption")
                    return actual_update(model, optimizer, scheduler, update, samples, device)

                with patch("gainweave.training.engine.update_once", side_effect=interrupt_second):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        run_training(config, root / "data", root / "resumed", "cpu")
                checkpoint = root / "resumed/checkpoints/update_000001_attempt_0001.pt"
                self.assertTrue(checkpoint.is_file())
                config.write_text(text + f"resume: {checkpoint}\n")
                resumed = run_training(config, root / "data", root / "resumed", "cpu")
                self.assertEqual(expected["head_sha256"], resumed["head_sha256"])
                self.assertTrue((root / "resumed/final/config.json").is_file())
                self.assertFalse(resumed["reference_recipe"])
                with self.assertRaises(FileExistsError):
                    run_training(config, root / "data", root / "resumed", "cpu")

    def test_admission_failure_can_retry_and_partial_export_resumes_without_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            np.save(inputs / "synthetic.npy", np.ones((8, 8, 3), np.float32))
            (inputs / "dataset.json").write_text(json.dumps(dict(schema="gainweave-input-dataset-v1",
                color_space="linear_rgb", rows=[dict(source_id="synthetic", path="synthetic.npy", split="train")])))
            prepare_dataset(inputs, root / "data")
            config = root / "train.yaml"
            text = "schema: gainweave-train-v1\nmodel: {pretrained: synthetic}\nseed: 0\nupdates: 1\ncheckpoint_every: 1\noutput_dir: unused\n"
            config.write_text(text)
            target = root / "output"
            with patch("gainweave.GainWeave.from_pretrained", side_effect=RuntimeError("synthetic loader failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
                    run_training(config, root / "data", target, "cpu")
            self.assertFalse(target.exists())

            def interrupted_export(model, directory):
                directory.mkdir()
                (directory / "bridge.safetensors").write_bytes(b"synthetic partial artifact")
                raise RuntimeError("synthetic export failure")

            with patch("gainweave.GainWeave.from_pretrained", side_effect=lambda *args, **kwargs: ToyFrozenModel()):
                with patch.object(ToyFrozenModel, "save_pretrained", interrupted_export):
                    with self.assertRaisesRegex(RuntimeError, "synthetic export failure"):
                        run_training(config, root / "data", target, "cpu")
                partial = target / "attempts/attempt_0001/final_export/bridge.safetensors"
                self.assertTrue(partial.is_file())
                self.assertFalse((target / "final").exists())
                checkpoint = target / "checkpoints/update_000001_attempt_0001.pt"
                self.assertTrue(checkpoint.is_file())
                config.write_text(text + f"resume: {checkpoint}\n")
                with patch("gainweave.training.engine.update_once", side_effect=AssertionError("No new update is allowed")) as step:
                    result = run_training(config, root / "data", target, "cpu")
                self.assertEqual(step.call_count, 0)
                self.assertEqual(result["completed_update"], 1)
                self.assertTrue((target / "final/config.json").is_file())
                self.assertEqual(partial.read_bytes(), b"synthetic partial artifact")
                self.assertFalse((target / "attempts/attempt_0002/final_export").exists())

    def test_group_split_is_deterministic_and_duplicate_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            rows = []
            for index in range(5):
                np.save(inputs / f"image_{index}.npy", np.full((8, 8, 3), index + 1, np.float32))
                rows.append(dict(source_id=f"source_{index}", path=f"image_{index}.npy"))
            declaration = dict(schema="gainweave-input-dataset-v1", color_space="linear_rgb", rows=rows)
            (inputs / "dataset.json").write_text(json.dumps(declaration))
            prepare_dataset(inputs, root / "first", seed=7)
            prepare_dataset(inputs, root / "second", seed=7)
            first, _ = load_dataset(root / "first")
            second, _ = load_dataset(root / "second")
            self.assertEqual(first, second)
            self.assertEqual({row["split"] for row in first["rows"]}, {"train", "dev"})
            with self.assertRaises(FileExistsError):
                prepare_dataset(inputs, root / "first", seed=7)
            # A byte-identical image cannot masquerade as an independent source.
            np.save(inputs / "duplicate.npy", np.full((8, 8, 3), 1, np.float32))
            declaration["rows"].append(dict(source_id="another_source", path="duplicate.npy"))
            (inputs / "dataset.json").write_text(json.dumps(declaration))
            with self.assertRaisesRegex(ValueError, "Duplicate|identical|leakage|cross source|across source|RGB content"):
                prepare_dataset(inputs, root / "bad")


class EvaluationChecks(unittest.TestCase):
    def test_errors_keep_exposure_and_ignore255_and_empty_support(self):
        reference = np.full((2, 2, 3), 1., np.float32)
        prediction = reference * 2
        mask = np.array([[1, 255], [0, 0]], np.uint8)
        result = evaluate_case(prediction, reference, valid=mask)
        expected = math.log(3) - math.log(2)
        self.assertEqual(result["full"]["support_pixels"], 1)
        self.assertAlmostEqual(result["full"]["MAE"], expected)
        self.assertAlmostEqual(result["full"]["MSE"], expected ** 2)
        self.assertEqual(result["full"]["under"], 0.)
        self.assertIsNone(result["full"]["gradient"])
        self.assertEqual(result["full"]["gradient_edges"], 0)
        self.assertAlmostEqual(result["continuous_bright"]["MAE"], expected)
        empty = evaluate_case(prediction, reference, valid=np.zeros((2, 2), np.uint8))
        self.assertTrue(all(empty["full"][key] is None for key in ("MAE", "MSE", "signed", "over", "under", "gradient")))
        black = evaluate_case(prediction, np.zeros_like(reference))
        self.assertIsNone(black["continuous_bright"]["MAE"])
        self.assertEqual(black["continuous_bright"]["weight_sum"], 0.)
        labels = np.array([[1, 255], [0, 1]], np.uint8)
        masked = evaluate_case(prediction, reference, valid=labels, region=labels)
        self.assertEqual(masked["full"]["support_pixels"], 2)
        self.assertEqual(masked["region"]["support_pixels"], 2)

    def test_gradient_and_signed_decomposition(self):
        reference = np.zeros((2, 2, 3), np.float32)
        prediction = np.zeros_like(reference)
        prediction[0, 0] = 1
        full = evaluate_case(prediction, reference)["full"]
        self.assertEqual(full["gradient_edges"], 4)
        self.assertAlmostEqual(full["gradient"], math.log(2) / 2)
        self.assertAlmostEqual(full["MAE"], full["over"] + full["under"])
        self.assertAlmostEqual(full["signed"], full["over"] - full["under"])

    def test_macro_is_equal_source_not_equal_view_and_has_EV_panels(self):
        cases = []
        ref = np.ones((2, 2, 3), np.float32)
        for source, ev, multiplier in [("A", -1, 2), ("A", 1, 2), ("B", -1, 1)]:
            cases.append(dict(source_id=source, EV=ev, method="example", regions=evaluate_case(ref * multiplier, ref)))
        aggregates = aggregate_cases(cases)
        all_sources = next(row for row in aggregates if row["region"] == "full" and row["EV"] == "ALL")
        self.assertAlmostEqual(all_sources["equal_source"]["MAE"], (math.log(3) - math.log(2)) / 2)
        self.assertEqual(all_sources["total_sources"], 2)
        self.assertTrue({"-1", "1", "ALL"}.issubset({row["EV"] for row in aggregates}))

    def test_manifest_runner_requires_named_linear_data_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "reference.npy", np.ones((2, 2, 3), np.float32))
            np.savez(root / "prediction.npz", hdr_linear_rgb=np.full((2, 2, 3), 2, np.float32))
            declaration = dict(schema="gainweave-paired-evaluation-v1", color_space="linear_rgb",
                               rows=[dict(source_id="synthetic", prediction="prediction.npz", reference="reference.npy", EV=0)])
            (root / "pairs.json").write_text(json.dumps(declaration))
            config = root / "evaluate.yaml"
            config.write_text("schema: gainweave-eval-v1\nmanifest: pairs.json\noutput_dir: scores\n")
            result = run_evaluation(config)
            report = json.loads(Path(result["report"]).read_text())
            self.assertEqual(result["pairs"], 1)
            self.assertGreater(report["cases"][0]["regions"]["full"]["MAE"], 0)
            with self.assertRaises(FileExistsError):
                run_evaluation(config)
            np.savez(root / "prediction.npz", unknown_array=np.ones((2, 2, 3), np.float32))
            config.write_text("schema: gainweave-eval-v1\nmanifest: pairs.json\noutput_dir: other_scores\n")
            with self.assertRaisesRegex(ValueError, "hdr_linear_rgb"):
                run_evaluation(config)


if __name__ == "__main__":
    unittest.main()
