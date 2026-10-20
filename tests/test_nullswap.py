"""Bounds and geometry of the native NullSwap residual adaptation."""
import tempfile
import json
import math
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch

from nullswap_runtime import EPSILON, NullSwapProtector, map_residual, readiness


class NullSwapTests(unittest.TestCase):
    def test_readiness_requires_local_checkpoint_and_detector_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(readiness(root)["ready"])
            for rel in ("nullswap/epoch_0001_step_012000.pth", "mtcnn/pnet.pt", "mtcnn/rnet.pt", "mtcnn/onet.pt"):
                path = root / "weights" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            self.assertTrue(readiness(root)["ready"])

    def test_residual_mapping_preserves_full_canvas_and_eight_pixel_bound(self):
        clean = np.full((720, 1280, 3), 128, np.uint8)
        delta = np.zeros((256, 256, 3), np.float32)
        delta[:, :, 0], delta[:, :, 1] = EPSILON, -EPSILON
        # Native 256px patch occupies only a 512px square in the full canvas.
        matrix = np.array([[.5, 0, -150], [0, .5, -50]], dtype=np.float32)
        output = map_residual(clean, delta, matrix)
        self.assertEqual(output.shape, clean.shape)
        self.assertEqual(output[300, 500].tolist(), [136, 120, 128])
        self.assertEqual(output[0, 0].tolist(), [128, 128, 128])
        self.assertLessEqual(int(np.abs(output.astype(np.int16) - clean).max()), 8)

    def test_zero_residual_keeps_original_pixels_and_invalid_delta_fails(self):
        clean = np.random.default_rng(1).integers(0, 256, (200, 300, 3), dtype=np.uint8)
        transform = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        zero = np.zeros((256, 256, 3), np.float32)
        np.testing.assert_array_equal(clean, map_residual(clean, zero, transform))
        for invalid in (np.full_like(zero, np.nan), np.full_like(zero, EPSILON * 2), zero[:20]):
            with self.assertRaises(RuntimeError):
                map_residual(clean, invalid, transform)
        with self.assertRaises(ValueError):
            map_residual(clean, zero, np.zeros((2, 3), np.float32))

    def test_bad_parameters_fail_before_loading(self):
        engine = NullSwapProtector(".", "cpu")
        engine.load_models = Mock(side_effect=AssertionError("must not load"))
        invalid = [{"max_size": 1281}]
        invalid += [{"strength": value} for value in (-1, 4.01, True, "1", float("nan"), float("inf"))]
        invalid += [{"seed": value} for value in (-1, 2**32, 1.5, True, "0", float("nan"))]
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                engine.protect("unused.png", "unused-output", **params)
        engine.load_models.assert_not_called()

    def test_strength_scales_only_residual_with_zero_fractional_and_max_bounds(self):
        clean = np.full((300, 400, 3), 128, np.uint8)
        delta = np.full((256, 256, 3), EPSILON, np.float32)
        transform = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        for strength in (0, .1, .5, 1, 2.75, 4):
            with self.subTest(strength=strength):
                output = map_residual(clean, delta, transform, strength=strength)
                self.assertEqual(output[100, 100, 0], round(128 + 8 * strength))
                np.testing.assert_array_equal(output[280:, :, :], clean[280:, :, :])
                self.assertLessEqual(np.abs(output.astype(int) - clean).max(), math.ceil(8 * strength))
                if strength == 0:
                    np.testing.assert_array_equal(output, clean)

    def test_seed_is_reproducible_local_and_persisted_with_strength(self):
        class RandomModel:
            def __call__(self, value):
                return {"cloak": (value + (torch.rand_like(value) * 2 - 1) * EPSILON).clamp(0, 1)}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.png"
            Image.new("RGB", (128, 128), (120, 130, 140)).save(source)
            engine = NullSwapProtector(root, torch.device("cpu"))
            engine.model, engine.detector = RandomModel(), Mock()
            points = np.array([[[35, 40], [90, 40], [64, 65], [40, 90], [85, 90]]], np.float32)
            engine.detector.detect.return_value = (np.array([[10, 10, 110, 110]]), np.array([.99]), points)
            before = torch.random.get_rng_state().clone()
            outputs = []
            for index, seed in enumerate((19, 19, 4294967295)):
                result = engine.protect(source, root / str(index), strength=2.75, seed=seed)
                with Image.open(result["output_path"]) as image:
                    outputs.append(np.asarray(image).copy())
                saved = json.loads((root / str(index) / "nullswap_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["seed"], seed)
                self.assertEqual(saved["perturbation_gain"], 2.75)
                self.assertEqual(saved["noise_limit"], 22)
                self.assertLessEqual(saved["max_pixel_change"], 22)
            np.testing.assert_array_equal(outputs[0], outputs[1])
            self.assertTrue(np.any(outputs[0] != outputs[2]))
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_protect_keeps_landscape_size_and_reports_gain_one(self):
        class DummyModel:
            def __call__(self, value):
                return {"cloak": (value + .01).clamp(0, 1)}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.png"
            Image.new("RGB", (800, 600), (120, 130, 140)).save(source)
            engine = NullSwapProtector(root, "cpu")
            engine.device = torch.device("cpu")
            engine.model = DummyModel()
            engine.detector = Mock()
            engine.detector.detect.return_value = (
                np.array([[100, 80, 500, 550]], dtype=object), np.array([.99]),
                np.array([[[200, 200], [400, 200], [300, 300], [220, 400], [380, 400]]], np.float32))
            steps = []
            result = engine.protect(source, root / "output", progress=lambda p, text: steps.append(p))
            with Image.open(result["output_path"]) as actual:
                self.assertEqual(actual.size, (800, 600))
            self.assertEqual(result["perturbation_gain"], 1)
            self.assertEqual(result["noise_limit"], 8)
            self.assertEqual(result["seed"], 0)
            self.assertGreater(result["max_pixel_change"], 0)
            self.assertLessEqual(result["max_pixel_change"], 8)
            self.assertEqual(steps, sorted(steps))
            self.assertEqual(steps[-1], 1)

    def test_missing_face_fails_without_returning_original_as_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.png"
            Image.new("RGB", (128, 128)).save(source)
            engine = NullSwapProtector(folder, torch.device("cpu"))
            engine.model = Mock()
            engine.detector = Mock()
            engine.detector.detect.return_value = (None, None, None)
            with self.assertRaisesRegex(ValueError, "No clear face"):
                engine.protect(source, Path(folder) / "output")
            self.assertFalse((Path(folder) / "output" / "nullswap.png").exists())
            engine.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
