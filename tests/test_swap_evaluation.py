"""Shared evaluation keeps references, detection gates and stochastic inputs fair."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from swap_evaluation import benchmark, swap_models
from swap_adapters import FaceAlignmentError


class Evaluator:
    def detect(self, image):
        if image.getpixel((0, 0))[0] == 0:
            raise ValueError("No face")
        return np.array([0, 0, 16, 16]), np.ones((5, 2)), 1

    def face_embedding(self, image, box):
        return torch.tensor([[float(image.getpixel((0, 0))[0]), 100.]])


class Swapper:
    preprocessing = {"source": "fixture"}

    def __init__(self, output=80):
        self.calls, self.output = [], output

    def swap(self, source, target, seed):
        self.calls.append((source.size, target.tobytes(), seed))
        return Image.new("RGB", (256, 256), (self.output, 20, 30))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.paths = {}
        for name, color, size in (("clean", 90, (128, 64)), ("final", 70, (128, 64)),
                                  ("reference", 110, (128, 64)), ("target", 40, (224, 224))):
            self.paths[name] = self.root / f"{name}.png"
            Image.new("RGB", size, (color, 20, 30)).save(self.paths[name])
        self.swapper = Swapper()

    def evaluate(self, **kwargs):
        return benchmark(self.root, self.paths["clean"], self.paths["final"], self.paths["final"],
                         self.paths["target"], self.root / "report.json", backend="uniface",
                         models=(Evaluator(), self.swapper), reference_path=self.paths["reference"], **kwargs)

    def test_fixed_reference_seed_target_native_geometry_and_duplicate_reuse(self):
        report = self.evaluate()
        self.assertEqual(len(self.swapper.calls), 2)
        self.assertEqual(self.swapper.calls[0], self.swapper.calls[1])
        self.assertEqual(self.swapper.calls[0][0], (128, 64))
        self.assertEqual(self.swapper.calls[0][2], 0)
        expected = torch.nn.functional.cosine_similarity(
            torch.tensor([[80., 100.]]), torch.tensor([[110., 100.]]), dim=1).item()
        for row in report["rows"].values():
            self.assertAlmostEqual(row["facenet_cosine_to_clean_source"], expected)
            with Image.open(row["output_path"]) as image:
                self.assertEqual(image.size, (256, 256))
        self.assertEqual(report["method"]["generator_input"], self.swapper.preprocessing)

    def test_failed_source_or_output_detection_is_not_a_score(self):
        self.swapper.output = 0
        Image.new("RGB", (128, 64), (0, 20, 30)).save(self.paths["final"])
        report = self.evaluate()
        self.assertEqual(len(self.swapper.calls), 1)
        self.assertEqual(report["rows"]["clean"]["status"], "swap_detection_failed")
        self.assertEqual(report["rows"]["final"]["status"], "source_detection_failed")
        self.assertTrue(all(row["facenet_cosine_to_clean_source"] is None for row in report["rows"].values()))
        self.assertEqual(report["final_vs_stage1"]["baseline_status"], "inconclusive_weak_baseline")

    def test_partial_load_failure_closes_both_models(self):
        with patch("face_evaluation.FaceEvaluator") as evaluator, patch("swap_evaluation.create_adapter") as create, \
                patch("protection_pipeline.release_gpu") as release:
            create.return_value.load.side_effect = RuntimeError("bad checkpoint")
            with self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                with swap_models(self.root, "uniface"):
                    self.fail("Load should fail")
            create.return_value.close.assert_called_once()
            evaluator.return_value.close.assert_called_once()
            release.assert_called_once()

    def test_invalid_cutoff_fails_without_swap(self):
        with self.assertRaises(ValueError):
            self.evaluate(minimum_baseline=float("nan"))
        self.assertEqual(self.swapper.calls, [])

    def test_backend_detection_failure_is_explicit_but_other_errors_propagate(self):
        with patch.object(self.swapper, "swap", side_effect=FaceAlignmentError("No native landmarks")):
            report = self.evaluate()
        self.assertTrue(all(row["status"] == "backend_alignment_failed" for row in report["rows"].values()))
        self.assertTrue(all(row["facenet_cosine_to_clean_source"] is None for row in report["rows"].values()))
        with patch.object(self.swapper, "swap", side_effect=RuntimeError("CUDA error")):
            with self.assertRaisesRegex(RuntimeError, "CUDA error"):
                self.evaluate()


if __name__ == "__main__":
    unittest.main()
