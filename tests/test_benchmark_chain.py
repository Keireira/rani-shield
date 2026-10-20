"""CPU-only validation of benchmark comparisons and input invariants."""
import importlib.util
import unittest
from pathlib import Path

from PIL import Image


SPEC = importlib.util.spec_from_file_location(
    "benchmark_chain", Path(__file__).resolve().parents[1] / "benchmark_chain.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_preserves_same_size_source_and_requires_fixed_target(self):
        image = Image.new("RGB", (1280, 853))
        inputs = {name: image for name in ("clean", "stage1", "final")}
        benchmark.validate_images(inputs, Image.new("RGB", (224, 224)))
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            benchmark.validate_images({**inputs, "final": Image.new("RGB", (512, 341))}, Image.new("RGB", (224, 224)))
        with self.assertRaisesRegex(ValueError, "224x224"):
            benchmark.validate_images(inputs, Image.new("RGB", (512, 512)))

    def test_delta_direction_uses_same_clean_baseline(self):
        rows = {
            "clean": {"facenet_cosine_to_clean_source": .9, "source_face_detected": True, "swap_face_detected": True},
            "stage1": {"facenet_cosine_to_clean_source": 0.10, "source_face_detected": True, "swap_face_detected": True},
            "final": {"facenet_cosine_to_clean_source": 0.35, "source_face_detected": True, "swap_face_detected": True},
        }
        result = benchmark.comparison(rows)
        self.assertAlmostEqual(result["final_minus_stage1_cosine"], 0.25)
        self.assertEqual(result["direction"], "higher_clean_identity_similarity")
        self.assertTrue(result["both_swap_faces_detected"])

    def test_detection_failure_remains_null_without_forced_crop_score(self):
        result = benchmark.comparison({
            "stage1": {"facenet_cosine_to_clean_source": None, "swap_face_detected": False},
            "final": {"facenet_cosine_to_clean_source": .42, "swap_face_detected": True},
        })
        self.assertIsNone(result["final_minus_stage1_cosine"])
        self.assertIsNone(result["direction"])
        self.assertFalse(result["both_swap_faces_detected"])

    def test_weak_unprotected_swap_does_not_support_a_defense_verdict(self):
        rows = {key: {"facenet_cosine_to_clean_source": score, "source_face_detected": True,
                      "swap_face_detected": True} for key, score in
                (("clean", .6806), ("stage1", .7102), ("final", .7227))}
        result = benchmark.comparison(rows)
        self.assertEqual(result["baseline_status"], "inconclusive_weak_baseline")
        self.assertIsNone(result["direction"])
        self.assertIsNone(result["final_minus_stage1_cosine"])
        self.assertAlmostEqual(result["raw_final_minus_stage1_cosine"], .0125)
        self.assertEqual(result["minimum_baseline_cosine"], .8)

    def test_quality_cutoff_is_configurable_and_invalid_values_are_rejected(self):
        rows = {key: {"facenet_cosine_to_clean_source": score, "source_face_detected": True,
                      "swap_face_detected": True} for key, score in
                (("clean", .85), ("stage1", .3), ("final", .2))}
        self.assertEqual(benchmark.comparison(rows, .8)["baseline_status"], "numeric_gate_met_visual_review_required")
        self.assertIsNone(benchmark.comparison(rows, .9)["direction"])
        for cutoff in (float("nan"), -.1, 1.01, True):
            with self.assertRaises(ValueError):
                benchmark.comparison(rows, cutoff)


if __name__ == "__main__":
    unittest.main()
