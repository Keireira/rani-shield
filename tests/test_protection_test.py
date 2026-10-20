"""Actual orchestration contracts with fake CPU benchmark/model contexts."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from PIL import Image

from protection_test import TARGETS, run_protection_test, testing_readiness


class ProtectionTestRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        targets = self.root / "data/phantomseal_cloaks"
        targets.mkdir(parents=True)
        for value in TARGETS.values():
            Image.new("RGB", (224, 224), (40, 80, 120)).save(targets / value["file"])
        pixels = np.random.default_rng(5).integers(30, 220, (64, 96, 3), dtype=np.uint8)
        self.paths = {}
        for key, change in (("clean", 0), ("stage1", 3), ("final", 7)):
            path = self.root / (key + ".png")
            Image.fromarray(pixels + change).save(path)
            self.paths[key] = path
        self.calls, self.events, self.model_refs = [], [], []
        self.fail_call = None

    def tearDown(self):
        self.temporary.cleanup()

    @contextmanager
    def models(self, root):
        class FakeModel:
            pass
        model = FakeModel()
        self.model_refs.append(weakref.ref(model))
        self.events.append("load")
        try:
            yield (model,)
        finally:
            self.events.append("close")

    def benchmark(self, root, clean, stage1, final, target, output, *, models, reference_path):
        self.calls.append({"clean": clean, "stage1": stage1, "final": final,
                           "target": target, "reference": reference_path, "models_id": id(models)})
        if self.fail_call == len(self.calls):
            raise RuntimeError("fake benchmark failed")
        folder = output.parent / (output.stem + "_images")
        folder.mkdir()
        rows = {}
        for key, source in (("clean", clean), ("stage1", stage1), ("final", final)):
            path = folder / (key + "_swap.png")
            Image.new("RGB", (224, 224), (50, 90, 130)).save(path)
            rows[key] = {"output_path": str(path), "source_face_detected": True,
                         "swap_face_detected": True, "status": "ok",
                         "facenet_cosine_to_clean_source": {"clean": .9, "stage1": .3, "final": .2}[key]}
        return {"rows": rows}

    def run_test(self, **kwargs):
        return run_protection_test(root_dir=self.root, output_dir=self.root / "results/test",
            source_path=self.paths["clean"], result_path=self.paths["final"], stage1_path=self.paths["stage1"],
            benchmark_runner=self.benchmark, models_context=self.models, **kwargs)

    def test_two_conditions_share_one_model_context_and_original_png_reference(self):
        result = self.run_test()
        self.assertEqual(self.events, ["load", "close"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0]["models_id"], self.calls[1]["models_id"])
        self.assertTrue(all(call["reference"] == self.paths["clean"] for call in self.calls))
        self.assertEqual([condition["name"] for condition in result["conditions"]], ["png", "jpeg95"])
        for condition in result["conditions"]:
            self.assertEqual([row["key"] for row in condition["rows"]], ["clean", "stage1", "final"])
            self.assertAlmostEqual(condition["delta_final_minus_stage1"], -.1)
        for key in ("clean", "stage1", "final"):
            with Image.open(self.paths[key]) as original, Image.open(self.calls[0][key]) as png, Image.open(self.calls[1][key]) as jpeg:
                np.testing.assert_array_equal(np.asarray(png), np.asarray(original))
                self.assertEqual(jpeg.size, original.size)
                self.assertGreater(float(np.abs(np.asarray(jpeg).astype(float) - np.asarray(original)).mean()), 1)
        self.assertTrue(all(reference() is None for reference in self.model_refs))

    def test_equivalent_input_paths_use_the_same_canonical_reference(self):
        self.paths = {name: path.parent / "data" / ".." / path.name for name, path in self.paths.items()}
        self.run_test(jpeg95=False)
        self.assertEqual(self.calls[0]["reference"], self.paths["clean"].resolve())

    @unittest.skipUnless(os.name == "nt", "Windows 8.3 path aliases")
    def test_windows_short_alias_keeps_original_clean_reference(self):
        import ctypes
        folder = self.root / "Long source image fixture directory"
        folder.mkdir()
        for name, path in self.paths.items():
            with Image.open(path) as source:
                source.save(folder / path.name)
        buffer = ctypes.create_unicode_buffer(32768)
        size = ctypes.windll.kernel32.GetShortPathNameW(str(folder), buffer, len(buffer))
        if not size or size >= len(buffer) or Path(buffer.value) == folder:
            self.skipTest("This volume does not provide an 8.3 alias for the fixture.")
        alias = Path(buffer.value)
        self.assertEqual(alias.resolve(), folder)
        self.paths = {name: alias / path.name for name, path in self.paths.items()}
        self.run_test(jpeg95=False)
        self.assertEqual(self.calls[0]["reference"], folder / "clean.png")

    def test_single_method_uses_final_internally_and_exposes_only_clean_final(self):
        result = self.run_test(mode="faceshield", jpeg95=False)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["stage1"], self.calls[0]["final"])
        condition = result["conditions"][0]
        self.assertEqual([row["key"] for row in condition["rows"]], ["clean", "final"])
        self.assertEqual(condition["rows"][-1]["label"], "FaceShield")
        self.assertIsNone(condition["delta_final_minus_stage1"])

    def test_all_backends_keep_original_reference_and_nullswap_label(self):
        for backend in ("simswap", "diffswap", "infoswap", "uniface"):
            with self.subTest(backend=backend):
                # Separate outputs, as the fixture intentionally refuses overwrites.
                result = run_protection_test(
                    root_dir=self.root, output_dir=self.root / backend,
                    source_path=self.paths["clean"], result_path=self.paths["final"],
                    mode="nullswap", backend=backend, jpeg95=False,
                    benchmark_runner=self.benchmark, models_context=self.models)
                self.assertEqual(result["backend"], backend)
                self.assertIn("NullSwap", result["conditions"][0]["rows"][-1]["label"])
                self.assertEqual(self.calls[-1]["reference"], self.paths["clean"])

    def test_invalid_backend_rejected_before_loading_models(self):
        for backend in ("bad", "../simswap", True, None, []):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                self.run_test(backend=backend)
        self.assertEqual(self.events, [])

    def test_failure_exits_shared_model_context_and_releases_its_last_reference(self):
        self.fail_call = 2
        try:
            self.run_test()
        except RuntimeError as error:
            self.assertIn("fake benchmark failed", str(error))
        else:
            self.fail("Expected the fake benchmark failure")
        self.assertEqual(self.events, ["load", "close"])
        self.assertTrue(all(reference() is None for reference in self.model_refs))

    def test_missing_generator_is_explicit_in_lightweight_readiness(self):
        for name in ("benchmark_chain.py", "benchmark_alignment.py", "third_party/phantomseal/simswap_networks.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with patch("phantomseal_assets.ensemble_readiness", return_value={"ready": True, "message": "Ready"}):
            status = testing_readiness(self.root)
        self.assertFalse(status["ready"])
        self.assertIn("latest_net_G.pth", status["message"])


if __name__ == "__main__":
    unittest.main()
