"""Reproducibility contracts for suite inputs, cache reuse and JPEG condition."""
import json
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_suite import baseline_cutoff, condition_image, environment_fingerprint, read_manifest, reusable_preparation, run
from benchmark_chain import sha256_file


class SuiteTests(unittest.TestCase):
    def test_environment_hashes_central_models_and_detector_without_legacy_weight_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["weights/simswap/latest_net_G.pth", "weights/simswap/arcface_checkpoint.tar",
                     "weights/facenet/facenet_vggface2.pt", "weights/faceshifter/model_ir_se50.pth",
                     "weights/hififace/hififace_opensource_299999.ckpt", "weights/ip-adapter/ip-adapter_sd15.bin",
                     "weights/faceshield/arcface50_checkpoint.tar", "weights/faceshield/arcface100_checkpoint.tar",
                     "weights/mtcnn/pnet.pt", "weights/mtcnn/rnet.pt", "weights/mtcnn/onet.pt",
                     "weights/sd15/unet/model.safetensors", "models/arcface_models.py",
                     "protection_pipeline.py", "protector.py", "phantomseal_runtime.py",
                     "phantomseal_assets.py", "ensemble_encoders.py", "asset_paths.py"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            (root / "phantomseal_config.json").write_text("{}")
            old = root / ".cache/models/unused.safetensors"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"old location")
            with patch("benchmark_suite.importlib.metadata.version", return_value="fixture"):
                first = environment_fingerprint(root)
                detector = root / "weights/mtcnn/pnet.pt"
                detector.write_bytes(b"changed detector")
                second = environment_fingerprint(root)
            self.assertNotIn(str(old), first["files"])
            self.assertIn(str(root / "asset_paths.py"), first["files"])
            self.assertIn(str(root / "models/arcface_models.py"), first["files"])
            self.assertNotEqual(first["files"][str(detector)], second["files"][str(detector)])

    def test_baseline_gate_is_finite_in_range_and_validated_before_any_model_work(self):
        self.assertEqual(baseline_cutoff(0), 0.)
        self.assertEqual(baseline_cutoff(.8), .8)
        self.assertEqual(baseline_cutoff(1), 1.)
        for value in (True, False, None, "0.8", -.001, 1.001, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "minimum_baseline"):
                    baseline_cutoff(value)
                # No other arguments or runtime setup exist: a bad cutoff must
                # fail before reading even root, let alone loading GPU models.
                with self.assertRaisesRegex(ValueError, "minimum_baseline"):
                    run(Namespace(minimum_baseline=value))

    def test_manifest_rejects_duplicate_ids_and_wrong_target_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (224, 224)).save(root / "image.png")
            source = {"id": "person", "path": "image.png"}
            payload = {"schema_version": 1, "sources": [source, source],
                       "targets": [{"id": "target", "path": "image.png"}], "params": {}}
            path = root / "manifest.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_manifest(path, root)
            payload["sources"] = [source]
            Image.new("RGB", (223, 224)).save(root / "image.png")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "224x224"):
                read_manifest(path, root)

    def test_resume_requires_exact_fingerprint_and_output_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "stage1").mkdir()
            names = ("original.png", "stage1/phantomseal.png", "protected.png")
            for name in names:
                (folder / name).write_bytes(b"verified image content")
            prepared = {"fingerprint": "test", "outputs": {name: sha256_file(folder / name) for name in names}}
            (folder / "prepared.json").write_text(json.dumps(prepared))
            self.assertIsNotNone(reusable_preparation(folder, "test"))
            self.assertIsNone(reusable_preparation(folder, "changed parameters or weights"))
            (folder / "protected.png").write_bytes(b"modified result")
            self.assertIsNone(reusable_preparation(folder, "test"))

    def test_jpeg_condition_is_lossy_same_dimensions_and_does_not_modify_input(self):
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, size=(93, 127, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        png = condition_image(image, "png")
        jpeg = condition_image(image, "jpeg95")
        self.assertEqual(jpeg.size, image.size)
        self.assertEqual(jpeg.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(image), pixels)
        np.testing.assert_array_equal(np.asarray(png), pixels)
        self.assertGreater(np.abs(np.asarray(jpeg).astype(float) - pixels).mean(), 1)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            condition_image(image, "jpeg50")


if __name__ == "__main__":
    unittest.main()
