"""No network, CUDA, process control or model loading in these preflight tests."""
import importlib.metadata as metadata
import json
from pathlib import Path
import tempfile
import unittest

from scripts.manage_checks import configure, environment


class ManagePreflightTests(unittest.TestCase):
    def test_valid_cuda_environment_is_reusable_and_version_drift_is_detected(self):
        with tempfile.TemporaryDirectory(prefix="FaceShield environment ") as temporary:
            root = Path(temporary)
            (root / "requirements-windows.txt").write_text("torch==2.7.1\ntorchvision==0.22.1\nnumpy==1.26.4\nscipy>=1.10 # range\n", encoding="utf-8")
            versions = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "numpy": "1.26.4", "scipy": "1.14.0"}
            self.assertTrue(environment(root, versions.__getitem__, (3, 11, 9))["ready"])
            versions["torch"] = "2.7.1+cpu"
            report = environment(root, versions.__getitem__, (3, 11, 9))
            self.assertFalse(report["ready"])
            self.assertFalse(report["torch_ready"])
            self.assertTrue(report["requirements_ready"], "CUDA wheel selection is checked separately from PEP440 requirements.")
            versions["torch"] = "2.7.1+cu128"
            versions["numpy"] = "2.0.0"
            self.assertFalse(environment(root, versions.__getitem__, (3, 11))["requirements_ready"])
            self.assertFalse(environment(root, versions.__getitem__, (3, 12))["python_ready"])

    def test_missing_package_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "requirements-windows.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
            def missing(name):
                raise metadata.PackageNotFoundError(name)
            result = environment(root, missing, (3, 11))
            self.assertFalse(result["ready"])
            self.assertTrue(any("numpy: missing" in item for item in result["errors"]))

    def fixture(self, root, count=214):
        checkpoints = root / "External models with spaces" / "checkpoints"
        for filename in ("simswap/arcface_checkpoint.tar", "faceshifter/model_ir_se50.pth", "hififace/hififace_opensource_299999.ckpt"):
            path = checkpoints / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture, never deserialized")
        cloaks = root / "External data" / "mix_full"
        cloaks.mkdir(parents=True)
        for index in range(count):
            (cloaks / f"{index}.jpg").write_bytes(b"fixture, not decoded")
        return checkpoints, cloaks

    def test_configure_writes_only_paths_and_keeps_external_assets(self):
        with tempfile.TemporaryDirectory(prefix="FaceShield paths ") as temporary:
            root = Path(temporary)
            checkpoints, cloaks = self.fixture(root)
            configure(root, checkpoints, cloaks)
            config = json.loads((root / "phantomseal_config.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(config["checkpoint_dir"]), checkpoints)
            self.assertEqual(Path(config["cloak_dir"]), cloaks)
            self.assertEqual(config["expected_cloaks"], 214)
            self.assertEqual(len(list(cloaks.glob("*.jpg"))), 214)
            self.assertTrue((checkpoints / "hififace/hififace_opensource_299999.ckpt").is_file())
            self.assertFalse(list(root.glob(".phantomseal-config-*.tmp")))

    def test_partial_bank_or_checkpoint_cannot_replace_existing_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints, cloaks = self.fixture(root, count=213)
            target = root / "phantomseal_config.json"
            target.write_text('{"keep":"existing"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "214"):
                configure(root, checkpoints, cloaks)
            self.assertEqual(target.read_text(), '{"keep":"existing"}')
            (cloaks / "last.png").write_bytes(b"fixture")
            (checkpoints / "faceshifter/model_ir_se50.pth").unlink()
            with self.assertRaisesRegex(ValueError, "checkpoints"):
                configure(root, checkpoints, cloaks)
            self.assertEqual(target.read_text(), '{"keep":"existing"}')


if __name__ == "__main__":
    unittest.main()
