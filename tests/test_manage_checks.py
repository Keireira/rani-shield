"""No network, CUDA, process control or model loading in these preflight tests."""
import importlib.metadata as metadata
import json
from pathlib import Path
import tempfile
import unittest

from scripts.manage_checks import configure, environment, runtime_paths


class ManagePreflightTests(unittest.TestCase):
    def test_valid_cuda_environment_is_reusable_and_version_drift_is_detected(self):
        with tempfile.TemporaryDirectory(prefix="FaceShield environment ") as temporary:
            root = Path(temporary).resolve()
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
            root = Path(temporary).resolve()
            (root / "requirements-windows.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
            def missing(name):
                raise metadata.PackageNotFoundError(name)
            result = environment(root, missing, (3, 11))
            self.assertFalse(result["ready"])
            self.assertTrue(any("numpy: missing" in item for item in result["errors"]))

    def fixture(self, root, count=214):
        checkpoints = root / "Local models with spaces" / "checkpoints"
        for filename in ("simswap/arcface_checkpoint.tar", "faceshifter/model_ir_se50.pth", "hififace/hififace_opensource_299999.ckpt"):
            path = checkpoints / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture, never deserialized")
        cloaks = root / "Local data" / "mix_full"
        cloaks.mkdir(parents=True)
        for index in range(count):
            (cloaks / f"{index}.jpg").write_bytes(b"fixture, not decoded")
        return checkpoints, cloaks

    def test_configure_writes_portable_relative_paths_and_preserves_local_assets(self):
        with tempfile.TemporaryDirectory(prefix="FaceShield paths ") as temporary:
            root = Path(temporary).resolve()
            checkpoints, cloaks = self.fixture(root)
            configure(root, checkpoints, cloaks)
            config = json.loads((root / "phantomseal_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["checkpoint_dir"], checkpoints.relative_to(root).as_posix())
            self.assertEqual(config["cloak_dir"], cloaks.relative_to(root).as_posix())
            self.assertEqual(config["expected_cloaks"], 214)
            self.assertEqual(len(list(cloaks.glob("*.jpg"))), 214)
            self.assertTrue((checkpoints / "hififace/hififace_opensource_299999.ckpt").is_file())
            self.assertFalse(list(root.glob(".phantomseal-config-*.tmp")))

    def test_partial_bank_or_checkpoint_cannot_replace_existing_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
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

    def test_external_configuration_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            root = folder / "project"; root.mkdir()
            checkpoints, cloaks = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, "inside the project"):
                configure(root, checkpoints, cloaks)
            self.assertEqual(list(root.iterdir()), [])

    def test_custom_checkpoint_directory_needs_its_own_simswap_before_replacing_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoints, cloaks = self.fixture(root)
            (checkpoints / "simswap/arcface_checkpoint.tar").unlink()
            central = root / "weights/simswap/arcface_checkpoint.tar"
            central.parent.mkdir(parents=True)
            central.write_bytes(b"central model must not satisfy a different configured directory")
            target = root / "phantomseal_config.json"
            target.write_text('{"keep":"existing"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing ensemble checkpoints"):
                configure(root, checkpoints, cloaks)
            self.assertEqual(target.read_text(), '{"keep":"existing"}')

    def test_runtime_preflight_detects_external_base_search_path_and_pth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runtime = dict(executable=root / ".venv/Scripts/python.exe",
                           base_prefix=root / ".runtime/python",
                           search_paths=[str(root), str(root / ".runtime/python/Lib")])
            self.assertTrue(runtime_paths(root, **runtime)["ready"])
            report = runtime_paths(root, **{**runtime, "base_prefix": root.parent / "foreign"})
            self.assertFalse(report["ready"])
            self.assertIn("base_prefix", report["errors"][0])
            report = runtime_paths(root, **{**runtime, "search_paths": [str(root.parent / "foreign-packages")]})
            self.assertFalse(report["ready"])
            site = root / ".venv/Lib/site-packages"; site.mkdir(parents=True)
            (site / "external.pth").write_text(str(root.parent / "foreign-package"), encoding="utf-8")
            self.assertFalse(runtime_paths(root, **runtime)["ready"])


if __name__ == "__main__":
    unittest.main()
