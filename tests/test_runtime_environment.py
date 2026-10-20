"""Portable setup checks use tiny fixtures and never start a server or download."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.bootstrap_runtime import REQUIRED_FILES, install_node, install_python
from scripts.runtime_environment import CACHE_PATHS, configure_environment, environment_values


class PortableSetupTests(unittest.TestCase):
    def test_python_copy_is_independent_excludes_packages_and_reuses_identical_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"; project = root / "project"; project.mkdir()
            for name in REQUIRED_FILES:
                path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("fixture " + name).encode())
            for name in ("Lib/site-packages/external.py", "Lib/__pycache__/os.pyc"):
                path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"exclude")
            first = install_python(source, project)
            self.assertEqual(first["copied"], len(REQUIRED_FILES))
            runtime = project / ".runtime/python"
            self.assertFalse((runtime / "Lib/site-packages").exists())
            self.assertFalse((runtime / "Lib/__pycache__").exists())
            self.assertEqual(install_python(source, project)["copied"], 0)
            original = (runtime / "Lib/os.py").read_bytes()
            (source / "Lib/os.py").write_bytes(b"changed source")
            self.assertEqual((runtime / "Lib/os.py").read_bytes(), original)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                install_python(source, project)
            report = json.loads((runtime / "manifest.json").read_text())
            self.assertTrue(all(row["path"].startswith(".runtime/python/") for row in report["files"]))

    def test_node_copy_requires_and_retains_license(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"; source.mkdir()
            executable = source / "node.exe"; executable.write_bytes(b"node fixture")
            with self.assertRaisesRegex(ValueError, "LICENSE"):
                install_node(executable, root)
            license_path = root / "downloaded-license"; license_path.write_text("Node license fixture")
            install_node(executable, root, license_path)
            self.assertEqual((root / ".runtime/node/LICENSE").read_text(), "Node license fixture")
            self.assertEqual((root / ".runtime/node/node.exe").read_bytes(), b"node fixture")

    def test_environment_replaces_ambient_paths_and_tempfile_cache(self):
        previous_tempdir = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                values = environment_values(root)
                self.assertEqual(list(root.iterdir()), [], "Querying env must be read-only.")
                with patch.dict(os.environ, {"HF_HOME": "outside", "PYTHONPATH": "outside",
                                           "TRANSFORMERS_CACHE": "outside"}):
                    configure_environment(root)
                    for name in CACHE_PATHS:
                        self.assertEqual(os.environ[name], values[name])
                        self.assertTrue(Path(os.environ[name]).is_relative_to(root))
                    self.assertNotIn("PYTHONPATH", os.environ)
                    self.assertNotIn("TRANSFORMERS_CACHE", os.environ)
                    self.assertEqual(Path(tempfile.gettempdir()), root / ".cache/tmp")
        finally:
            tempfile.tempdir = previous_tempdir


if __name__ == "__main__":
    unittest.main()
