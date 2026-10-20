"""Small installer checks; no network, model loading, or GPU."""
import importlib.util
import json
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "download_phantomseal", Path(__file__).resolve().parents[1] / "download_phantomseal.py"
)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def temporary_directory(self):
        # Keep Windows sandbox temporary files inside the explicit writable root.
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])

    def test_extracts_only_generator_and_records_hash(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            archive = root / "models.zip"
            output = root / "weights" / "latest_net_G.pth"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("people/latest_net_G.pth", b"test-state-dict")
                zipped.writestr("../../outside.txt", b"not extracted")
                zipped.writestr("people/latest_net_D1.pth", b"not extracted")
            selected = installer.extract_generator(archive, output)
            self.assertEqual(selected, "people/latest_net_G.pth")
            self.assertEqual(output.read_bytes(), b"test-state-dict")
            self.assertEqual(list(output.parent.iterdir()), [output])
            record = installer.file_record(output, root)
            self.assertEqual(record["bytes"], len(b"test-state-dict"))
            self.assertEqual(len(record["sha256"]), 64)
            self.assertTrue(installer.reusable(output, record, force=False))
            output.write_bytes(b"corrupted")
            self.assertFalse(installer.reusable(output, record, force=False))

    def test_rejects_nonallowlisted_or_duplicate_generator(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            for entries in (
                ["../people/latest_net_G.pth"],
                ["people/latest_net_G.pth", "checkpoints/people/latest_net_G.pth"],
            ):
                archive = root / "models.zip"
                output = root / "out.pth"
                with zipfile.ZipFile(archive, "w") as zipped:
                    for entry in entries:
                        zipped.writestr(entry, b"payload")
                with self.assertRaises(ValueError):
                    installer.extract_generator(archive, output)
                self.assertFalse(output.exists())

    def test_rejects_generator_symlink(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            archive = root / "models.zip"
            info = zipfile.ZipInfo("people/latest_net_G.pth")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr(info, "../../outside")
            with self.assertRaises(ValueError):
                installer.extract_generator(archive, root / "out.pth")

    def test_cloak_provenance_is_explicit_and_json_is_atomic(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            bank = root / "data" / "phantomseal_cloaks"
            bank.mkdir(parents=True)
            (bank / "face.jpg").write_bytes(b"not decoded by installer")
            record = installer.cloak_provenance(root)
            saved = json.loads((bank / "PROVENANCE.json").read_text(encoding="utf-8"))
            self.assertEqual(record, saved)
            self.assertEqual(saved["image_count"], 1)
            self.assertIn("not PhantomSeal", saved["adaptation_notice"])
            self.assertEqual(len(list(bank.glob("*.part"))), 0)

    def test_reuses_central_runtime_and_copies_only_bundled_mtcnn_without_network(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            files = {
                "generator": "weights/simswap/latest_net_G.pth",
                "arcface": "weights/simswap/arcface_checkpoint.tar",
                "facenet_vggface2": "weights/facenet/facenet_vggface2.pt",
            }
            previous = {"sources": {"retained": "original provenance"},
                        "custom_provenance": "keep this", "assets": {}}
            for key, relative in files.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("runtime " + key).encode())
                previous["assets"][key] = installer.file_record(path, root, url="https://example.invalid/upstream")
            manifest_path = root / "weights/manifests/phantomseal.json"
            installer.atomic_json(manifest_path, previous)
            for name in ("pnet", "rnet", "onet"):
                path = root / f".venv/Lib/site-packages/facenet_pytorch/data/{name}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("package " + name).encode())
            with patch.object(installer, "download_file", side_effect=AssertionError("No network expected")):
                first = installer.install(root)
                self.assertEqual(first["custom_provenance"], "keep this")
                self.assertEqual(first["sources"]["retained"], "original provenance")
                times = {}
                for name in ("pnet", "rnet", "onet"):
                    path = root / f"weights/mtcnn/{name}.pt"
                    package = root / f".venv/Lib/site-packages/facenet_pytorch/data/{name}.pt"
                    self.assertEqual(path.read_bytes(), package.read_bytes())
                    times[name] = path.stat().st_mtime_ns
                    record = first["assets"]["mtcnn_" + name]
                    self.assertEqual(record["sha256"], installer.sha256_file(path))
                second = installer.install(root)
                for name, previous_time in times.items():
                    self.assertEqual((root / f"weights/mtcnn/{name}.pt").stat().st_mtime_ns, previous_time)
                self.assertTrue(all(record["path"].startswith("weights/") for record in second["assets"].values()))
            self.assertFalse((root / "models").exists())
            self.assertFalse(list(root.rglob("*.part")))


if __name__ == "__main__":
    unittest.main()
