import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.import_swap_assets import MANIFEST, MODEL_FILES, import_assets


class OptionalModelImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / "checkpoints"
        self.root = self.base / "rani-shield"
        self.source.mkdir()
        self.root.mkdir()

    def populate(self, model):
        for relative in MODEL_FILES[model]:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("utf-8"))

    def test_directory_selection_hash_copy_idempotency_and_manifest_merge(self):
        self.populate("uniface")
        self.populate("nullswap")
        first = import_assets(self.source, self.root, models=["uniface"])
        self.assertEqual(first["copied"], 1)
        self.assertFalse((self.root / "weights/nullswap").exists())
        self.assertEqual((self.root / "weights/uniface/500000.pt").read_bytes(), b"uniface/500000.pt")
        second = import_assets(self.source, self.root, models=["uniface"])
        self.assertEqual(second["skipped"], 1)
        import_assets(self.source, self.root, models=["nullswap"])
        manifest = json.loads((self.root / MANIFEST).read_text())
        self.assertEqual(len(manifest["assets"]), 2)
        self.assertTrue(all(row["path"].startswith("weights/") for row in manifest["assets"]))

    def test_different_existing_model_rejected_before_any_writes(self):
        self.populate("diffswap")
        conflict = self.root / "weights/diffswap/glint360k_r100.pth"
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            import_assets(self.source, self.root, models=["diffswap"])
        self.assertEqual(conflict.read_bytes(), b"keep me")
        self.assertFalse((conflict.parent / "diffswap.pth").exists())
        self.assertFalse((self.root / MANIFEST).exists())

    def test_zip_reads_only_exact_allowed_members_without_bulk_extraction(self):
        bundle = self.base / "weights.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("checkpoints/uniface/500000.pt", b"released weights")
            archive.writestr("../outside.txt", b"never extract")
            archive.writestr("checkpoints/unused/training.pt", b"never import")
        report = import_assets(bundle, self.root, models=["uniface"])
        self.assertEqual(report["copied"], 1)
        self.assertFalse((self.base / "outside.txt").exists())
        self.assertFalse((self.root / "weights/unused").exists())
        self.assertEqual((self.root / "weights/uniface/500000.pt").read_bytes(), b"released weights")

    def test_ambiguous_zip_aliases_fail_without_writes(self):
        bundle = self.base / "duplicate.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("checkpoints/uniface/500000.pt", b"a")
            archive.writestr("uniface/500000.pt", b"b")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            import_assets(bundle, self.root, models=["uniface"])
        self.assertFalse((self.root / "weights").exists())

    def test_dry_run_never_writes_or_downloads(self):
        self.populate("infoswap")
        with patch("scripts.import_swap_assets.urlopen") as network:
            report = import_assets(self.source, self.root, models=["infoswap"], dry_run=True)
        self.assertEqual(report["downloaded"], 11)
        self.assertEqual(report["copied"], 4)
        network.assert_not_called()
        self.assertFalse((self.root / "weights").exists())

    def test_readout_download_verified_and_skipped_when_installed(self):
        self.populate("infoswap")
        payload = b"pinned readout statistics"
        hashes = ((hashlib.sha256(payload).hexdigest(), len(payload)),)
        with patch("scripts.import_swap_assets.READOUT_HASHES", hashes):
            with patch("scripts.import_swap_assets.urlopen", return_value=io.BytesIO(payload)) as network:
                first = import_assets(self.source, self.root, models=["infoswap"])
                self.assertEqual(first["downloaded"], 1)
                self.assertIn("0484ef5fcb35bf811f2c3ec52ecf77ccade822db", network.call_args.args[0])
            with patch("scripts.import_swap_assets.urlopen") as network:
                second = import_assets(self.source, self.root, models=["infoswap"], download=False)
                network.assert_not_called()
                self.assertEqual(second["skipped"], 5)

    def test_corrupt_download_is_not_published_and_partial_file_is_cleaned(self):
        self.populate("infoswap")
        hashes = ((hashlib.sha256(b"correct").hexdigest(), 7),)
        with patch("scripts.import_swap_assets.READOUT_HASHES", hashes), \
             patch("scripts.import_swap_assets.urlopen", return_value=io.BytesIO(b"corrupt")):
            with self.assertRaisesRegex(ValueError, "checksum/size"):
                import_assets(self.source, self.root, models=["infoswap"])
        self.assertFalse((self.root / "weights/infoswap/readout/readout_layer0.pth").exists())
        self.assertEqual(list(self.root.rglob("*.part")), [])
        self.assertFalse((self.root / MANIFEST).exists())

    def test_no_download_fails_before_import_when_statistics_missing(self):
        self.populate("infoswap")
        with self.assertRaisesRegex(FileNotFoundError, "readout"):
            import_assets(self.source, self.root, models=["infoswap"], download=False)
        self.assertFalse((self.root / "weights").exists())

    def test_destination_symlink_is_rejected(self):
        self.populate("uniface")
        outside = self.base / "outside"
        outside.mkdir()
        try:
            (self.root / "weights").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Creating symlinks is unavailable on this Windows host.")
        with self.assertRaisesRegex(ValueError, "link|junction"):
            import_assets(self.source, self.root, models=["uniface"])
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
