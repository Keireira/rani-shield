"""Tiny CPU-only fixtures; no actual weights, network, model loading or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.import_local_assets import import_assets
from scripts.weight_assets import (CLOAK_DEST, IMPORT_MANIFEST, LEGACY_DEST, LEGACY_FILE,
    MODEL_MANIFEST, MTCNN_RECORDS, OLD_CONFIG, PATH_MAP, RELATIVE_CONFIG, RUNTIME_RECORDS,
    checked_destination, sha256_file)


def make_installation(folder, unified=False):
    root = folder / ("unified source" if unified else "original source")
    root.mkdir()
    for old, new in PATH_MAP:
        path = root / (new if unified else old)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixture " + new).encode())
    checkpoint = root / ("weights" if unified else OLD_CONFIG["checkpoint_dir"])
    bank = root / (CLOAK_DEST if unified else OLD_CONFIG["cloak_dir"])
    for relative in ("simswap/arcface_checkpoint.tar", "faceshifter/model_ir_se50.pth", "hififace/hififace_opensource_299999.ckpt"):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = ("fixture weights/" + relative).encode()
        path.write_bytes(contents)
    bank.mkdir(parents=True)
    for number in range(214):
        (bank / f"{number:03}.png").write_bytes(b"image fixture, never decoded")
    (root / "phantomseal_config.json").write_text(json.dumps(RELATIVE_CONFIG if unified else OLD_CONFIG))
    assets = {}
    mapping = dict((new, old) for old, new in PATH_MAP)
    for name, relative in RUNTIME_RECORDS.items():
        original = relative if unified else mapping[relative]
        path = root / original
        assets[name] = {"path": original, "bytes": path.stat().st_size, "sha256": sha256_file(path), "url": "https://example.invalid/official-source"}
    manifest = root / (MODEL_MANIFEST if unified else "models/phantomseal/manifest.json")
    manifest.write_text(json.dumps({"assets": assets, "sources": {"keep": "original provenance"}}))
    for name, relative in MTCNN_RECORDS.items():
        path = root / (relative if unified else f".venv/Lib/site-packages/facenet_pytorch/data/{name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("package " + name).encode())
    for relative in ("results/editor/uploads/private.png", ".venv/private.txt", ".cache/manage/server.json", ".cache/unused-attacker.pth", LEGACY_FILE):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not import by default")
    return root, bank


class AssetImportTests(unittest.TestCase):
    def test_original_and_unified_sources_converge_and_rerun_is_idempotent(self):
        for unified in (False, True):
            with self.subTest(unified=unified), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
                folder = Path(temporary)
                source, _ = make_installation(folder, unified)
                destination = folder / "new studio"; destination.mkdir()
                first = import_assets(source, destination)
                self.assertEqual(first["copied"], len(PATH_MAP) + 2 + 214 + 3)
                self.assertEqual(len(list((destination / CLOAK_DEST).glob("*.png"))), 214)
                self.assertEqual(json.loads((destination / "phantomseal_config.json").read_text()), RELATIVE_CONFIG)
                manifest = json.loads((destination / MODEL_MANIFEST).read_text())
                self.assertEqual(manifest["assets"]["arcface"]["path"], RUNTIME_RECORDS["arcface"])
                self.assertEqual(manifest["sources"], {"keep": "original provenance"})
                self.assertIn("mtcnn_pnet", manifest["assets"])
                for relative in ("results", ".venv", ".cache", LEGACY_DEST, "models", "utils"):
                    self.assertFalse((destination / relative).exists(), relative)
                asset = destination / PATH_MAP[0][1]
                original_time = asset.stat().st_mtime_ns
                second = import_assets(source, destination)
                self.assertEqual(second["copied"], 0)
                self.assertEqual(asset.stat().st_mtime_ns, original_time)
                self.assertEqual(asset.stat().st_nlink, 1)
                self.assertTrue((destination / IMPORT_MANIFEST).is_file())
                self.assertFalse(list(destination.rglob("*.part")))

    def test_dry_run_partial_bank_and_alias_mismatch_never_write(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary); source, bank = make_installation(folder)
            destination = folder / "destination"; destination.mkdir()
            self.assertTrue(import_assets(source, destination, dry_run=True)["dry_run"])
            self.assertEqual(list(destination.iterdir()), [])
            arc = source / OLD_CONFIG["checkpoint_dir"] / "simswap/arcface_checkpoint.tar"
            arc.write_bytes(b"different identity space")
            with self.assertRaisesRegex(ValueError, "aliases disagree"):
                import_assets(source, destination)
            next(bank.glob("*.png")).unlink()
            with self.assertRaisesRegex(ValueError, "214"):
                import_assets(source, destination)
            self.assertEqual(list(destination.iterdir()), [])

    def test_conflicts_refused_before_copy_and_explicit_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary); source, _ = make_installation(folder)
            destination = folder / "destination"; destination.mkdir()
            existing = destination / PATH_MAP[0][1]; existing.parent.mkdir(parents=True)
            existing.write_bytes(b"keep existing")
            with self.assertRaisesRegex(FileExistsError, "different existing"):
                import_assets(source, destination)
            self.assertEqual(existing.read_bytes(), b"keep existing")
            self.assertFalse((destination / PATH_MAP[1][1]).exists())
            import_assets(source, destination, overwrite=True, include_legacy=True)
            self.assertTrue((destination / LEGACY_DEST).is_file())

    def test_bad_provenance_and_destination_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary); source, _ = make_installation(folder)
            destination = folder / "destination"; destination.mkdir()
            (source / "models/phantomseal/latest_net_G.pth").write_bytes(b"changed after manifest")
            with self.assertRaisesRegex(ValueError, "provenance"):
                import_assets(source, destination)
            self.assertEqual(list(destination.iterdir()), [])
            with self.assertRaises(ValueError):
                checked_destination(destination, "../outside.bin")


if __name__ == "__main__":
    unittest.main()
