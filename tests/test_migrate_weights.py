"""Only tiny project-local CPU fixtures; no installed assets are migrated."""
import json
from pathlib import Path
import tempfile
import unittest
from test_import_local_assets import make_installation
from scripts.migrate_weights import migrate
from scripts.weight_assets import (CLOAK_DEST, IMPORT_MANIFEST, MODEL_MANIFEST, OLD_CONFIG,
    PATH_MAP, RELATIVE_CONFIG, RUNTIME_RECORDS, sha256_file)


class WeightMigrationTests(unittest.TestCase):
    def test_plan_then_apply_deduplicates_and_preserves_package_private_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root, bank = make_installation(Path(temporary))
            model = root / "models/phantomseal/latest_net_G.pth"
            original_sha = sha256_file(model)
            hf = root / ".cache/huggingface/hub/version.txt"; hf.parent.mkdir(parents=True); hf.write_text("1")
            cache = root / ".cache/phantomseal/full-cloak-index.json"; cache.parent.mkdir(parents=True); cache.write_text('{"old":"cache"}')
            imported = root / ".cache/imported-assets.json"
            imported.write_text(json.dumps({"source_installation": "historical source", "assets": [
                {"path": "models/phantomseal/latest_net_G.pth", "bytes": model.stat().st_size, "sha256": original_sha, "source_path": "historical source file"}]}))
            plan = migrate(root)
            self.assertFalse(plan["apply"])
            self.assertTrue(model.is_file())
            self.assertFalse((root / "weights").exists())
            migrate(root, apply=True)
            self.assertEqual(sha256_file(root / RUNTIME_RECORDS["generator"]), original_sha)
            self.assertFalse(model.exists())
            self.assertFalse((root / "models/phantomseal/arcface_checkpoint.tar").exists())
            self.assertFalse((root / OLD_CONFIG["checkpoint_dir"] / "simswap/arcface_checkpoint.tar").exists())
            self.assertTrue((root / RUNTIME_RECORDS["arcface"]).is_file())
            self.assertEqual(len(list((root / CLOAK_DEST).glob("*.png"))), 214)
            self.assertTrue((root / ".venv/Lib/site-packages/facenet_pytorch/data/pnet.pt").is_file())
            self.assertTrue((root / "results/editor/uploads/private.png").is_file())
            self.assertEqual(json.loads((root / "phantomseal_config.json").read_text()), RELATIVE_CONFIG)
            manifest = json.loads((root / MODEL_MANIFEST).read_text())
            self.assertEqual(manifest["assets"]["generator"]["path"], RUNTIME_RECORDS["generator"])
            self.assertEqual(manifest["sources"]["keep"], "original provenance")
            record = json.loads((root / IMPORT_MANIFEST).read_text())
            self.assertEqual(record["source_installation"], "historical source")
            self.assertEqual(record["assets"][0]["path"], RUNTIME_RECORDS["generator"])
            self.assertEqual((root / "weights/cache/huggingface/hub/version.txt").read_text(), "1")
            self.assertEqual((root / "weights/cache/phantomseal/full-cloak-index.json").read_text(), '{"old":"cache"}')
            before = sha256_file(root / MODEL_MANIFEST)
            migrate(root, apply=True)
            self.assertEqual(sha256_file(root / MODEL_MANIFEST), before)
            self.assertFalse(list(root.rglob("*.part")))

    def test_existing_conflict_prevents_all_moves_and_config_changes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root, _ = make_installation(Path(temporary))
            target = root / PATH_MAP[-1][1]; target.parent.mkdir(parents=True); target.write_bytes(b"keep conflicting model")
            config_before = (root / "phantomseal_config.json").read_bytes()
            with self.assertRaisesRegex(FileExistsError, "destination"):
                migrate(root, apply=True)
            self.assertEqual((root / "phantomseal_config.json").read_bytes(), config_before)
            self.assertTrue((root / PATH_MAP[0][0]).is_file())
            self.assertFalse((root / PATH_MAP[0][1]).exists())

    def test_outside_configuration_and_different_alias_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary); root, _ = make_installation(folder)
            config = dict(OLD_CONFIG); config["checkpoint_dir"] = str(folder / "other project")
            (root / "phantomseal_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "outside"):
                migrate(root, apply=True)
            (root / "phantomseal_config.json").write_text(json.dumps(OLD_CONFIG))
            (root / OLD_CONFIG["checkpoint_dir"] / "simswap/arcface_checkpoint.tar").write_bytes(b"different ArcFace")
            with self.assertRaisesRegex(FileExistsError, "aliases"):
                migrate(root, apply=True)
            self.assertTrue((root / PATH_MAP[0][0]).is_file())

    def test_resumes_a_partially_moved_bank_and_rejects_same_name_conflicts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root, old_bank = make_installation(Path(temporary))
            new_bank = root / CLOAK_DEST; new_bank.mkdir(parents=True)
            for image in sorted(old_bank.glob("*.png"))[:87]:
                image.rename(new_bank / image.name)
            migrate(root, apply=True)
            self.assertEqual(len(list(new_bank.glob("*.png"))), 214)
            self.assertFalse(old_bank.exists())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root, old_bank = make_installation(Path(temporary))
            new_bank = root / CLOAK_DEST; new_bank.mkdir(parents=True)
            (new_bank / "000.png").write_bytes(b"conflicting existing image")
            with self.assertRaisesRegex(FileExistsError, "destination|aliases"):
                migrate(root, apply=True)
            self.assertTrue((root / PATH_MAP[0][0]).is_file())


if __name__ == "__main__":
    unittest.main()
