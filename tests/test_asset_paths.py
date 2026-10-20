"""Central storage defaults and explicit custom configuration, without GPU models."""
import json
from pathlib import Path
import tempfile
import unittest

from asset_paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_CLOAK_DIR, weight_path, weights_dir
from ensemble_encoders import ensemble_paths
from phantomseal_assets import configuration
from protector import FaceShieldProtector


class AssetPathTests(unittest.TestCase):
    def test_all_default_model_paths_and_cache_use_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("sd15/unet/config.json", "ip-adapter/models/image_encoder/config.json"):
                path = weight_path(root, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            runtime = FaceShieldProtector(root_dir=root, device="cpu")
            self.assertEqual(Path(runtime.model_path), root / "weights/sd15")
            self.assertEqual(Path(runtime.image_encoder_path), root / "weights/ip-adapter")
            self.assertEqual(runtime.ip_adapter_path, root / "weights/ip-adapter/ip-adapter_sd15.bin")
            self.assertEqual(runtime.arcface50_path, root / "weights/faceshield/arcface50_checkpoint.tar")
            self.assertEqual(runtime.arcface100_path, root / "weights/faceshield/arcface100_checkpoint.tar")
            self.assertEqual(runtime.cache_dir, root / "weights/cache/huggingface")
            self.assertEqual(weights_dir(root), root / "weights")
            config = configuration(root)
            self.assertEqual(config["checkpoint_dir"], root / DEFAULT_CHECKPOINT_DIR)
            self.assertEqual(config["cloak_dir"], root / DEFAULT_CLOAK_DIR)
            self.assertEqual(ensemble_paths(root)["simswap"], root / "weights/simswap/arcface_checkpoint.tar")

    def test_explicit_custom_configuration_is_respected_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "external_bank"
            (root / "phantomseal_config.json").write_text(json.dumps({
                "checkpoint_dir": "custom/checkpoints", "cloak_dir": str(bank), "expected_cloaks": 214}))
            legacy = root / "models/phantomseal/arcface_checkpoint.tar"
            legacy.parent.mkdir(parents=True)
            legacy.touch()
            config = configuration(root)
            self.assertEqual(config["checkpoint_dir"], root / "custom/checkpoints")
            self.assertEqual(config["cloak_dir"], bank)
            self.assertEqual(ensemble_paths(root)["simswap"], root / "custom/checkpoints/simswap/arcface_checkpoint.tar")

    def test_weight_relative_paths_cannot_escape_the_central_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            for path in ("../outside.pt", "a/../../outside.pt", str(Path(temporary) / "absolute.pt"), "\\rooted.pt", "C:drive-relative.pt"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    weight_path(temporary, path)


if __name__ == "__main__":
    unittest.main()
