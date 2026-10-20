"""InfoSwap image/color boundaries, detection failure and safe asset loading."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from swap_adapters.infoswap import (
    InfoSwapAdapter, REQUIRED_FILES, from_model_tensor, load_tensor_state,
    readout_parameters, readiness, to_model_tensor,
)


class InfoSwapTests(unittest.TestCase):
    def test_readiness_needs_all_inference_weights_but_no_discriminator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(readiness(root)["ready"])
            for relative in REQUIRED_FILES:
                path = root / "weights" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            self.assertTrue(readiness(root)["ready"])
            self.assertFalse(any("ckpt_ks_D" in item for item in REQUIRED_FILES))
            (root / "weights/infoswap/readout/readout_layer10.pth").unlink()
            self.assertFalse(readiness(root)["ready"])

    def test_rgb_boundary_matches_official_bgr_model_and_roundtrips(self):
        image = Image.new("RGB", (512, 512), (255, 127, 0))
        tensor = to_model_tensor(image, "cpu")
        self.assertEqual(tensor[0, 0, 0, 0].item(), -1)
        self.assertEqual(tensor[0, 2, 0, 0].item(), 1)
        np.testing.assert_array_equal(np.asarray(from_model_tensor(tensor)), np.asarray(image))

    def test_invalid_generated_images_raise(self):
        with self.assertRaises(RuntimeError):
            from_model_tensor(torch.zeros(1, 3, 224, 224))
        with self.assertRaises(RuntimeError):
            from_model_tensor(torch.full((1, 3, 512, 512), float("nan")))

    def test_alignment_fails_on_missing_face(self):
        adapter = InfoSwapAdapter(".", device="cpu")
        adapter.detector = Mock()
        adapter.detector.detect.return_value = (None, None, None)
        with self.assertRaisesRegex(ValueError, "could not detect"):
            adapter.align(Image.new("RGB", (224, 224)))

    def test_restricted_cpu_checkpoint_loading(self):
        with patch("torch.load", return_value={"weight": torch.ones(1)}) as loader:
            load_tensor_state(Path("weights.pth"))
        loader.assert_called_once_with(Path("weights.pth"), map_location="cpu", weights_only=True)
        with patch("torch.load", return_value={"weight": "not a tensor"}):
            with self.assertRaises(ValueError):
                load_tensor_state(Path("weights.pth"))

    def test_invalid_readout_statistics_rejected(self):
        state = {"n_samples": torch.tensor([1]), "m": torch.zeros(3, 4, 4),
                 "s": torch.zeros(3, 4, 4), "neuron_nonzero": torch.zeros(3, 4, 4)}
        with patch("swap_adapters.infoswap.load_tensor_state", return_value=state):
            with self.assertRaisesRegex(ValueError, "sample count"):
                readout_parameters(Path("statistics.pth"), "cpu")

    def test_invalid_seed_rejected_before_loading(self):
        adapter = InfoSwapAdapter(".", device="cpu")
        adapter.load = Mock(side_effect=AssertionError("must not load"))
        for seed in (True, -1, 2**32, 0.5):
            with self.assertRaises(ValueError):
                adapter.swap(None, None, seed=seed)
        adapter.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
