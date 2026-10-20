"""DiffSwap's pure-CPU contracts; no checkpoints, server or network required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from swap_adapters.diffswap import (
    DiffSwapAdapter, MODEL_FILES, affine_theta, blend_crop, ffhq_crop,
    portrait_masks, readiness,
)
from third_party.diffswap.inference import DiffSwapModel, VectorQuantizer, read_state
from swap_adapters import FaceAlignmentError


class DiffSwapAdapterTests(unittest.TestCase):
    def test_readiness_uses_only_local_model_and_detector_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(readiness(root)["ready"])
            for group, names in (("diffswap", MODEL_FILES), ("mtcnn", ("pnet.pt", "rnet.pt", "onet.pt"))):
                for name in names:
                    path = root / "weights" / group / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"file check only")
            with patch("swap_adapters.diffswap.importlib.util.find_spec", return_value=object()):
                self.assertTrue(readiness(root)["ready"])
            (root / "weights/mtcnn/pnet.pt").unlink()
            self.assertIn("pnet.pt", readiness(root)["message"])

    def test_bad_landmarks_fail_explicitly(self):
        for points in (np.zeros((5, 2)), np.full((68, 2), np.nan)):
            with self.assertRaises(FaceAlignmentError):
                ffhq_crop(Image.new("RGB", (256, 256)), points)
            with self.assertRaises(FaceAlignmentError):
                portrait_masks(points)
        with self.assertRaises(FaceAlignmentError):
            ffhq_crop(Image.new("RGB", (256, 256)), np.zeros((68, 2)))
        with self.assertRaises(FaceAlignmentError):
            portrait_masks(np.zeros((68, 2)))
        with self.assertRaises(FaceAlignmentError):
            affine_theta(np.zeros((5, 2)))

    def test_affine_reference_maps_112_crop_back_into_256(self):
        from third_party.diffswap.align_trans import get_reference_facial_points
        reference = get_reference_facial_points(default_square=True)
        theta = affine_theta(reference)
        expected = np.array([[111 / 255, 0, 111 / 255 - 1],
                             [0, 111 / 255, 111 / 255 - 1]], dtype=np.float32)
        np.testing.assert_allclose(theta, expected, atol=1e-6)

    def test_blending_extreme_masks_preserve_exact_inputs(self):
        random = np.random.default_rng(42)
        first = random.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        second = random.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(blend_crop(first, second, np.ones((256, 256)))), first)
        np.testing.assert_array_equal(
            np.asarray(blend_crop(first, second, np.zeros((256, 256)))), second)

    def test_detection_failure_never_returns_the_target_as_a_swap(self):
        adapter = DiffSwapAdapter(".", "cpu")
        adapter.detector = lambda _image, _upsample: []
        with self.assertRaisesRegex(FaceAlignmentError, "could not detect"):
            adapter._landmarks(np.zeros((256, 256, 3), dtype=np.uint8))

    def test_close_releases_owned_models(self):
        adapter = DiffSwapAdapter(".")
        adapter.model = adapter.detector = adapter.predictor = adapter.mtcnn = object()
        adapter.close()
        self.assertIsNone(adapter.model)
        self.assertIsNone(adapter.detector)
        self.assertIsNone(adapter.predictor)
        self.assertIsNone(adapter.mtcnn)

    def test_invalid_input_and_seed_fail_before_loading_models(self):
        adapter = DiffSwapAdapter(".")
        # Common benchmark reports must be able to describe this backend
        # without initializing its large models.
        self.assertIsInstance(adapter.preprocessing, dict)
        self.assertIn("200 DDIM steps", adapter.preprocessing["sampling"])
        self.assertIn("68-point", adapter.preprocessing["source"])
        self.assertIn("MTCNN", adapter.preprocessing["identity"])
        self.assertIn("256px", adapter.preprocessing["output"])
        portrait, target = Image.new("RGB", (512, 512)), Image.new("RGB", (224, 224))
        with patch.object(adapter, "load") as load:
            for source, destination, seed in (
                    (None, target, 0), (portrait, portrait, 0),
                    (Image.new("RGB", (8, 8)), target, 0),
                    (portrait, target, -1), (portrait, target, True)):
                with self.assertRaises(ValueError):
                    adapter.swap(source, destination, seed=seed)
            load.assert_not_called()

    def test_tensor_loader_rejects_pickled_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            torch.save({"state_dict": {"test": torch.ones(2)}}, path)
            torch.testing.assert_close(read_state(path)["test"], torch.ones(2))
            torch.save(nn.Linear(2, 2), path)
            with self.assertRaises(Exception):
                read_state(path)

    def test_chunked_vq_matches_full_nearest_code_search(self):
        with torch.random.fork_rng():
            torch.manual_seed(123)
            quantizer = VectorQuantizer()
            latent = torch.randn(1, 3, 40, 32)
        flat = latent.permute(0, 2, 3, 1).reshape(-1, 3)
        weights = quantizer.embedding.weight
        distance = flat.square().sum(1, keepdim=True) + weights.square().sum(1) - 2 * flat @ weights.T
        expected = weights[distance.argmin(1)].reshape(1, 40, 32, 3).permute(0, 3, 1, 2)
        torch.testing.assert_close(quantizer(latent), expected, rtol=0, atol=0)

    def test_sampler_encodes_source_separately_and_preserves_global_rng(self):
        class Stage(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def encode(self, image):
                self.inputs.append(image.clone())
                return image

            def decode(self, latent):
                return latent

        class Conditioner(nn.Module):
            def forward(self, image, latent, organs, theta, landmarks):
                self.source = latent.clone()
                return latent.mean().reshape(1, 1, 1)

        class Denoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.times = []

            def forward(self, image, step, context):
                self.times.append(int(step))
                return image * .1 + context[..., None]

        model = DiffSwapModel.__new__(DiffSwapModel)
        nn.Module.__init__(model)
        model.first_stage, model.conditioner, model.unet = Stage(), Conditioner(), Denoiser()
        model.register_buffer("alphas", torch.linspace(.999, .1, 1000))
        source = {"image": torch.ones(1, 3, 2, 2) * .2, "organs": None, "theta": None}
        target = {"image": torch.ones(1, 3, 2, 2) * -.4,
                  "mask": torch.tensor([[[1., 0.], [0., 1.]]]), "landmarks": None}
        for steps in (0, 1, 3, 1000, True):
            with self.assertRaises(ValueError):
                model.generate(source, target, steps=steps)
        before = torch.random.get_rng_state().clone()
        first = model.generate(source, target, seed=130)
        second = model.generate(source, target, seed=130)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        self.assertEqual(model.unet.times[:200], list(range(996, 0, -5)))
        torch.testing.assert_close(model.conditioner.source, source["image"])
        self.assertFalse(torch.equal(model.first_stage.inputs[0], model.first_stage.inputs[1]))
        self.assertEqual(first.shape, (1, 3, 2, 2))
        self.assertTrue(torch.isfinite(first).all())
        # Recorded from official DDIMSampler at the pinned upstream commit,
        # with this toy denoiser/schedule, CPU seed130,200steps,eta0,tgt_scale.01.
        # Covers the +1 timetable, background q_sample and target guidance.
        expected = torch.tensor([
            .3993842304, .2670300305, .3134586215, .8734630346,
            0., .2959592938, .3104380071, 0.,
            0., .2814384103, .3197453022, .9691079855,
        ]).reshape(1, 3, 2, 2)
        torch.testing.assert_close(first, expected, atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
