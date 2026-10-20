"""Model-free contract checks. Actual defence efficacy needs independent model evaluation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image
import torch

from protector import FaceShieldProtector, analysis_view, fit_dimensions, smooth_step


REPOSITORY = Path(__file__).resolve().parents[1]


class SyntheticProtector(FaceShieldProtector):
    """Exercise the real image/PGD/filter/save path with a deterministic synthetic gradient."""
    def load_models(self, progress=None):
        self._loaded = True
        self._notify(progress, 0.2, "Synthetic test objective")
        return self

    def _prepare_reference(self, source, analysis_size):
        return {}

    def _iteration_gradient(self, source, delta, reference, analysis_size):
        width = delta.shape[-1]
        pattern = torch.sin(torch.arange(width, device=delta.device) / 25)
        return pattern[None, None, None].expand_as(delta).clone()


class ProtectorContractTests(unittest.TestCase):
    def test_fit_preserves_aspect_and_never_enlarges(self):
        self.assertEqual(fit_dimensions(1920, 1080, 1280), (1280, 720))
        self.assertEqual(fit_dimensions(640, 960, 1280), (640, 960))
        self.assertEqual(fit_dimensions(1, 4000, 1280), (1, 1280))

    def test_analysis_view_is_differentiable_and_letterboxed(self):
        image = torch.ones((1, 3, 48, 96), requires_grad=True)
        view = analysis_view(image, 64)
        self.assertEqual(tuple(view.shape), (1, 3, 64, 64))
        self.assertTrue(torch.all(view[:, :, :16] == 0.5))
        view.sum().backward()
        self.assertIsNotNone(image.grad)
        self.assertGreater(image.grad.sum().item(), 0)

    def test_zero_update_is_finite(self):
        self.assertTrue(torch.isfinite(smooth_step(torch.zeros((1, 3, 64, 64)))).all())

    def test_full_resolution_pixels_bound_and_progress(self):
        # Fine detail would be lost by protecting at 512 then enlarging the PNG.
        source = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            input_path = folder / "input.png"
            Image.fromarray(source).save(input_path)
            runtime = SyntheticProtector(root_dir=REPOSITORY, device="cpu")
            events = []
            result = runtime.protect(input_path, folder / "out", iterations=2,
                                     progress=lambda fraction, message: events.append((fraction, message)))
            actual_source = np.array(Image.open(result["source_path"]))
            protected = np.array(Image.open(result["output_path"]))
            np.testing.assert_array_equal(actual_source, source)
            self.assertEqual(protected.shape, source.shape)
            difference = np.abs(protected.astype(np.int16) - source.astype(np.int16))
            self.assertLessEqual(difference.max(), 12)
            self.assertGreater(difference.max(), 0)
            self.assertEqual((result["width"], result["height"]), (1280, 720))
            self.assertEqual(result["method"], "multi_scale_approximation")
            self.assertEqual(events[0][0], 0)
            self.assertEqual(events[-1][0], 1)
            self.assertEqual(sorted(event[0] for event in events), [event[0] for event in events])

    def test_invalid_input_fails_before_loading_models(self):
        runtime = SyntheticProtector(root_dir=REPOSITORY, device="cpu")
        with self.assertRaises(ValueError):
            runtime.protect("missing.png", "out", max_size=1281)
        with self.assertRaises(FileNotFoundError):
            runtime.protect("missing.png", "out")
        self.assertFalse(runtime._loaded)

    def test_checkpoint_loader_accepts_only_audited_module_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.tar"
            torch.save(torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.PReLU()), path)
            loaded = FaceShieldProtector._load_arcface(path)
            self.assertIsInstance(loaded, torch.nn.Sequential)
            torch.save(torch.nn.Identity(), path)
            with self.assertRaisesRegex(ValueError, "unaudited classes"):
                FaceShieldProtector._load_arcface(path)

    def test_missing_local_models_never_fall_back_to_remote_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = FaceShieldProtector(root_dir=root, device="cpu", local_files_only=False)
            self.assertEqual(Path(runtime.model_path), root / "weights/sd15")
            self.assertEqual(Path(runtime.image_encoder_path), root / "weights/ip-adapter")
            self.assertTrue(runtime._common()["local_files_only"])
            with patch("scripts.runtime_environment.configure_environment"), \
                 patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Unexpected download")):
                with self.assertRaisesRegex(FileNotFoundError, "Missing local model directory"):
                    runtime.load_models()
                with self.assertRaisesRegex(FileNotFoundError, "No local UNet weights"):
                    runtime._weight_file(runtime.model_path, "unet")
            self.assertFalse(runtime._loaded)

    def test_model_and_cache_overrides_are_project_relative_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = FaceShieldProtector(root_dir=root, device="cpu", model_path="custom/sd",
                image_encoder_path="custom/clip", cache_dir="custom/cache",
                pretrained_arcface50_path="custom/arcface.tar")
            self.assertEqual(Path(runtime.model_path), root / "custom/sd")
            self.assertEqual(Path(runtime.image_encoder_path), root / "custom/clip")
            self.assertEqual(runtime.cache_dir, root / "custom/cache")
            self.assertEqual(runtime.arcface50_path, root / "custom/arcface.tar")
            for name in ("model_path", "image_encoder_path", "cache_dir", "pretrained_ip_adapter_path",
                         "pretrained_arcface50_path", "pretrained_arcface100_path"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "inside the project"):
                    FaceShieldProtector(root_dir=root, device="cpu", **{name: "../outside"})

    def test_failed_model_load_releases_partial_models_and_allows_retry(self):
        runtime = FaceShieldProtector(root_dir=REPOSITORY, device="cpu")
        references = []
        def failing_load(progress):
            runtime.unet = torch.nn.Linear(2, 2)
            runtime._dct_cache = torch.ones(1)
            references.append(weakref.ref(runtime.unet))
            references.append(weakref.ref(runtime._dct_cache))
            raise RuntimeError("Partial checkpoint failure")
        with patch.object(runtime, "_load_models", side_effect=failing_load):
            with self.assertRaisesRegex(RuntimeError, "Partial checkpoint failure"):
                runtime.load_models()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertFalse(runtime._loaded)
        def successful_load(progress):
            runtime.unet = torch.nn.Linear(2, 2)
            runtime._loaded = True
            return runtime
        with patch.object(runtime, "_load_models", side_effect=successful_load):
            self.assertIs(runtime.load_models(), runtime)
            self.assertTrue(runtime._loaded)
        runtime.close()
        runtime.close()
        self.assertIsNone(runtime.unet)
        self.assertFalse(runtime._loaded)

    def test_optional_detector_binds_central_weight_path_and_remains_disabled_by_default(self):
        enabled = FaceShieldProtector(root_dir=REPOSITORY, device="cpu", enable_mtcnn=True)
        disabled = FaceShieldProtector(root_dir=REPOSITORY, device="cpu")
        with patch("utils.landmark.mtcnn_attack.mtcnn_attack", return_value="detected") as detector:
            enabled._configure_mtcnn_attack()
            disabled._configure_mtcnn_attack()
            self.assertIsNone(disabled._mtcnn_attack)
            self.assertEqual(enabled._mtcnn_attack("pixels"), "detected")
            detector.assert_called_once_with("pixels", weights_path=
                REPOSITORY.resolve() / "weights/mtcnn-attack/pnet-6b6ef92b.pth")


if __name__ == "__main__":
    unittest.main()
