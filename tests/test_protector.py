"""Model-free contract checks. Actual defence efficacy needs independent model evaluation."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from protector import FaceShieldProtector, analysis_view, fit_dimensions, smooth_step


REPOSITORY = Path(os.environ.get("FACESHIELD_ROOT", str(Path(__file__).resolve().parents[1])))


class SyntheticProtector(FaceShieldProtector):
    """Exercise the real image/PGD/filter/save path with a deterministic synthetic gradient."""
    def load_models(self, progress=None):
        if str(self.root_dir) not in sys.path:
            sys.path.insert(0, str(self.root_dir))
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
        if str(REPOSITORY) not in sys.path:
            sys.path.insert(0, str(REPOSITORY))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.tar"
            torch.save(torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.PReLU()), path)
            loaded = FaceShieldProtector._load_arcface(path)
            self.assertIsInstance(loaded, torch.nn.Sequential)
            torch.save(torch.nn.Identity(), path)
            with self.assertRaisesRegex(ValueError, "unaudited classes"):
                FaceShieldProtector._load_arcface(path)


if __name__ == "__main__":
    unittest.main()
