import argparse
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from swap_adapters.uniface import (
    ARCHITECTURE, UniFaceAdapter, _image_tensor, _load_checkpoint, _output_image,
    readiness,
)
from third_party.uniface.ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class UniFaceOperatorsTests(unittest.TestCase):
    def test_fused_activation_matches_scalar_definition_and_gradients(self):
        values = torch.tensor([[[[-2., 1., .7]]], [[[3., -.8, -.5]]]], requires_grad=True)
        bias = torch.tensor([.25], requires_grad=True)
        result = fused_leaky_relu(values, bias)
        shifted = values + bias.reshape(1, 1, 1, 1)
        expected = torch.where(shifted >= 0, shifted, shifted * .2) * math.sqrt(2)
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        derivative = torch.where(shifted >= 0, math.sqrt(2), .2 * math.sqrt(2))
        torch.testing.assert_close(values.grad, derivative)
        torch.testing.assert_close(bias.grad, derivative.sum().reshape(1))
        layer = FusedLeakyReLU(1)
        self.assertEqual(list(layer.state_dict()), ["bias"])

    def test_upfirdn_matches_independent_numpy_filter_with_crop_and_decimation(self):
        rng = np.random.default_rng(45)
        values = rng.normal(size=(2, 3, 4, 5)).astype(np.float64)
        kernel = np.array([[1., 2., 4.], [3., 7., 6.]]) / 23.
        for up, down, pad in ((1, 1, (1, 2)), (2, 1, (1, 0)),
                              (1, 2, (-1, 3)), (2, 3, (-2, 1))):
            with self.subTest(up=up, down=down, pad=pad):
                expanded = np.zeros((2, 3, 4 * up, 5 * up))
                expanded[:, :, ::up, ::up] = values
                p0, p1 = pad
                expanded = np.pad(expanded, ((0, 0), (0, 0),
                    (max(p0, 0), max(p1, 0)), (max(p0, 0), max(p1, 0))))
                expanded = expanded[:, :, max(-p0, 0):expanded.shape[2] - max(-p1, 0),
                                    max(-p0, 0):expanded.shape[3] - max(-p1, 0)]
                filtered = np.empty((2, 3, expanded.shape[2] - 1, expanded.shape[3] - 2))
                for y in range(filtered.shape[2]):
                    for x in range(filtered.shape[3]):
                        filtered[:, :, y, x] = (expanded[:, :, y:y+2, x:x+3]
                                               * kernel[::-1, ::-1]).sum(axis=(-2, -1))
                actual = upfirdn2d(torch.from_numpy(values), torch.from_numpy(kernel),
                                  up=up, down=down, pad=pad)
                np.testing.assert_allclose(actual.numpy(), filtered[:, :, ::down, ::down],
                                           atol=1e-12, rtol=1e-12)


class UniFaceAdapterTests(unittest.TestCase):
    def test_preprocessing_matches_official_pil_torchvision_transform(self):
        image = Image.fromarray(np.random.default_rng(9).integers(
            0, 256, (87, 131, 3), dtype=np.uint8))
        expected = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5), inplace=True),
        ])(image).unsqueeze(0)
        torch.testing.assert_close(_image_tensor(image, "cpu"), expected, rtol=0, atol=1.2e-7)

    def test_output_quantization_clamps_and_rejects_nonfinite(self):
        values = torch.zeros(1, 3, 256, 256)
        values[:, 0] = -2
        values[:, 2] = 2
        image = _output_image(values)
        self.assertEqual(image.size, (256, 256))
        self.assertEqual(image.getpixel((0, 0)), (0, 128, 255))
        values[0, 1, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _output_image(values)

    def test_missing_checkpoint_is_explicit_and_invalid_target_never_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(readiness(temporary)["ready"])
            adapter = UniFaceAdapter(temporary, device="cpu")
            with patch.object(adapter, "load") as load:
                with self.assertRaisesRegex(ValueError, "224x224"):
                    adapter.swap(Image.new("RGB", (512, 512)), Image.new("RGB", (128, 128)))
                load.assert_not_called()
            with self.assertRaises(FileNotFoundError):
                adapter.load()

    def test_checkpoint_allows_only_metadata_namespace_and_uses_cpu_weights_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save({"train_args": argparse.Namespace(**ARCHITECTURE),
                        "g_ema": {"weight": torch.ones(2)},
                        "e_ema": {"weight": torch.zeros(2)}}, path)
            original = torch.load
            with patch("torch.load", wraps=original) as loader:
                generator, encoder = _load_checkpoint(path)
                self.assertTrue(torch.equal(generator["weight"], torch.ones(2)))
                self.assertTrue(torch.equal(encoder["weight"], torch.zeros(2)))
                self.assertTrue(loader.call_args.kwargs["weights_only"])
                self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
            invalid_path = Path(temporary) / "invalid.pt"
            torch.save({"train_args": argparse.Namespace(**{**ARCHITECTURE, "size": 512}),
                        "g_ema": {"weight": torch.ones(2)},
                        "e_ema": {"weight": torch.zeros(2)}}, invalid_path)
            with self.assertRaisesRegex(ValueError, "size"):
                _load_checkpoint(invalid_path)

    def test_target_source_order_and_reversed_target_features_match_upstream(self):
        adapter = UniFaceAdapter(".", device="cpu")
        observed = {}

        def encoder(batch):
            observed["batch"] = batch
            return batch[:, :, :8, :8], [batch[:, :, :32, :32], batch[:, :, :64, :64]]

        def generator(args):
            observed["args"] = args
            return torch.zeros(1, 3, 256, 256)

        adapter.encoder, adapter.generator = encoder, generator
        source = Image.new("RGB", (500, 300), "white")
        target = Image.new("RGB", (224, 224), "black")
        result = adapter.swap(source, target)
        self.assertEqual(result.size, (256, 256))
        torch.testing.assert_close(observed["batch"][0], -torch.ones(3, 256, 256))
        torch.testing.assert_close(observed["batch"][1], torch.ones(3, 256, 256))
        target_code, source_code, features = observed["args"]
        self.assertEqual(float(target_code.mean()), -1)
        self.assertEqual(float(source_code.mean()), 1)
        self.assertEqual([x.shape[-1] for x in features], [64, 32])
        self.assertTrue(all(float(x.mean()) == -1 for x in features))
        adapter.close()
        self.assertIsNone(adapter.encoder)
        self.assertIsNone(adapter.generator)


if __name__ == "__main__":
    unittest.main()
