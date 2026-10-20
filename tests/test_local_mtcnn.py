"""CPU parity with pinned packaged MTCNN; production only reads central weights."""
from importlib.util import find_spec
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from asset_paths import MTCNN_FILES, weight_path
from phantomseal_runtime import load_mtcnn


class LocalMTCNNTests(unittest.TestCase):
    def test_same_parameters_defaults_and_public_sample_detection_without_package_weight_reads(self):
        from facenet_pytorch import MTCNN
        package = Path(next(iter(find_spec("facenet_pytorch").submodule_search_locations)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in MTCNN_FILES:
                target = weight_path(root, "mtcnn", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(package / "data" / name, target)
            native_load = torch.load
            with patch("torch.load", wraps=native_load) as loaded:
                actual = load_mtcnn(root, device="cpu")
            self.assertEqual(len(loaded.call_args_list), 3)
            for call, name in zip(loaded.call_args_list, MTCNN_FILES):
                self.assertEqual(Path(call.args[0]), weight_path(root, "mtcnn", name))
                self.assertEqual(call.kwargs, {"map_location": "cpu", "weights_only": True})
            expected = MTCNN(image_size=160, margin=0, keep_all=True, device="cpu")
            for attribute in ("image_size", "margin", "min_face_size", "thresholds", "factor",
                              "post_process", "select_largest", "keep_all", "selection_method", "device"):
                self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))
            expected_state, actual_state = expected.state_dict(), actual.state_dict()
            self.assertEqual(set(expected_state), set(actual_state))
            for key in expected_state:
                torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
            self.assertFalse(any(parameter.requires_grad for parameter in actual.parameters()))
            sample = Path(__file__).resolve().parents[1] / "data/test/image.png"
            threads = torch.get_num_threads()
            try:
                torch.set_num_threads(4)
                with Image.open(sample) as opened:
                    image = opened.convert("RGB")
                expected_detection = expected.detect(image, landmarks=True)
                actual_detection = actual.detect(image, landmarks=True)
            finally:
                torch.set_num_threads(threads)
            self.assertIsNotNone(expected_detection[0], "The supplied public sample must have a detected face.")
            for actual_values, expected_values in zip(actual_detection, expected_detection):
                np.testing.assert_allclose(np.asarray(actual_values, dtype=np.float64),
                                           np.asarray(expected_values, dtype=np.float64), rtol=0, atol=1e-5)

    def test_missing_central_weights_fail_without_reading_packaged_weights(self):
        with tempfile.TemporaryDirectory() as temporary, patch("torch.load") as loaded:
            with self.assertRaisesRegex(FileNotFoundError, "Missing local MTCNN"):
                load_mtcnn(temporary, device="cpu")
            loaded.assert_not_called()


if __name__ == "__main__":
    unittest.main()
