"""The optional attack detector must use explicit local weights without Hub I/O."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_mtcnn.get_nets import PNet
from utils.landmark.mtcnn_attack import PNet_attack


class LocalAttackDetectorTests(unittest.TestCase):
    def test_local_weights_match_upstream_probabilities_and_input_gradient(self):
        # Construct the upstream architecture with deterministic synthetic weights;
        # its network loader is replaced before construction, not invoked.
        torch.manual_seed(7)
        with patch('torch_mtcnn.get_nets.load_state', return_value={}), \
             patch.object(torch.nn.Module, 'load_state_dict'), \
             patch('torch_mtcnn.get_nets.device', torch.device('cpu')):
            reference = PNet().cpu().eval()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'pnet.pth'
            torch.save(reference.state_dict(), checkpoint)
            with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('No Hub access')), \
                 patch('torch_mtcnn.get_nets.load_state', side_effect=AssertionError('No parent auto-loader')):
                model = PNet_attack('cpu', weights_path=checkpoint).eval()
            pixels = torch.randn(1, 3, 32, 32, requires_grad=True)
            with patch('torch_mtcnn.get_nets.device', torch.device('cpu')):
                expected = reference(pixels)[1]
            actual = model(pixels)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradient = torch.autograd.grad(actual[:, 1].square().mean(), pixels)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_missing_explicit_file_fails_without_downloading(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('No Hub access')):
            with self.assertRaisesRegex(FileNotFoundError, 'manage.cmd models'):
                PNet_attack('cpu', weights_path=Path(temporary) / 'missing.pth')


if __name__ == '__main__':
    unittest.main()
