"""No-model tests for SimSwap's benchmark-only pose alignment."""
import unittest

import numpy as np
from PIL import Image

from benchmark_alignment import POSE_TEMPLATES, aligned_crop, estimate_norm


class BenchmarkAlignmentTests(unittest.TestCase):
    def test_recovers_known_similarity_for_each_of_the_five_poses(self):
        theta = .27
        scale = .72
        expected = np.array([[scale * np.cos(theta), -scale * np.sin(theta), 17.3],
                             [scale * np.sin(theta), scale * np.cos(theta), -9.1]])
        for pose in range(5):
            with self.subTest(pose=pose):
                target = POSE_TEMPLATES[pose] * 2
                source = (target - expected[:, 2]) @ np.linalg.inv(expected[:, :2]).T
                matrix, chosen = estimate_norm(source)
                self.assertEqual(chosen, pose)
                np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-11)
                np.testing.assert_allclose(source @ matrix[:, :2].T + matrix[:, 2], target, atol=1e-11)

    def test_noisy_landmarks_match_independent_similarity_svd_and_pose_ranking(self):
        source = POSE_TEMPLATES[3].astype(np.float64) * 2 + np.array(
            [[1.2, -.5], [-.4, 1.1], [.7, -.9], [-.6, .1], [.3, -.2]])
        matrix, chosen = estimate_norm(source)
        candidates = []
        for target in POSE_TEMPLATES * 2:
            src_center = source.mean(0)
            dst_center = target.astype(np.float64).mean(0)
            src = source - src_center
            dst = target - dst_center
            u, singular, vh = np.linalg.svd(dst.T @ src / len(src))
            correction = np.diag([1., np.sign(np.linalg.det(u @ vh))])
            rotation = u @ correction @ vh
            scale = (singular * np.diag(correction)).sum() / np.square(src).sum(1).mean()
            linear = scale * rotation
            offset = dst_center - linear @ src_center
            candidate = np.column_stack((linear, offset))
            error = np.linalg.norm(source @ linear.T + offset - target, axis=1).sum()
            candidates.append((error, candidate))
        best = min(range(5), key=lambda i: candidates[i][0])
        self.assertEqual(chosen, best)
        self.assertEqual(chosen, 3)
        np.testing.assert_allclose(matrix, candidates[best][1], atol=1e-11)

    def test_warp_preserves_rgb_and_uses_black_constant_border(self):
        image = Image.new("RGB", (224, 224), (20, 120, 230))
        source = POSE_TEMPLATES[2] * 2 + np.array([0., 40.])
        crop = aligned_crop(image, source)
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(crop.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(crop)[0, 0], [20, 120, 230])
        np.testing.assert_array_equal(np.asarray(crop)[-1, 0], [0, 0, 0])

    def test_degenerate_nonfinite_and_invalid_shapes_fail_explicitly(self):
        for landmarks in (np.ones((5, 2)), np.full((5, 2), np.nan), np.zeros((4, 2))):
            with self.subTest(shape=landmarks.shape):
                with self.assertRaises(ValueError):
                    aligned_crop(Image.new("RGB", (224, 224)), landmarks)


if __name__ == "__main__":
    unittest.main()
