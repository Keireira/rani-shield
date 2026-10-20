"""SimSwap 224px five-pose source alignment, for the benchmark only.

Pose templates and minimum summed landmark-error selection are adapted from
neuralchen/SimSwap, insightface_func/utils/face_align_ffhqandnewarc.py,
commit bd7b7686a17f41dd11cfcd5d82f7e4c5eb94b780 (Naiyuan Liu and contributors).
Upstream license: CC BY-NC 4.0; retained in
third_party/phantomseal/SIMSWAP_LICENSE. Changes: NumPy least-squares similarity
fit replaces skimage SimilarityTransform; PIL RGB input/output and validation.
No detector, weights, or protection preprocessing is changed here.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


POSE_TEMPLATES = np.array([
    [[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
     [51.157, 89.050], [57.025, 89.702]],
    [[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
     [45.177, 86.190], [64.246, 86.758]],
    [[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
     [42.463, 87.010], [69.537, 87.010]],
    [[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
     [48.167, 86.758], [67.236, 86.190]],
    [[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
     [55.388, 89.702], [61.257, 89.050]],
], dtype=np.float32)


def estimate_norm(landmarks, size=224):
    """Return input-to-crop similarity matrix and minimum-error pose index.

    The fit minimizes squared landmark distances, as upstream's similarity
    estimator does. Pose selection then minimizes the SUM of Euclidean errors,
    matching estimate_norm(mode='None'), rather than choosing by squared error.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Crop size must be a positive integer.")
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Alignment needs five finite [x,y] face landmarks.")
    centered = points - points.mean(axis=0)
    if np.square(centered).sum() <= np.finfo(np.float64).eps:
        raise ValueError("Face landmarks are degenerate; cannot fit alignment.")
    # dx=a*x-b*y+tx, dy=b*x+a*y+ty: rotation plus uniform scale, no shear.
    design = np.zeros((10, 4), dtype=np.float64)
    design[0::2, 0] = points[:, 0]
    design[0::2, 1] = -points[:, 1]
    design[0::2, 2] = 1
    design[1::2, 0] = points[:, 1]
    design[1::2, 1] = points[:, 0]
    design[1::2, 3] = 1
    templates = POSE_TEMPLATES * (size / 112)
    best = None
    best_error = float("inf")
    best_pose = None
    for pose, target in enumerate(templates):
        coefficients, _, rank, _ = np.linalg.lstsq(design, target.reshape(-1), rcond=None)
        if rank != 4 or not np.isfinite(coefficients).all():
            raise ValueError("Face landmarks are degenerate; cannot fit alignment.")
        a, b, tx, ty = coefficients
        matrix = np.array([[a, -b, tx], [b, a, ty]])
        if a * a + b * b <= np.finfo(np.float64).eps:
            continue
        projected = points @ matrix[:, :2].T + matrix[:, 2]
        error = np.linalg.norm(projected - target, axis=1).sum()
        if error < best_error:
            best, best_error, best_pose = matrix, error, pose
    if best is None:
        raise ValueError("Face landmarks are degenerate; cannot fit alignment.")
    return best, best_pose


def aligned_crop(image: Image.Image, landmarks, size=224) -> Image.Image:
    """Warp RGB source to the official 224-model five-pose crop convention."""
    matrix, _ = estimate_norm(landmarks, size)
    rgb = np.asarray(image.convert("RGB"))
    crop = cv2.warpAffine(rgb, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(crop)
