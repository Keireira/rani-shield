"""Local DiffSwap adapter: portrait alignment, conditioning and native 256px swap.

No import-time model loading, process-wide import rewrites, network access or
external asset fallback. See third_party/diffswap/NOTICE.md for upstream credits.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from asset_paths import MTCNN_FILES, weight_path
from swap_adapters import FaceAlignmentError


MODEL_FILES = ("diffswap.pth", "glint360k_r100.pth",
               "shape_predictor_68_face_landmarks.dat")
PREPROCESSING = {
    "source": "Dlib 68-point landmarks; FFHQ alignment with a 1024px bicubic transform then a 256px RGB crop.",
    "target": "Prepared 224px RGB target, independently Dlib/FFHQ aligned to 256px.",
    "identity": "Central facenet-pytorch MTCNN five-point landmarks supply the author's 112px ArcFace affine crop; source latent encoded separately.",
    "conditioning": "Source identity and organ latents, target 68 landmarks and convex-hull mask.",
    "sampling": "200 DDIM steps, eta 0, target guidance 0.01, official EMA denoiser; identical local random seed per variant.",
    "output": "Native 256px RGB; vector-quantized decoder and seven-layer multiband blending with the aligned target.",
    "precision": "Float32 model inference; ArcFace's CUDA fp16 autocast follows the author configuration.",
}


def readiness(root_dir):
    """Cheap disk/package checks; never deserialize weights or initialize CUDA."""
    try:
        paths = [weight_path(root_dir, "diffswap", name) for name in MODEL_FILES]
        paths.extend(weight_path(root_dir, "mtcnn", name) for name in MTCNN_FILES)
    except ValueError as exc:
        return {"ready": False, "message": str(exc)}
    missing = [path.name for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return {"ready": False, "message": "DiffSwap files missing: " + ", ".join(missing)}
    if importlib.util.find_spec("dlib") is None:
        return {"ready": False, "message": "DiffSwap needs dlib; run the project setup."}
    return {"ready": True, "message": "DiffSwap: 200 DDIM steps, native 256px aligned crop."}


def ffhq_crop(image, landmarks):
    """Author pipeline's FFHQ crop: no padding, 1024 transform, then 256."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise FaceAlignmentError("DiffSwap needs 68 finite face landmarks.")
    eye_left, eye_right = points[36:42].mean(0), points[42:48].mean(0)
    eye_center = (eye_left + eye_right) * .5
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = (points[48] + points[54]) * .5 - eye_center
    axis_x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    norm = np.hypot(*axis_x)
    if norm < 1e-6:
        raise FaceAlignmentError("DiffSwap could not align degenerate face landmarks.")
    axis_x /= norm
    axis_x *= max(np.hypot(*eye_to_eye) * 2, np.hypot(*eye_to_mouth) * 1.8)
    axis_y = np.flipud(axis_x) * [-1, 1]
    center = eye_center + eye_to_mouth * .1
    quad = np.stack((center - axis_x - axis_y, center - axis_x + axis_y,
                     center + axis_x + axis_y, center + axis_x - axis_y))
    qsize = np.hypot(*axis_x) * 2
    image = image.convert("RGB")
    shrink = int(np.floor(qsize / 256 * .5))
    if shrink > 1:
        size = tuple(int(np.rint(dimension / shrink)) for dimension in image.size)
        image = image.resize(size, Image.Resampling.BICUBIC)
        quad /= shrink
        qsize /= shrink
    border = max(int(np.rint(qsize * .1)), 3)
    crop = (max(int(np.floor(quad[:, 0].min())) - border, 0),
            max(int(np.floor(quad[:, 1].min())) - border, 0),
            min(int(np.ceil(quad[:, 0].max())) + border, image.width),
            min(int(np.ceil(quad[:, 1].max())) + border, image.height))
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise FaceAlignmentError("DiffSwap face crop lies outside the input image.")
    if crop[2] - crop[0] < image.width or crop[3] - crop[1] < image.height:
        image = image.crop(crop)
        quad -= crop[:2]
    quad = (quad + .5).flatten()
    affine = (-(quad[0] - quad[6]) / 1024, -(quad[0] - quad[2]) / 1024, quad[0],
              -(quad[1] - quad[7]) / 1024, -(quad[1] - quad[3]) / 1024, quad[1])
    return image.transform((1024, 1024), Image.Transform.AFFINE, affine,
                            Image.Resampling.BICUBIC).resize(
                                (256, 256), Image.Resampling.BICUBIC)


def portrait_masks(landmarks):
    import cv2
    from scipy.spatial import ConvexHull, QhullError

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise FaceAlignmentError("DiffSwap needs 68 finite face landmarks.")

    def hull(indices):
        selected = points[indices]
        try:
            vertices = ConvexHull(selected).vertices
        except QhullError as error:
            raise FaceAlignmentError("DiffSwap could not form a face mask from these landmarks.") from error
        mask = np.zeros((256, 256), dtype=np.float32)
        cv2.fillPoly(mask, [selected[vertices].astype(np.int32)], 1)
        return mask

    organs = np.stack([hull(slice(a, b)) for a, b in ((36, 42), (42, 48),
                                                     (27, 36), (48, 68))])
    return organs, hull(slice(None))


def affine_theta(five_points):
    from third_party.diffswap.align_trans import (
        get_reference_facial_points, warp_and_crop_face,
    )
    points = np.asarray(five_points, dtype=np.float64)
    if (points.shape != (5, 2) or not np.isfinite(points).all()
            or np.linalg.matrix_rank(points - points.mean(0)) < 2):
        raise FaceAlignmentError("DiffSwap identity landmarks are invalid or degenerate.")
    reference = get_reference_facial_points(default_square=True)
    try:
        transform = warp_and_crop_face(None, points, reference,
                                        crop_size=(112, 112), return_tfm=True)
        inverse = np.linalg.inv(np.concatenate((transform, [[0, 0, 1]])))
    except np.linalg.LinAlgError as error:
        raise FaceAlignmentError("DiffSwap could not fit the identity crop transform.") from error
    source_normalizer = np.array([[2 / 255, 0, -1], [0, 2 / 255, -1], [0, 0, 1]])
    target_denormalizer = np.linalg.inv(
        np.array([[2 / 111, 0, -1], [0, 2 / 111, -1], [0, 0, 1]]))
    return (source_normalizer @ inverse @ target_denormalizer)[:2].astype(np.float32)


def blend_crop(generated, target, mask):
    """Author's 7-layer multiband reconstruction of the unmasked background."""
    import cv2
    from scipy.ndimage import correlate1d

    def reduce(image):
        kernel = np.array([1, 5, 8, 5, 1]) / 20
        return correlate1d(correlate1d(image, kernel, 0), kernel, 1)[::2, ::2]

    def expand(image, template):
        value = np.zeros_like(template)
        value[::2, ::2] = image
        kernel = np.array([1, 5, 8, 5, 1]) / 10
        return correlate1d(correlate1d(value, kernel, 0, mode="constant"),
                            kernel, 1, mode="constant")

    def pyramid(image):
        gaussian = [np.asarray(image).astype(np.int32)]
        for _ in range(7):
            gaussian.append(reduce(gaussian[-1]))
        laplacian = [a - expand(b, a) for a, b in zip(gaussian, gaussian[1:])]
        return laplacian + [gaussian[-1]], gaussian

    first, gaussian = pyramid(generated)
    second, _ = pyramid(target)
    masks = [cv2.resize(mask.astype(np.uint8), (level.shape[1], level.shape[0]))
             for level in gaussian]
    joined = [a * m[..., None] + b * (1 - m[..., None])
              for a, b, m in zip(first, second, masks)]
    result = joined[-1]
    for level in reversed(joined[:-1]):
        result = expand(result, level) + level
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


class DiffSwapAdapter:
    name = "DiffSwap"
    native_size = 256
    preprocessing = PREPROCESSING

    def __init__(self, root_dir, device=None):
        self.root = Path(root_dir).resolve()
        self.device = device
        self.model = self.detector = self.predictor = self.mtcnn = None
        self.steps = 200

    def load(self):
        if self.model is not None:
            return self
        status = readiness(self.root)
        if not status["ready"]:
            raise RuntimeError(status["message"])
        import dlib
        import torch
        from phantomseal_runtime import load_mtcnn
        from third_party.diffswap.inference import DiffSwapModel

        self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(weight_path(self.root, "diffswap", MODEL_FILES[2])))
            self.mtcnn = load_mtcnn(self.root, keep_all=True, device="cpu")
            self.model = DiffSwapModel.from_checkpoints(
                weight_path(self.root, "diffswap", MODEL_FILES[0]),
                weight_path(self.root, "diffswap", MODEL_FILES[1]), self.device)
        except BaseException:
            self.close()
            raise
        return self

    def _landmarks(self, rgb):
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = []
        for upsample in range(3):
            faces = self.detector(gray, upsample)
            if faces:
                break
        if not faces:
            raise FaceAlignmentError("DiffSwap could not detect a face for 68-point alignment.")
        face = max(faces, key=lambda item: item.width() * item.height())
        shape = self.predictor(rgb, face)
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)],
                         dtype=np.float32)

    def _portrait(self, image):
        import cv2
        import torch

        if not isinstance(image, Image.Image) or min(image.size) < 16:
            raise ValueError("DiffSwap needs a readable RGB portrait at least 16px wide and high.")
        rgb = np.asarray(image.convert("RGB"))
        while max(rgb.shape[:2]) > 2000:
            rgb = cv2.resize(rgb, None, fx=.5, fy=.5, interpolation=cv2.INTER_CUBIC)
        while min(rgb.shape[:2]) < 400:
            if max(rgb.shape[:2]) > 2000:
                raise FaceAlignmentError("DiffSwap needs a face portrait with a less extreme aspect ratio.")
            rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        crop = ffhq_crop(Image.fromarray(rgb), self._landmarks(rgb))
        points = self._landmarks(np.asarray(crop))
        boxes, probabilities, five = self.mtcnn.detect(crop, landmarks=True)
        if boxes is None or five is None or not len(boxes):
            raise FaceAlignmentError("DiffSwap could not detect five landmarks for the identity crop.")
        boxes, probabilities, five = (np.asarray(value, dtype=np.float32)
                                     for value in (boxes, probabilities, five))
        valid = [i for i, score in enumerate(probabilities)
                 if np.isfinite(score) and score >= .9 and np.isfinite(boxes[i]).all()
                 and np.isfinite(five[i]).all()]
        if not valid:
            raise FaceAlignmentError("DiffSwap could not detect clear identity landmarks.")
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        selected = max(valid, key=lambda i: area[i])
        organs, mask = portrait_masks(points)

        def tensor(value):
            return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32,
                                     device=self.device)[None]

        return {"crop": crop, "image": tensor(np.asarray(crop) / 127.5 - 1).permute(0, 3, 1, 2),
                "landmarks": tensor(points / 256), "theta": tensor(affine_theta(five[selected])),
                "organs": tensor(organs), "mask": tensor(mask), "mask_array": mask}

    def swap(self, source, target, seed=0):
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or not 0 <= seed < 2**32):
            raise ValueError("DiffSwap seed must be an integer in [0, 2**32).")
        if not isinstance(source, Image.Image) or not isinstance(target, Image.Image):
            raise ValueError("DiffSwap inputs must be PIL images.")
        if target.size != (224, 224):
            raise ValueError("Use a prepared 224x224 target face crop.")
        if min(source.size) < 16:
            raise ValueError("DiffSwap needs a source portrait at least 16px wide and high.")
        import torch

        self.load()
        with torch.no_grad():
            prepared_source = self._portrait(source)
            prepared_target = self._portrait(target)
            output = self.model.generate(prepared_source, prepared_target,
                                         seed=seed, steps=self.steps)
            if output.shape != (1, 3, 256, 256) or not torch.isfinite(output).all():
                raise RuntimeError("DiffSwap produced an invalid image.")
            pixels = (output[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return blend_crop(pixels, prepared_target["crop"], prepared_target["mask_array"])

    def close(self):
        self.model = self.detector = self.predictor = self.mtcnn = None
        # Adapter references are released; the caller manages shared CUDA cleanup.
