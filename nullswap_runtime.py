"""NullSwap-style defense using PhantomSeal's published reconstruction.

The checkpoint's native 256px aligned face produces a residual bounded to 8/255.
Its residual is mapped onto the photograph and scaled by user strength (default 1).
This adaptation is not an official NullSwap author-model reproduction.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

from asset_paths import MTCNN_FILES, weight_path
from runtime_contracts import nullswap_parameters


CHECKPOINT = "nullswap/epoch_0001_step_012000.pth"
EPSILON = 8 / 255


def readiness(root_dir):
    files = [CHECKPOINT, *(f"mtcnn/{name}" for name in MTCNN_FILES)]
    missing = [name for name in files if not weight_path(root_dir, name).is_file()
               or not weight_path(root_dir, name).stat().st_size]
    return {"ready": not missing, "message": "Ready" if not missing else "Missing: " + ", ".join(missing),
            "native_size": 256, "implementation": "PhantomSeal NullSwap reconstruction",
            "perturbation_gain": 1.0}


def face_transform(landmarks):
    """Return the full-image to native-face transform, failing on bad geometry."""
    import cv2
    import numpy as np
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("NullSwap requires five finite face landmarks.")
    template = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32) * (256 / 112)
    matrix, _ = cv2.estimateAffinePartial2D(points, template, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
        raise ValueError("Cannot align this face for NullSwap. Use a clear portrait.")
    return matrix


def map_residual(original, native_delta, image_to_face, strength=1.0):
    """Scale only the residual; strength 1 preserves the original 8/255 default."""
    strength, _ = nullswap_parameters(strength)
    import cv2
    import numpy as np
    clean = np.asarray(original, dtype=np.uint8)
    delta = np.asarray(native_delta, dtype=np.float32)
    if delta.shape != (256, 256, 3) or not np.isfinite(delta).all():
        raise RuntimeError("NullSwap produced an invalid perturbation.")
    if float(np.abs(delta).max()) > EPSILON + 1e-6:
        raise RuntimeError("NullSwap exceeded its native 8/255 perturbation bound.")
    matrix = np.asarray(image_to_face)
    if (matrix.shape != (2, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix[:, :2])) < 1e-8):
        raise ValueError("Invalid NullSwap face transform.")
    full_delta = cv2.warpAffine(delta, cv2.invertAffineTransform(matrix),
                               (clean.shape[1], clean.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    full_delta = np.clip(full_delta, -EPSILON, EPSILON) * strength
    return np.rint(np.clip(clean.astype(np.float32) + 255 * full_delta, 0, 255)).astype(np.uint8)


class NullSwapProtector:
    def __init__(self, root_dir, device=None):
        self.root = Path(root_dir).resolve()
        self.device = device
        self.model = self.detector = None

    def load_models(self, progress=None):
        if self.model is not None:
            return
        available = readiness(self.root)
        if not available["ready"]:
            raise FileNotFoundError(available["message"])
        from scripts.runtime_environment import configure_environment
        configure_environment(self.root)
        import torch
        from phantomseal_runtime import load_mtcnn
        from third_party.nullswap.model import NullSwap
        self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if progress:
            progress(.05, "Loading NullSwap reconstruction")
        try:
            checkpoint = torch.load(weight_path(self.root, CHECKPOINT), map_location="cpu", weights_only=True)
            state = checkpoint.get("generator") if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict) or not state or not all(
                    isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
                raise ValueError("NullSwap checkpoint must contain a tensor generator state dictionary.")
            model = NullSwap(epsilon=EPSILON)
            model.load_state_dict(state, strict=True)
            del checkpoint, state
            self.detector = load_mtcnn(self.root, device="cpu", keep_all=True)
            self.model = model.eval().requires_grad_(False).to(self.device)
        except BaseException:
            self.close()
            raise

    def protect(self, input_path, output_dir, max_size=1280, progress=None, strength=1.0, seed=0):
        from image_io import fit_rgb
        from runtime_contracts import number
        max_size = number(max_size, "max_size", 128, 1280, integer=True)
        strength, seed = nullswap_parameters(strength, seed)
        image = fit_rgb(input_path, max_size)
        started = time.perf_counter()
        self.load_models(progress)
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if progress:
            progress(.15, "Aligning the largest face for NullSwap")
        boxes, probabilities, landmarks = self.detector.detect(image, landmarks=True)
        if boxes is not None:
            boxes, probabilities, landmarks = (np.asarray(values, dtype=np.float32)
                                               for values in (boxes, probabilities, landmarks))
        valid = [] if boxes is None else [
            i for i, score in enumerate(probabilities)
            if np.isfinite(score) and score >= .9 and np.isfinite(boxes[i]).all()
            and np.isfinite(landmarks[i]).all()
        ]
        if not valid:
            raise ValueError("No clear face detected for NullSwap. Use a visible, front-facing portrait.")
        index = max(valid, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        matrix = face_transform(landmarks[index])
        face = cv2.warpAffine(np.asarray(image), matrix, (256, 256), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        native = torch.from_numpy(face.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255
        if progress:
            progress(.35, "Generating the NullSwap perturbation")
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            output = self.model(native)
            cloak = output["cloak"]
            if cloak.shape != native.shape or not torch.isfinite(cloak).all():
                raise RuntimeError("NullSwap produced an invalid protected face.")
            # Preserve the sampled residual; user strength is applied only after inverse warping.
            delta = (cloak - native)[0].permute(1, 2, 0).cpu().numpy()
        if progress:
            progress(.85, "Mapping the perturbation to the full image")
        protected = map_residual(image, delta, matrix, strength=strength)
        folder = Path(output_dir).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        source_path, output_path = folder / "original.png", folder / "nullswap.png"
        image.save(source_path)
        Image.fromarray(protected).save(output_path)
        difference = np.abs(protected.astype(np.int16) - np.asarray(image).astype(np.int16))
        report = {
            "source_path": str(source_path), "output_path": str(output_path),
            "method": "nullswap", "implementation": "PhantomSeal NullSwap reconstruction",
            "width": image.width, "height": image.height, "analysis_size": 256,
            "face_count": len(valid), "protected_face": "largest", "perturbation_gain": strength,
            "native_noise_limit": 8, "noise_limit_before_rounding": 8 * strength,
            "noise_limit": math.ceil(8 * strength), "seed": seed, "max_pixel_change": int(difference.max()),
            "max_pixel_change_rgb": [int(value) for value in difference.max(axis=(0, 1))],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 1024**3 if devices else 0.,
            "note": "PhantomSeal's NullSwap reconstruction; aligned 256px residual mapped to full resolution. "
                    f"User strength {strength:g}; seed {seed}. Defaults remain gain 1 and seed 0. "
                    "Protection efficacy requires independent tests.",
        }
        (folder / "nullswap_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if progress:
            progress(1., "NullSwap protected PNG saved")
        return report

    def close(self):
        self.model = self.detector = None
        gc.collect()
        if self.device is not None:
            import torch
            if torch.device(self.device).type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
