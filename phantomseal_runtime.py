"""Local single-photo PhantomSeal full identity-ensemble adapter.

The four-encoder objective and projected updates follow src/unify/defense.py
(MIT, Liangqin Ren, 2026). A differentiable aligned 256px face view preserves
the full output dimensions. This is a local adaptation, not a reproduction
of the paper's entire evaluation. Upstream notices are retained separately.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from asset_paths import MTCNN_FILES, weight_path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7),
                                         nn.BatchNorm2d(64), nn.ReLU())
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU())

    def forward(self, x):
        return self.down3(self.down2(self.down1(self.first_layer(x))))


def load_generator_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and
            isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("SimSwap generator must contain a tensor state dictionary.")
    return state


def image_tensor(image, device):
    array = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32) / 255


def load_mtcnn(root_dir, *, image_size=160, margin=0, keep_all=True, device="cpu"):
    """Pinned facenet-pytorch 2.5.3 detector with explicit local P/R/O weights.

    Its public MTCNN constructor always loads package-bundled weights. Build
    the same object/settings with pretrained=False networks to avoid that read.
    """
    from facenet_pytorch.models.mtcnn import MTCNN, PNet, RNet, ONet
    detector = MTCNN.__new__(MTCNN)
    nn.Module.__init__(detector)
    detector.image_size, detector.margin = image_size, margin
    detector.min_face_size, detector.thresholds, detector.factor = 20, [.6, .7, .7], .709
    detector.post_process, detector.select_largest = True, True
    detector.keep_all, detector.selection_method = keep_all, "largest"
    for name, model_type, filename in zip(("pnet", "rnet", "onet"), (PNet, RNet, ONet), MTCNN_FILES):
        path = weight_path(root_dir, "mtcnn", filename)
        if not path.is_file():
            raise FileNotFoundError(f"Missing local MTCNN detector weights: {path}. Run manage.cmd models.")
        network = model_type(pretrained=False)
        network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        setattr(detector, name, network)
    detector.device = device if device is not None else torch.device("cpu")
    return detector.eval().requires_grad_(False).to(detector.device)


def aligned_grid(landmarks, width, height, device, size=224):
    template = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], np.float32) * (size / 112)
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), template,
                                           method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Cannot align this face. Use a clear, front-facing portrait.")
    inverse = cv2.invertAffineTransform(matrix)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    sx = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
    sy = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
    grid = np.stack((2 * (sx + .5) / width - 1, 2 * (sy + .5) / height - 1), axis=-1)
    return torch.as_tensor(grid, device=device, dtype=torch.float32).unsqueeze(0)


ENSEMBLE_TERMS = {"simswap": (10000., 10000., .003),
                  "hififace_3d": (500., 5000., .03),
                  "hififace_id": (150., 15000., .0003),
                  "faceshifter": (35000., 1000000., .003)}


def ensemble_loss(name, feature, self_reference, cloak_reference):
    self_weight, cloak_weight, cap = ENSEMBLE_TERMS[name]
    own = (feature.float() - self_reference.float()).square().flatten(1).mean(1).clamp(max=cap)
    cloak = (feature.float() - cloak_reference.float()).square().flatten(1).mean(1)
    return (-self_weight * own + cloak_weight * cloak).mean()


def sequential_objective(delta, original, view_fn, branches, references):
    """Sum exact branch gradients, then leave the single sign update to caller."""
    if set(branches) != set(ENSEMBLE_TERMS):
        raise ValueError("The full ensemble requires all four encoder branches.")
    gradient = torch.zeros_like(delta)
    losses = []
    face = view_fn((original + delta).clamp(0, 1))
    loss = 1000 * (face - references["_clean_face"]).square().mean()
    partial, = torch.autograd.grad(loss, delta)
    gradient.add_(partial)
    losses.append(loss.detach())
    del face, loss, partial
    for name, branch in branches.items():
        face = view_fn((original + delta).clamp(0, 1))
        feature = branch(face)
        loss = ensemble_loss(name, feature, references[name][0], references[name][1])
        partial, = torch.autograd.grad(loss, delta)
        gradient.add_(partial)
        losses.append(loss.detach())
        del face, feature, loss, partial
    total = torch.stack(losses).sum()
    if not torch.isfinite(total) or not torch.isfinite(gradient).all():
        raise RuntimeError("PhantomSeal ensemble produced non-finite loss or gradients.")
    return float(total), gradient


class PhantomSealProtector:
    def __init__(self, root_dir, device=None):
        self.root = Path(root_dir).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.arcface = self.encoder = self.facenet = self.detector = None
        self.branches = {}
        self._loaded = False

    def load_models(self, progress=None):
        if self._loaded:
            return
        try:
            self._load_models(progress)
        except BaseException:
            self.close()
            raise

    def _load_models(self, progress):
        from scripts.runtime_environment import configure_environment
        configure_environment(self.root)
        if progress:
            progress(.01, "Loading PhantomSeal ensemble")
        from ensemble_encoders import load_ensemble
        from facenet_pytorch import InceptionResnetV1
        self.branches = load_ensemble(self.root, self.device)
        self.arcface = self.branches["simswap"].backbone
        self.facenet = InceptionResnetV1(pretrained=None, classify=False)
        face_state = torch.load(weight_path(self.root, "facenet", "facenet_vggface2.pt"), map_location="cpu", weights_only=True)
        # Official pretrained files include the classification head.
        self.facenet.load_state_dict({k: v for k, v in face_state.items() if not k.startswith("logits.")}, strict=True)
        del face_state
        self.detector = load_mtcnn(self.root, image_size=160, margin=0, keep_all=True, device="cpu")
        self.facenet.eval().requires_grad_(False).to(self.device)
        self._loaded = True

    def close(self):
        self.arcface = self.encoder = self.facenet = self.detector = None
        self.branches = {}
        self._loaded = False
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def detect(self, image):
        boxes, probabilities, landmarks = self.detector.detect(image, landmarks=True)
        if boxes is None:
            raise ValueError("No face detected. Use a clear portrait with a visible face.")
        valid = np.where(np.asarray(probabilities) >= .9)[0]
        if not len(valid):
            raise ValueError("Face detection confidence is too low. Use a clearer portrait.")
        index = max(valid, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        return boxes[index], landmarks[index], len(boxes)

    def face_embedding(self, image, box):
        # Match FaceNet's public MTCNN extraction and fixed normalization.
        face = self.detector.extract(image, np.asarray([box]), save_path=None)
        with torch.no_grad():
            return self.facenet(face.to(self.device)).detach()

    def cloak_index(self, progress=None):
        from phantomseal_assets import configuration, bank_images
        from importlib.metadata import version
        config = configuration(self.root)
        bank = config["cloak_dir"]
        paths = bank_images(bank)
        if len(paths) < config["expected_cloaks"]:
            raise ValueError(f"Full cloak bank is incomplete: {len(paths)}/{config['expected_cloaks']} images.")
        digest = hashlib.sha256((str(bank) + "|mtcnn160-largest-confidence.9-v2|" + version("facenet-pytorch")).encode())
        for path in [weight_path(self.root, "facenet", "facenet_vggface2.pt"),
                     *(weight_path(self.root, "mtcnn", name) for name in MTCNN_FILES), *paths]:
            digest.update(str(path).encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        fingerprint = digest.hexdigest()
        cache = weight_path(self.root, "cache", "phantomseal", "full-cloak-index.json")
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            entries = saved["entries"]
            allowed = {path.relative_to(bank).as_posix() for path in paths}
            if saved["fingerprint"] == fingerprint and entries and len(entries) <= len(paths):
                names = [entry["path"] for entry in entries]
                embeds = np.asarray([entry["embedding"] for entry in entries], dtype=np.float32)
                points = np.asarray([entry["landmarks"] for entry in entries], dtype=np.float32)
                if (len(set(names)) == len(names) and set(names).issubset(allowed) and
                    embeds.shape == (len(entries), 512) and points.shape == (len(entries), 5, 2) and
                    np.isfinite(embeds).all() and np.isfinite(points).all()):
                    self._bank_fingerprint = fingerprint
                    return bank, entries, len(paths), True
        except (OSError, ValueError, TypeError, KeyError):
            pass
        entries = []
        for index, path in enumerate(paths):
            with Image.open(path) as opened:
                candidate = opened.convert("RGB")
                try:
                    candidate_box, landmarks, _ = self.detect(candidate)
                    embedding = self.face_embedding(candidate, candidate_box)[0].detach().float().cpu().tolist()
                except ValueError:
                    continue
            entries.append({"path": path.relative_to(bank).as_posix(), "landmarks": np.asarray(landmarks).tolist(),
                            "embedding": embedding})
            if progress and ((index + 1) % 5 == 0 or index + 1 == len(paths)):
                progress(.04 + .10 * (index + 1) / len(paths), f"Indexing full cloak bank: {index + 1}/{len(paths)}")
        if not entries:
            raise ValueError("The full PhantomSeal cloak bank has no detectable faces.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps({"fingerprint": fingerprint, "bank": config["bank_name"],
                "total_images": len(paths), "entries": entries}, allow_nan=False), encoding="utf-8")
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
        self._bank_fingerprint = fingerprint
        return bank, entries, len(paths), False

    def select_cloak(self, image, box, progress=None):
        bank, entries, total, cache_hit = self.cloak_index(progress)
        embedding = self.face_embedding(image, box).detach().float().cpu()
        candidates = torch.tensor([entry["embedding"] for entry in entries], dtype=torch.float32)
        distances = torch.linalg.vector_norm(candidates - embedding, dim=1)
        preferred = torch.where(distances > 1.04)[0]
        if not len(preferred):
            raise ValueError("No sufficiently different cloak identity was found in the full bank (minimum distance 1.04).")
        choices = preferred
        index = int(choices[(distances[choices] - 1.20).abs().argmin()])
        selected = entries[index]
        with Image.open(bank / selected["path"]) as opened:
            cloak = opened.convert("RGB")
        self._cloak_index_metadata = {"total": total, "usable": len(entries), "cache_hit": cache_hit,
                                      "fingerprint": self._bank_fingerprint}
        return cloak, np.asarray(selected["landmarks"], np.float32), selected["path"], float(distances[index]), total

    def identity(self, face):
        mean = face.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = face.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        return F.normalize(self.arcface(F.interpolate((face - mean) / std, (112, 112), mode="nearest")), dim=1)

    @staticmethod
    def view(image, grid):
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def protect(self, input_path, output_dir, iterations=1000, progress=None):
        from runtime_contracts import number
        iterations = number(iterations, "phantom_iterations", 10, 2000, integer=True)
        started = time.perf_counter()
        self.load_models(progress)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with Image.open(input_path) as opened:
            image = opened.convert("RGB")
        if progress:
            progress(.03, "Detecting face and choosing a cloak")
        box, landmarks, faces = self.detect(image)
        cloak, cloak_landmarks, cloak_name, cloak_distance, bank_size = self.select_cloak(image, box, progress)
        grid = aligned_grid(landmarks, *image.size, self.device, size=256)
        original = image_tensor(image, self.device)
        cloak_tensor = self.view(image_tensor(cloak, self.device), aligned_grid(cloak_landmarks, *cloak.size, self.device, size=256))
        with torch.no_grad():
            clean_face = self.view(original, grid)
            references = {name: (branch(clean_face).detach(), branch(cloak_tensor).detach())
                          for name, branch in self.branches.items()}
            references["_clean_face"] = clean_face.detach()
            clean_identity, cloak_identity = references["simswap"]
            step = .003 * (clean_face.max() - clean_face.min()).clamp_min(1e-6) / 2
        delta = torch.zeros_like(original)
        limits = original.new_tensor([.075, .030, .075]).view(1, 3, 1, 1)
        best = delta.clone()
        best_loss = float("inf")
        for index in range(iterations):
            delta = delta.detach().requires_grad_(True)
            score, gradient = sequential_objective(delta, original, lambda x: self.view(x, grid), self.branches, references)
            if score < best_loss:
                best_loss = score
                best = delta.detach().clone()
            with torch.no_grad():
                delta = (delta - step * gradient.sign()).clamp(-limits, limits)
                delta = (original + delta).clamp(0, 1) - original
            if progress and ((index + 1) % 10 == 0 or index + 1 == iterations):
                progress(.15 + .83 * (index + 1) / iterations, f"PhantomSeal ensemble: {index + 1}/{iterations}")
            del gradient
        with torch.no_grad():
            protected = ((original + best).clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
            quantized = torch.from_numpy(protected.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255
            protected_id = self.identity(self.view(quantized, grid))
            clean_cosine = float(F.cosine_similarity(protected_id, clean_identity))
            cloak_cosine = float(F.cosine_similarity(protected_id, cloak_identity))
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "phantomseal.png"
        Image.fromarray(protected).save(output)
        difference = np.abs(protected.astype(np.int16) - np.asarray(image).astype(np.int16))
        report = {"output_path": str(output), "method": "phantomseal_ensemble_aligned_multiscale",
                  "ensemble_branches": list(self.branches), "loss_weights": ENSEMBLE_TERMS,
                  "surrogate_models": {name: {"checkpoint": getattr(branch, "weight_path", None),
                      "prefix": getattr(branch, "weight_prefix", None)} for name, branch in self.branches.items()},
                  "max_pixel_change": int(difference.max()),
                  "max_pixel_change_rgb": [int(v) for v in difference.max(axis=(0, 1))],
                  "iterations": iterations, "analysis_size": 256, "face_count": faces,
                  "protected_face": "largest", "cloak": cloak_name, "cloak_bank_size": bank_size,
                  "cloak_bank": "PhantomSeal VGGFace2 mixed full", "cloak_index": self._cloak_index_metadata,
                  "cloak_selection_distance": cloak_distance,
                  "rgb_noise_limits": [19.125, 7.65, 19.125], "elapsed_seconds": time.perf_counter() - started,
                  "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 1024**3 if self.device.type == "cuda" else 0,
                  "surrogate_identity_cosine": clean_cosine, "surrogate_cloak_cosine": cloak_cosine,
                  "note": "Full four-encoder objective and original mixed cloak bank, adapted to aligned full-resolution images; surrogate scores do not certify face-swap protection."}
        (folder / "phantomseal_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
