"""Official InfoSwap 512px smooth-checkpoint inference, with explicit local assets.

The author demo feeds OpenCV BGR pixels to the network. We convert RGB to BGR
at that boundary and convert its generated pixels back to RGB. Both portraits
are independently aligned using the author's 0.85-scaled square template.
"""
from __future__ import annotations

import gc
from pathlib import Path

from asset_paths import MTCNN_FILES, weight_path
from swap_adapters import FaceAlignmentError


REQUIRED_FILES = (
    "infoswap/model_128_ir_se50.pth",
    "infoswap/w_kernel_smooth/ckpt_ks_E.pth",
    "infoswap/w_kernel_smooth/ckpt_ks_G.pth",
    "infoswap/w_kernel_smooth/ckpt_ks_I.pth",
    *(f"infoswap/readout/readout_layer{i}.pth" for i in range(11)),
    *(f"mtcnn/{name}" for name in MTCNN_FILES),
)


def readiness(root_dir):
    missing = [name for name in REQUIRED_FILES if not weight_path(root_dir, name).is_file()
               or not weight_path(root_dir, name).stat().st_size]
    return {"ready": not missing, "message": "Ready" if not missing else "Missing: " + ", ".join(missing),
            "native_size": 512, "variant": "smooth", "stochastic": True}


def load_tensor_state(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError(f"InfoSwap requires a tensor state dictionary: {path.name}")
    return state


def readout_parameters(path, device):
    import torch
    state = load_tensor_state(path)
    if set(state) != {"m", "s", "n_samples", "neuron_nonzero"}:
        raise ValueError(f"Invalid InfoSwap readout statistics: {path.name}")
    samples = state["n_samples"].float()
    mean, variance, nonzero = state["m"].float(), state["s"].float(), state["neuron_nonzero"].float()
    if samples.numel() != 1 or samples.item() <= 1 or not torch.isfinite(samples).all():
        raise ValueError("Invalid InfoSwap statistics sample count.")
    if mean.ndim != 3 or variance.shape != mean.shape or nonzero.shape != mean.shape:
        raise ValueError("Invalid InfoSwap statistics shape.")
    if any(not torch.isfinite(value).all() for value in (mean, variance, nonzero)) or (variance < 0).any():
        raise ValueError("Non-finite or negative InfoSwap statistics.")
    return mean.to(device), (variance / (samples - 1)).sqrt().to(device), (nonzero / samples > .01).to(device)


def to_model_tensor(image, device):
    """Match the official demo's BGR pixels and [-1,1] normalization."""
    import numpy as np
    import torch
    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    return torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1


def from_model_tensor(value):
    import numpy as np
    import torch
    from PIL import Image
    if value.shape != (1, 3, 512, 512) or not torch.isfinite(value).all():
        raise RuntimeError("InfoSwap produced an invalid generated face.")
    bgr = ((value[0].detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))


class InfoSwapAdapter:
    native_size = 512

    def __init__(self, root_dir, device=None):
        self.root = Path(root_dir).resolve()
        self.device = device
        self.encoder = self.generator = self.decoder = self.iib = self.detector = None
        self.params = []

    def load(self):
        if self.generator is not None:
            return self
        available = readiness(self.root)
        if not available["ready"]:
            raise FileNotFoundError(available["message"])
        from scripts.runtime_environment import configure_environment
        configure_environment(self.root)
        import torch
        from phantomseal_runtime import load_mtcnn
        from third_party.infoswap.encoder128 import Backbone128
        from third_party.infoswap.aii_generator import AII512
        from third_party.infoswap.decoder512 import UnetDecoder512
        from third_party.infoswap.iib import IIB
        self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.encoder, self.generator, self.decoder = Backbone128(50, .6, "ir_se"), AII512(), UnetDecoder512()
            for model, filename in (
                (self.encoder, "model_128_ir_se50.pth"),
                (self.generator, "w_kernel_smooth/ckpt_ks_G.pth"),
                (self.decoder, "w_kernel_smooth/ckpt_ks_E.pth"),
            ):
                model.load_state_dict(load_tensor_state(weight_path(self.root, "infoswap", filename)), strict=True)
                model.eval().requires_grad_(False).to(self.device)
            self.params = [readout_parameters(weight_path(self.root, "infoswap", "readout", f"readout_layer{i}.pth"),
                                               self.device) for i in range(11)]
            channels = [values[0].shape[0] for values in self.params]
            self.iib = IIB(sum(channels), channels[:10], "cpu", smooth=True, kernel_size=1)
            self.iib.load_state_dict(load_tensor_state(weight_path(self.root, "infoswap", "w_kernel_smooth", "ckpt_ks_I.pth")),
                                     strict=True)
            self.iib.eval().requires_grad_(False).to(self.device)
            self.detector = load_mtcnn(self.root, device="cpu", keep_all=True)
        except BaseException:
            self.close()
            raise
        return self

    def align(self, image):
        import numpy as np
        from PIL import Image
        from third_party.infoswap.align_trans import FaceWarpException, get_reference_facial_points, warp_and_crop_face
        rgb = image.convert("RGB")
        boxes, probabilities, landmarks = self.detector.detect(rgb, landmarks=True)
        if boxes is not None:
            boxes, probabilities, landmarks = (np.asarray(values, dtype=np.float32)
                                               for values in (boxes, probabilities, landmarks))
        valid = [] if boxes is None else [
            i for i, score in enumerate(probabilities)
            if np.isfinite(score) and score >= .9 and np.isfinite(boxes[i]).all()
            and np.isfinite(landmarks[i]).all()
        ]
        if not valid:
            raise FaceAlignmentError("InfoSwap could not detect a clear face.")
        selected = max(valid, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        try:
            aligned = warp_and_crop_face(np.asarray(rgb), landmarks[selected],
                                         get_reference_facial_points(default_square=True), crop_size=(512, 512))
        except (FaceWarpException, np.linalg.LinAlgError) as error:
            raise FaceAlignmentError("InfoSwap could not align the detected landmarks.") from error
        except Exception as error:
            # Upstream exposes this one rank-deficiency case as a plain Exception.
            if error.args == ("cp2tform:twoUniquePointsReq",):
                raise FaceAlignmentError("InfoSwap detected degenerate face landmarks.") from error
            raise
        return Image.fromarray(aligned)

    def swap(self, source, target, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("InfoSwap seed must be an integer in [0, 2**32).")
        self.load()
        import torch
        import torch.nn.functional as functional
        source_tensor = to_model_tensor(self.align(source), self.device)
        target_tensor = to_model_tensor(self.align(target), self.device)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                identity = self.encoder(functional.interpolate(
                    torch.cat((source_tensor, target_tensor))[:, :, 37:475, 37:475],
                    size=(128, 128), mode="bilinear", align_corners=True), cache_feats=True)
                readout = [(self.encoder.features[i] - self.params[i][0]) / self.params[i][1].clamp_min(.01)
                           for i in range(11)]
                restricted = torch.zeros_like(identity)
                attributes, lambdas = [], []
                for i in range(10):
                    features = self.encoder.features[i]
                    mean, std, active = self.params[i]
                    latent, mask, _ = getattr(self.iib, f"iba_{i}")(
                        features, readout, m_r=mean, std_r=std, active_neurons=active)
                    restricted += self.encoder.restrict_forward(latent, i)
                    source_features, target_features = features[:1], features[1:]
                    source_mean = source_features.mean(dim=0)
                    # The author inference uses mean for BOTH these coefficients.
                    # Preserve its released inference behavior rather than changing the algorithm.
                    source_std = source_mean
                    noise = torch.randn(size=target_features.shape).to(self.device) * source_std + source_mean
                    attributes.append(target_features * (1 - mask[1:]) + mask[1:] * noise)
                    lambdas.append(mask[1:])
                attributes[0] = target_tensor
                attrs, attr_masks = self.decoder(attributes, lambs=lambdas, use_lambda=True)
                generated = self.generator((restricted / 10)[:1], attrs, attr_masks)
                return from_model_tensor(generated)
        finally:
            self.encoder.features = []

    def close(self):
        self.encoder = self.generator = self.decoder = self.iib = self.detector = None
        self.params = []
        gc.collect()
        if self.device is not None:
            import torch
            if torch.device(self.device).type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
