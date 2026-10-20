"""Offline UniFace swapping, preserving the author's image/tensor convention."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from asset_paths import project_path, weight_path


MODEL_PATH = ("uniface", "500000.pt")
ARCHITECTURE = {
    "size": 256,
    "latent_channel_size": 512,
    "latent_spatial_size": 8,
    "lr_mul": 1,
    "channel_multiplier": 2,
    "normalize_mode": "LayerNorm",
    "small_generator": False,
}
PREPROCESSING = {
    "source": "Official inference convention: supplied RGB portrait resized with PIL bilinear to 256x256, normalized to [-1,1]; no landmark alignment.",
    "target": "Prepared 224x224 RGB demo target resized with PIL bilinear to 256x256 and normalized to [-1,1].",
    "output": "Native 256x256 RGB, [-1,1] clamped and rounded to 8-bit, as torchvision.save_image(normalize=True, value_range=(-1,1)).",
    "precision": "Float32 inference with pure PyTorch StyleGAN operators; upstream demo uses float16.",
}


def readiness(root_dir):
    """Check local assets without importing Torch or loading any model."""
    try:
        root = Path(root_dir).resolve()
        path = weight_path(root, *MODEL_PATH)
        implementation = project_path(root, "third_party/uniface/model.py")
        operators = project_path(root, "third_party/uniface/ops.py")
        missing = [p.relative_to(root).as_posix()
                   for p in (path, implementation, operators)
                   if not p.is_file() or p.stat().st_size == 0]
    except (ValueError, OSError) as error:
        return {"ready": False, "message": str(error)}
    if missing:
        return {"ready": False, "message": "UniFace is missing: " + ", ".join(missing)}
    return {"ready": True, "message": "UniFace swapping checkpoint is available locally."}


def _image_tensor(image, device):
    import numpy as np
    import torch
    # torchvision Resize((256,256)) uses PIL bilinear for a PIL RGB image.
    resized = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(resized, dtype=np.uint8, copy=True))
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1)


def _output_image(tensor):
    import numpy as np
    import torch
    if tensor.shape != (1, 3, 256, 256) or not torch.isfinite(tensor).all():
        raise RuntimeError("UniFace returned an invalid or non-finite image.")
    array = tensor.detach().clamp(-1, 1).add(1).div(2).mul(255).add(.5)
    array = array.clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(array))


def _load_checkpoint(path):
    import argparse
    import torch
    # The released checkpoint contains training metadata in argparse.Namespace.
    # This audited standard-library container is the only allowed pickle class.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("UniFace checkpoint must contain a dictionary.")
    metadata = checkpoint.get("train_args")
    if not isinstance(metadata, (argparse.Namespace, dict)):
        raise ValueError("UniFace checkpoint has no supported architecture metadata.")
    metadata = vars(metadata) if isinstance(metadata, argparse.Namespace) else metadata
    for name, expected in ARCHITECTURE.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Unsupported UniFace architecture setting: {name}.")
    for key in ("g_ema", "e_ema"):
        state = checkpoint.get(key)
        if not isinstance(state, dict) or not state or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError(f"UniFace checkpoint has no valid {key} state dictionary.")
    return checkpoint["g_ema"], checkpoint["e_ema"]


class UniFaceAdapter:
    """The released swapping generator and encoder; no training dependencies."""

    name = "UniFace"
    native_size = 256
    preprocessing = PREPROCESSING

    def __init__(self, root_dir, device=None):
        self.root = Path(root_dir).resolve()
        self.device = device
        self.generator = None
        self.encoder = None

    def load(self):
        if self.generator is not None and self.encoder is not None:
            return self
        state = readiness(self.root)
        if not state["ready"]:
            raise FileNotFoundError(state["message"])
        import torch
        from third_party.uniface.model import Encoder_return_32, Generator_globalatt_return_32

        self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        generator_state, encoder_state = _load_checkpoint(weight_path(self.root, *MODEL_PATH))
        # Construct on CPU without changing the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            generator = Generator_globalatt_return_32(
                256, 512, 8, lr_mul=1, channel_multiplier=2,
                normalize_mode="LayerNorm", small_generator=False,
            )
            encoder = Encoder_return_32(256, 512, 8, channel_multiplier=2)
        generator.load_state_dict(generator_state, strict=True)
        encoder.load_state_dict(encoder_state, strict=True)
        self.generator = generator.eval().requires_grad_(False).to(self.device)
        self.encoder = encoder.eval().requires_grad_(False).to(self.device)
        return self

    def swap(self, source, target, seed=0):
        # The selected architecture contains no forward-time random noise.
        # The seed parameter is accepted for the common adapter interface.
        if not isinstance(source, Image.Image) or not isinstance(target, Image.Image):
            raise ValueError("UniFace inputs must be PIL images.")
        if target.size != (224, 224):
            raise ValueError("Use a prepared 224x224 target face crop.")
        if min(source.size) < 1:
            raise ValueError("Source portrait is empty.")
        self.load()
        import torch
        source_tensor = _image_tensor(source, self.device)
        target_tensor = _image_tensor(target, self.device)
        with torch.inference_mode():
            codes, features = self.encoder(torch.cat((target_tensor, source_tensor), dim=0))
            target_features = [torch.chunk(feature, 2, dim=0)[0]
                               for feature in features][::-1]
            target_code, source_code = torch.chunk(codes, 2, dim=0)
            result = self.generator([target_code, source_code, target_features])
        return _output_image(result)

    def close(self):
        self.generator = None
        self.encoder = None
