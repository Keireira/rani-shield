"""Four frozen feature branches used by PhantomSeal's upstream unify objective.

load_ensemble(root, device) returns ordinary differentiable nn.Modules named
simswap, faceshifter, hififace_id, hififace_3d. Inputs are RGB float [0,1],
normally [B,3,256,256]; input gradients remain enabled. No GAN decoder,
discriminator, Lightning training wrapper, BFM renderer, or metric model loads.
"""
from __future__ import annotations

import io
import pickletools
import zipfile
from collections.abc import Mapping
from pathlib import Path
from asset_paths import project_path

import torch
from torch import nn
import torch.nn.functional as F

from third_party.phantomseal.full.deep3d import ReconNetWrapper
from third_party.phantomseal.full.faceshifter import Backbone
from third_party.phantomseal.full.iresnet import iresnet100


def _input(image):
    if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError("Feature input must be a floating RGB [B,3,H,W] tensor.")
    return image


def _image256(image):
    image = _input(image)
    if image.shape[-2:] != (256, 256):
        return F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
    return image


class SimSwapFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        image = _input(image)
        mean = image.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = image.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        image = F.interpolate((image - mean) / std, size=(112, 112), mode="nearest")
        return F.normalize(self.backbone(image.contiguous()), p=2, dim=1)


class FaceShifterFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        image = _image256(image)
        face = F.interpolate(image[:, :, 19:237, 19:237], size=(112, 112),
                             mode="bilinear", align_corners=True)
        # Upstream unify passes raw [0,1]; Backbone itself normalizes output.
        return self.backbone(face.contiguous())


class HifiFaceIDFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        image = _image256(image)
        face = F.interpolate((image - .5) / .5, size=(112, 112),
                             mode="bilinear", align_corners=False)
        return F.normalize(self.backbone(face.contiguous()), p=2, dim=-1)


class HifiFace3DFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        # First 80 raw shape coefficients; no normalization or BFM rendering.
        return self.backbone(_image256(image).contiguous())[:, :80]


def checkpoint_directory(root_dir):
    from phantomseal_assets import configuration
    return configuration(root_dir)["checkpoint_dir"]


def ensemble_paths(root_dir):
    root = Path(root_dir).resolve()
    checkpoints = checkpoint_directory(root)
    simswap = checkpoints / "simswap" / "arcface_checkpoint.tar"
    return {name: project_path(root, path) for name, path in {
        "simswap": simswap,
        "faceshifter": checkpoints / "faceshifter" / "model_ir_se50.pth",
        "hififace": checkpoints / "hififace" / "hififace_opensource_299999.ckpt",
    }.items()}


def _safe_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ensemble checkpoint: {path}")
    if zipfile.is_zipfile(path):
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    else:
        # The supplied FaceShifter state dict uses torch's legacy format.
        # Inspect bounded pickle metadata without unpickling any object.
        allowed = {"collections OrderedDict", "torch FloatStorage",
                   "torch._utils _rebuild_tensor_v2"}
        with path.open("rb") as stream:
            metadata = io.BytesIO(stream.read(8 * 1024 * 1024))
        unsafe = []
        for section in range(5):
            try:
                operations = list(pickletools.genops(metadata))
            except (ValueError, EOFError) as error:
                raise ValueError(f"Invalid legacy tensor metadata in {path.name}") from error
            for opcode, argument, _ in operations:
                if opcode.name == "GLOBAL" and argument not in allowed:
                    unsafe.append(argument)
                if opcode.name in {"STACK_GLOBAL", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4"}:
                    unsafe.append(opcode.name)
            literals = [argument for opcode, argument, _ in operations
                        if opcode.name in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}]
            if section == 0 and literals != [torch.serialization.MAGIC_NUMBER]:
                raise ValueError("Legacy checkpoint has an invalid torch magic number.")
            if section == 1 and literals != [torch.serialization.PROTOCOL_VERSION]:
                raise ValueError("Legacy checkpoint has an invalid torch protocol version.")
    if unsafe:
        raise ValueError(f"Unexpected pickle globals in {path.name}: {sorted(unsafe)}")
    # Do not add globals or fall back to unrestricted pickle for state dicts.
    return torch.load(path, map_location="cpu", weights_only=True)


def _tensor_state(state, name):
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must contain a nonempty state dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError(f"{name} state dictionary must contain only named tensors.")
    return state


def _prefix_state(state, prefix):
    selected = {key[len(prefix):]: value for key, value in state.items()
                if isinstance(key, str) and key.startswith(prefix)}
    return _tensor_state(selected, prefix)


def load_ensemble(root_dir, device="cpu"):
    """Strictly load the same four pretrained heads used by upstream unify.

    The final HiFiFace checkpoint overrides the standalone initialization
    checkpoints in upstream HifiFace, so its exact two encoder prefixes are used.
    SimSwap's serialized module uses the existing audited 13-class safe loader.
    """
    from protector import FaceShieldProtector

    paths = ensemble_paths(root_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing ensemble checkpoint: {path}")
    simswap = FaceShieldProtector._load_arcface(paths["simswap"])
    faceshifter = Backbone(50, 0.6, "ir_se")
    face_state = _tensor_state(_safe_checkpoint(paths["faceshifter"]), "FaceShifter")
    faceshifter.load_state_dict(face_state, strict=True)
    del face_state

    full = _safe_checkpoint(paths["hififace"])
    if not isinstance(full, Mapping) or "state_dict" not in full:
        raise ValueError("The HiFiFace Lightning checkpoint has no state_dict.")
    state = _tensor_state(full["state_dict"], "HiFiFace")
    hififace_id = iresnet100(pretrained=False, fp16=False)
    hififace_3d = ReconNetWrapper()
    hififace_id.load_state_dict(_prefix_state(state, "generator.id_extractor.f_id."), strict=True)
    hififace_3d.load_state_dict(_prefix_state(state, "generator.id_extractor.f_3d."), strict=True)
    del state, full

    branches = {
        "simswap": SimSwapFeatures(simswap),
        "faceshifter": FaceShifterFeatures(faceshifter),
        "hififace_id": HifiFaceIDFeatures(hififace_id),
        "hififace_3d": HifiFace3DFeatures(hififace_3d),
    }
    for name, branch in branches.items():
        branch.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
        branch.weight_path = str(paths["hififace"] if name.startswith("hififace") else paths[name])
    branches["hififace_id"].weight_prefix = "generator.id_extractor.f_id."
    branches["hififace_3d"].weight_prefix = "generator.id_extractor.f_3d."
    return branches
