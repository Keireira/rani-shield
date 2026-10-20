"""Inference-only assembly of the released DiffSwap 256px model.

Architecture and sampling equations follow wl-zhao/DiffSwap, pinned in
PROVENANCE.json. This omits Lightning, training losses, datasets and downloads.
Source latent conditioning follows the corrected PhantomSeal runtime wrapper.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CrossAttention
from .autoencoder_modules import Decoder, Encoder
from .iresnet import iresnet100
from .openaimodel import UNetModel


class VectorQuantizer(nn.Module):
    """Inference equivalent of taming VectorQuantizer2's nearest-code lookup."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8192, 3)

    def forward(self, latent):
        pixels = latent.permute(0, 2, 3, 1).contiguous()
        flat = pixels.reshape(-1, 3)
        weights = self.embedding.weight
        # Chunk only the independent spatial rows; the nearest-code result is
        # the same as the full (4096 x 8192) distance matrix.
        indices = []
        weight_norm = weights.square().sum(dim=1)
        for chunk in flat.split(1024):
            distance = (chunk.square().sum(dim=1, keepdim=True) + weight_norm
                        - 2 * chunk @ weights.T)
            indices.append(distance.argmin(dim=1))
        quantized = self.embedding(torch.cat(indices)).reshape(pixels.shape)
        return quantized.permute(0, 3, 1, 2).contiguous()


class FirstStage(nn.Module):
    def __init__(self):
        super().__init__()
        config = dict(double_z=False, z_channels=3, resolution=256,
                      in_channels=3, out_ch=3, ch=128, ch_mult=(1, 2, 4),
                      num_res_blocks=2, attn_resolutions=(), dropout=0.0)
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        self.quantize = VectorQuantizer()
        self.quant_conv = nn.Conv2d(3, 3, 1)
        self.post_quant_conv = nn.Conv2d(3, 3, 1)

    def encode(self, image):
        return self.quant_conv(self.encoder(image))

    def decode(self, latent):
        return self.decoder(self.post_quant_conv(self.quantize(latent)))


class FaceConditioner(nn.Module):
    """Upstream FaceEmbedder with explicit, separately loaded ArcFace weights."""

    def __init__(self):
        super().__init__()
        self.face_model = iresnet100(fp16=False)
        self.landmark_encoder = nn.Sequential(nn.Linear(136, 1024), nn.GELU(),
                                             nn.Linear(1024, 256))
        self.id_fc = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))
        self.organ_fc = nn.Sequential(nn.Linear(3, 256), nn.GELU(), nn.Linear(256, 256))
        self.organ_norm = nn.LayerNorm(256)
        self.organ_attention = CrossAttention(query_dim=256, context_dim=256,
                                              dim_head=32, heads=4)

    def forward(self, source_image, source_latent, source_masks, source_theta,
                target_landmarks):
        # Merge the two eye masks exactly as the checkpoint's configuration.
        masks = source_masks[:, 1:].clone()
        masks[:, 0] = torch.logical_or(source_masks[:, 0], source_masks[:, 1])
        masks = F.interpolate(masks.float(), source_latent.shape[-2:], mode="nearest")
        organ = torch.einsum("bchw,bkhw->bkc", source_latent, masks)
        organ = organ / (masks.sum(dim=(-1, -2))[..., None] + 1e-6)
        organ = self.organ_fc(organ)
        organ = organ + self.organ_attention(self.organ_norm(organ))

        # Upstream uses align_corners=False (PyTorch default) for both calls.
        grid = F.affine_grid(source_theta, (source_image.shape[0], 3, 112, 112),
                             align_corners=False)
        identity_image = F.grid_sample(source_image, grid, align_corners=False)
        identity = self.id_fc(self.face_model(identity_image))
        landmarks = self.landmark_encoder(target_landmarks.reshape(-1, 136))
        return torch.cat((torch.stack((landmarks, identity), dim=1), organ), dim=1)


def read_state(path):
    """Load tensor dictionaries on CPU; no allowlisting or pickle fallback."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("DiffSwap checkpoint must contain a tensor state dictionary.")
    return state


def _subset(state, prefix):
    return {name[len(prefix):]: value for name, value in state.items()
            if name.startswith(prefix)}


def _load_exact(module, state, label):
    expected = module.state_dict()
    if expected.keys() != state.keys():
        missing = sorted(expected.keys() - state.keys())
        extra = sorted(state.keys() - expected.keys())
        raise ValueError(f"{label} checkpoint keys differ: missing={missing[:5]}, extra={extra[:5]}")
    for key, value in state.items():
        if expected[key].shape != value.shape:
            raise ValueError(f"{label} checkpoint has invalid tensor shape for {key}.")
    module.load_state_dict(state, strict=True, assign=True)


class DiffSwapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNetModel(image_size=64, in_channels=3, out_channels=3,
                             model_channels=224, attention_resolutions=(8, 4, 2),
                             num_res_blocks=2, channel_mult=(1, 2, 3, 4),
                             num_head_channels=32, use_spatial_transformer=True,
                             transformer_depth=1, context_dim=256)
        self.first_stage = FirstStage()
        self.conditioner = FaceConditioner()
        self.register_buffer("alphas", torch.empty(1000))

    @classmethod
    def from_checkpoints(cls, diffusion_path, identity_path, device):
        state = read_state(diffusion_path)
        identity = read_state(identity_path)
        # Meta construction avoids allocating and initializing a second copy of
        # a 3.5GB checkpoint; every inference tensor must then load strictly.
        with torch.device("meta"):
            model = cls()
        _load_exact(model.first_stage, _subset(state, "first_stage_model."), "DiffSwap VQ")
        conditioning = _subset(state, "cond_stage_model.")
        conditioning.update({"face_model." + key: value for key, value in identity.items()})
        _load_exact(model.conditioner, conditioning, "DiffSwap conditioner")

        # Official inference enters ema_scope. Load those denoiser weights
        # directly to avoid storing a second complete UNet at inference time.
        raw_unet = _subset(state, "model.diffusion_model.")
        if model.unet.state_dict().keys() != raw_unet.keys():
            raise ValueError("DiffSwap checkpoint does not match the released UNet.")
        ema = {}
        for key in raw_unet:
            name = "model_ema.diffusion_model" + key.replace(".", "")
            if name not in state:
                raise ValueError(f"Missing DiffSwap inference EMA weight: {name}")
            ema[key] = state[name]
        _load_exact(model.unet, ema, "DiffSwap EMA UNet")
        alphas = state["alphas_cumprod"]
        if (alphas.shape != (1000,) or not torch.isfinite(alphas).all()
                or not ((alphas > 0) & (alphas <= 1)).all()
                or not (alphas[1:] <= alphas[:-1]).all()):
            raise ValueError("Invalid DiffSwap diffusion schedule.")
        model.alphas = alphas
        model.conditioner.face_model.fp16 = torch.device(device).type == "cuda"
        model.requires_grad_(False).eval().to(device)
        return model

    @torch.no_grad()
    def generate(self, source, target, seed=0, steps=200):
        """200-step DDIM, eta=0, target guidance 0.01, as official inference."""
        if (isinstance(steps, bool) or not isinstance(steps, int)
                or not 2 <= steps < 1000 or 1000 % steps):
            raise ValueError("DDIM steps must divide 1000 and be between 2 and 999.")
        device = self.alphas.device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        source_latent = self.first_stage.encode(source["image"])
        target_latent = self.first_stage.encode(target["image"])
        conditioning = self.conditioner(source["image"], source_latent,
                                         source["organs"], source["theta"],
                                         target["landmarks"])
        mask = F.interpolate(1 - target["mask"][:, None],
                             target_latent.shape[-2:], mode="nearest")
        mask = (mask > 0).to(target_latent)
        # Preserve CompVis/DiffSwap's +1 DDIM offset and first previous alpha.
        timesteps = torch.arange(0, 1000, 1000 // steps, device=device) + 1
        alphas = self.alphas[timesteps]
        previous = torch.cat((self.alphas[:1], alphas[:-1]))
        latent = torch.randn(target_latent.shape, generator=generator, device=device)
        for index in range(steps - 1, -1, -1):
            step = timesteps[index]
            alpha, alpha_prev = alphas[index], previous[index]
            noise = torch.randn(target_latent.shape, generator=generator, device=device)
            original_noisy = alpha.sqrt() * target_latent + (1 - alpha).sqrt() * noise
            latent = original_noisy * mask + (1 - mask) * latent
            prediction = self.unet(latent, step[None], context=conditioning)
            estimate = (latent - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
            prediction = prediction - 0.01 * alpha.sqrt() * (estimate - target_latent)
            latent = alpha_prev.sqrt() * estimate + (1 - alpha_prev).sqrt() * prediction
            # Official eta=0 sampler still draws zero-scaled noise. Advance the
            # local generator identically for reproducible downstream q noise.
            torch.randn(latent.shape, generator=generator, device=device)
        return ((self.first_stage.decode(latent) + 1) / 2).clamp(0, 1)
