"""Single-image FaceShield runtime, adapted for a 16 GB CUDA device.

The perturbation lives at the output resolution (up to 1280 pixels per side).
The diffusion objective sees a differentiable, letterboxed analysis image,
normally 512 x 512. This is a multi-scale approximation of the upstream
512-pixel method; it is not native 1280-pixel UNet inference or a certified
defence. The upstream launcher's disabled MTCNN loss remains disabled by
default. No training is performed.
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
from asset_paths import weight_path
from functools import partial
import sys
import threading
import time
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F


Progress = Callable[[float, str], None]
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def fit_dimensions(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Fit inside a square while preserving aspect ratio; never enlarge."""
    if min(width, height, max_size) <= 0:
        raise ValueError("Image dimensions and max_size must be positive.")
    ratio = min(1.0, max_size / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def analysis_view(image: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiable aspect-preserving resize and symmetric neutral padding."""
    height, width = image.shape[-2:]
    ratio = size / max(height, width)
    target_h, target_w = max(1, round(height * ratio)), max(1, round(width * ratio))
    resized = F.interpolate(image, (target_h, target_w), mode="bilinear",
                            align_corners=False, antialias=True)
    pad_h, pad_w = size - target_h, size - target_w
    return F.pad(resized, (pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2), value=0.5)


def smooth_step(step: torch.Tensor) -> torch.Tensor:
    """Upstream Sobel-selected 9 x 9 Gaussian blur, with finite zero handling."""
    span = step.max() - step.min()
    normalized = (step - step.min()) / span.clamp_min(torch.finfo(step.dtype).eps)
    sobel = step.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_x = sobel[None, None].repeat(3, 1, 1, 1)
    kernel_y = sobel.T[None, None].repeat(3, 1, 1, 1)
    gx = F.conv2d(normalized, kernel_x, padding=1, groups=3).abs()
    gy = F.conv2d(normalized, kernel_y, padding=1, groups=3).abs()
    mask = ((gx == gx.max()) | (gy == gy.max())).to(step.dtype)
    mask = F.conv2d(mask, step.new_ones((3, 1, 9, 9)), padding=4, groups=3).clamp(0, 1)
    axis = torch.arange(9, device=step.device, dtype=step.dtype) - 4
    gaussian = torch.exp(-0.5 * (axis / 5).square())
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)[None, None].repeat(3, 1, 1, 1)
    blurred = F.conv2d(step, kernel, padding=4, groups=3)
    return blurred * mask + step * (1 - mask)


class FaceShieldProtector:
    """Load frozen public backbones once and optimise each input image."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        *,
        root_dir: str | Path | None = None,
        image_encoder_path: str | Path = "h94/IP-Adapter",
        pretrained_ip_adapter_path: str | Path | None = None,
        pretrained_arcface50_path: str | Path | None = None,
        pretrained_arcface100_path: str | Path | None = None,
        device: str | None = None,
        local_files_only: bool = False,
        enable_mtcnn: bool = False,
        seed: int = 7730,
    ):
        self.root_dir = Path(root_dir or Path(__file__).resolve().parent).resolve()
        self.cache_dir = Path(cache_dir or weight_path(self.root_dir, "cache", "huggingface")).resolve()
        local_sd = weight_path(self.root_dir, "sd15")
        local_clip = weight_path(self.root_dir, "ip-adapter")
        self.model_path = str(local_sd if str(model_path) == DEFAULT_MODEL and
                              (local_sd / "unet" / "config.json").is_file() else model_path)
        self.image_encoder_path = str(local_clip if str(image_encoder_path) == "h94/IP-Adapter" and
                                      (local_clip / "models" / "image_encoder" / "config.json").is_file()
                                      else image_encoder_path)
        self.ip_adapter_path = Path(pretrained_ip_adapter_path or weight_path(self.root_dir, "ip-adapter", "ip-adapter_sd15.bin"))
        self.arcface50_path = Path(pretrained_arcface50_path or weight_path(self.root_dir, "faceshield", "arcface50_checkpoint.tar"))
        self.arcface100_path = Path(pretrained_arcface100_path or weight_path(self.root_dir, "faceshield", "arcface100_checkpoint.tar"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or choose device='cpu'.")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.local_files_only = local_files_only
        self.enable_mtcnn = bool(enable_mtcnn)
        self.seed = seed
        self._loaded = False
        self._lock = threading.RLock()
        self._dct_cache = None

    @staticmethod
    def _notify(progress: Progress | None, fraction: float, message: str) -> None:
        if progress is not None:
            progress(float(fraction), message)

    def _autocast(self):
        return (torch.autocast("cuda", dtype=self.dtype) if self.device.type == "cuda"
                else nullcontext())

    def _common(self) -> dict:
        return {"cache_dir": str(self.cache_dir), "local_files_only": self.local_files_only}

    @staticmethod
    def _variant(repo: str, subfolder: str, stem: str) -> dict:
        # Remote SD1.5 has fp16 variants. Local downloads may use either name.
        folder = Path(repo) / subfolder
        if not Path(repo).is_dir() or (folder / f"{stem}.fp16.safetensors").is_file():
            return {"variant": "fp16"}
        return {}

    def _weight_file(self, repo: str, subfolder: str) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
        names = ("diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors",
                 "diffusion_pytorch_model.fp16.bin", "diffusion_pytorch_model.bin")
        if Path(repo).is_dir():
            for name in names:
                path = Path(repo) / subfolder / name
                if path.is_file():
                    return path
            raise FileNotFoundError(f"No UNet weights in {Path(repo) / subfolder}")
        for name in names:
            try:
                return Path(hf_hub_download(repo, filename=f"{subfolder}/{name}", **self._common()))
            except (EntryNotFoundError, LocalEntryNotFoundError):
                continue
        raise FileNotFoundError(f"No compatible UNet weights found in {repo}")

    @staticmethod
    def _load_arcface(path: Path) -> torch.nn.Module:
        if not path.is_file():
            raise FileNotFoundError(f"Missing ArcFace checkpoint: {path}. Download the author's public models.zip first.")
        # These files contain modules. Only allow the exact classes found by static
        # inspection; the three repository classes have been audited for pickle hooks.
        # Never fall back to unrestricted pickle loading for downloaded checkpoints.
        from models.arcface_models import IRBlock, SEBlock, ResNet
        allowed_classes = [
            torch.nn.BatchNorm1d, torch.nn.AdaptiveAvgPool2d, torch.nn.PReLU,
            torch.nn.MaxPool2d, torch.nn.Conv2d, torch.nn.Dropout, torch.nn.BatchNorm2d,
            IRBlock, SEBlock, torch.nn.Linear, ResNet, torch.nn.Sigmoid, torch.nn.Sequential,
        ]
        allowed_names = {f"{item.__module__}.{item.__qualname__}" for item in allowed_classes}
        unsafe_names = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        unexpected_names = unsafe_names - allowed_names
        if unexpected_names:
            raise ValueError(f"ArcFace checkpoint has unaudited classes: {sorted(unexpected_names)}")
        with torch.serialization.safe_globals(allowed_classes):
            model = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(model, dict):
            model = model.get("model", model.get("backbone", model))
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected the author's complete ArcFace model in {path}")
        return model

    def load_models(self, progress: Progress | None = None) -> "FaceShieldProtector":
        with self._lock:
            if self._loaded:
                self._notify(progress, 0.20, "Models are already loaded")
                return self
            # Keep imports lazy so UI startup, input validation and model-free tests need no downloads.
            root = str(self.root_dir)
            if root not in sys.path:
                sys.path.insert(0, root)
            from diffusers import AutoencoderKL
            from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
            from safetensors.torch import load_file
            from utils.unet.unet_attack import AttackUnet_IP_all, AttackCLIP
            from utils.unet.ip_adapter.ip_adapter import ImageProjModel
            from utils.unet.ip_adapter.attention_processor import IPAttnProcessor2_0, AttnProcessor2_0
            from utils.landmark.arcface_attack import AttackArcFace
            from utils.dct import make_dct_basis, dct_pass_filter

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._notify(progress, 0.02, "Loading Stable Diffusion tokenizer and text encoder")
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer", **self._common())
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_path, subfolder="text_encoder", torch_dtype=self.dtype,
                **self._variant(self.model_path, "text_encoder", "model"), **self._common())
            text_encoder.eval().requires_grad_(False).to(self.device)
            inputs = self.tokenizer([""], max_length=self.tokenizer.model_max_length,
                                    padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                self.encoder_hidden_states = text_encoder(inputs.input_ids.to(self.device))[0].detach()
            text_encoder.to("cpu")
            del text_encoder
            self._notify(progress, 0.06, "Loading VAE on CPU")
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path, subfolder="vae", torch_dtype=torch.float32,
                **self._variant(self.model_path, "vae", "diffusion_pytorch_model"), **self._common())
            self.vae.eval().requires_grad_(False)

            self._notify(progress, 0.09, "Loading diffusion UNet")
            config_path = self.root_dir / "utils/unet/unet_config15.json"
            config = {key: value for key, value in json.loads(config_path.read_text(encoding="utf-8")).items()
                      if not key.startswith("_")}
            self.unet = AttackUnet_IP_all(config_file=config).to(dtype=self.dtype)
            weight_path = self._weight_file(self.model_path, "unet")
            state = (load_file(str(weight_path), device="cpu") if weight_path.suffix == ".safetensors"
                     else torch.load(weight_path, map_location="cpu", weights_only=True))
            self.unet.load_state_dict(state, strict=True)
            del state

            self._notify(progress, 0.12, "Loading CLIP image encoder and IP-Adapter")
            clip_subfolder = (None if (Path(self.image_encoder_path) / "config.json").is_file()
                              else "models/image_encoder")
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.image_encoder_path, subfolder=clip_subfolder, torch_dtype=self.dtype, **self._common())
            self.image_preprocess = AttackCLIP()
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=self.unet.config.cross_attention_dim,
                clip_embeddings_dim=self.image_encoder.config.projection_dim, clip_extra_context_tokens=4)
            processors = {}
            channels = self.unet.config.block_out_channels
            for name in self.unet.attn_processors:
                if name.endswith("attn1.processor"):
                    processors[name] = AttnProcessor2_0()
                    continue
                if name.startswith("mid_block"):
                    hidden_size = channels[-1]
                elif name.startswith("down_blocks"):
                    hidden_size = channels[int(name.split(".")[1])]
                else:
                    hidden_size = list(reversed(channels))[int(name.split(".")[1])]
                processors[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size, cross_attention_dim=self.unet.config.cross_attention_dim)
            self.unet.set_attn_processor(processors)
            state = torch.load(self.ip_adapter_path, map_location="cpu", weights_only=True)
            self.image_proj_model.load_state_dict(state["image_proj"], strict=True)
            torch.nn.ModuleList(self.unet.attn_processors.values()).load_state_dict(state["ip_adapter"], strict=True)
            del state

            self._notify(progress, 0.16, "Loading ArcFace 50 and 100")
            self.face_embedder50 = self._load_arcface(self.arcface50_path)
            self.face_embedder100 = self._load_arcface(self.arcface100_path)
            self.id_preprocess = AttackArcFace()
            for model in (self.unet, self.image_encoder, self.image_proj_model,
                          self.face_embedder50, self.face_embedder100):
                model.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
            self._dct_cache = (make_dct_basis(8, self.device), dct_pass_filter(self.device)[0])
            self._mtcnn_attack = None
            if self.enable_mtcnn:
                # Upstream default launcher disables this branch. Do not silently enable it.
                from utils.landmark.mtcnn_attack import mtcnn_attack
                self._mtcnn_attack = partial(mtcnn_attack, weights_path=weight_path(
                    self.root_dir, "mtcnn-attack", "pnet-6b6ef92b.pth"))
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._loaded = True
            self._notify(progress, 0.20, "Frozen models loaded")
            return self

    def _prepare_reference(self, source: torch.Tensor, analysis_size: int) -> dict:
        from utils.utils import AttentionStore
        with torch.no_grad():
            analysis = analysis_view(source / 255, analysis_size)
            # VAE is needed once per input; keep it off the GPU during PGD.
            self.vae.to(self.device)
            try:
                generator = torch.Generator(device=self.device).manual_seed(self.seed)
                latent = self.vae.encode(analysis.float()).latent_dist.sample(generator=generator)
                latent = (latent * self.vae.config.scaling_factor).to(self.dtype).detach()
            finally:
                self.vae.to("cpu")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            with self._autocast():
                ref_50, ref_100 = self.id_preprocess.preprocess(source / 255)
                id50 = self.face_embedder50(ref_50).detach()
                id100 = self.face_embedder100(ref_100).detach()  # Fix upstream ArcFace-100 reference bug.
                encoded = self.image_encoder(self.image_preprocess(analysis)).image_embeds.detach()
                projected = self.image_proj_model(encoded)
                hidden = torch.cat([self.encoder_hidden_states, projected], dim=1)
                attention = AttentionStore()
                timestep = torch.tensor([0], device=self.device)
                self.unet(latent, timestep, hidden, store_controller=attention, unet_threshold=0.2)
            maps = [tuple(item.detach() for item in pair) for pair in attention.attn_map]
            return {"latent": latent, "timestep": timestep, "id50": id50, "id100": id100,
                    "encoded": encoded, "attention": maps}

    def _iteration_gradient(self, source: torch.Tensor, delta: torch.Tensor,
                            reference: dict, analysis_size: int) -> torch.Tensor:
        # Rebuild each branch from the same leaf delta, releasing its graph before the next one.
        # A fixed loss scale keeps image gradients representable through frozen fp16 networks.
        scale = 1024.0 if self.dtype == torch.float16 else 1.0
        gradient = torch.zeros_like(delta)
        for network, key, index in ((self.face_embedder50, "id50", 0),
                                    (self.face_embedder100, "id100", 1)):
            with self._autocast():
                adv = (source + delta).clamp(0, 255) / 255
                arc_input = self.id_preprocess.preprocess(adv)[index]
                embedding = network(arc_input)
                loss = -4 * (1 - F.cosine_similarity(embedding.float(), reference[key].float()).mean())
            gradient.add_(torch.autograd.grad(loss * scale, delta)[0].float() / scale)
            del adv, arc_input, embedding, loss
        if self._mtcnn_attack is not None:
            with self._autocast():
                adv = analysis_view((source + delta).clamp(0, 255) / 255, analysis_size)
                loss = 9 * self._mtcnn_attack(2 * adv - 1, loss_fn=F.mse_loss, device=self.device)
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                gradient.add_(torch.autograd.grad(loss * scale, delta)[0].float() / scale)
            del adv, loss
        with self._autocast():
            adv = analysis_view((source + delta).clamp(0, 255) / 255, analysis_size)
            encoded = self.image_encoder(self.image_preprocess(adv)).image_embeds
            projected = self.image_proj_model(encoded)
            hidden = torch.cat([self.encoder_hidden_states, projected], dim=1)
            clip_loss = F.l1_loss(encoded.float(), reference["encoded"].float())
            attention_loss = self.unet(
                reference["latent"], reference["timestep"], hidden,
                loss_fn=lambda a, b: F.mse_loss(a.float(), b.float()), loss=0,
                gt_attn_map=reference["attention"].copy())
            loss = -clip_loss + attention_loss
        gradient.add_(torch.autograd.grad(loss * scale, delta)[0].float() / scale)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite protection gradient. Try a smaller analysis_size or full precision.")
        return gradient

    def _filter_delta(self, delta: torch.Tensor) -> torch.Tensor:
        from utils.dct import blockfy, encode, decode, deblockfy
        if self._dct_cache is None:
            from utils.dct import make_dct_basis, dct_pass_filter
            self._dct_cache = (make_dct_basis(8, self.device), dct_pass_filter(self.device)[0])
        basis, mask = self._dct_cache
        blocks, pad_size = blockfy(delta.float(), 8)
        return deblockfy(decode(encode(blocks, basis) * mask, basis), pad_size)

    def protect(self, input_path: str | Path, output_dir: str | Path, max_size: int = 1280,
                iterations: int = 30, noise_clamp: int = 12, step_size: float = 1,
                analysis_size: int = 512, progress: Progress | None = None) -> dict:
        if not isinstance(max_size, int) or not 1 <= max_size <= 1280:
            raise ValueError("max_size must be an integer between 1 and 1280.")
        if not isinstance(iterations, int) or not 1 <= iterations <= 200:
            raise ValueError("iterations must be an integer between 1 and 200.")
        if not isinstance(analysis_size, int) or analysis_size < 64 or analysis_size % 64:
            raise ValueError("analysis_size must be a positive multiple of 64 (normally 512).")
        if not math.isfinite(noise_clamp) or not 0 <= noise_clamp <= 255:
            raise ValueError("noise_clamp must be finite and between 0 and 255.")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be a finite positive number.")
        input_path = Path(input_path).resolve()
        with Image.open(input_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            size = fit_dimensions(*image.size, max_size)
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            source_array = np.array(image, dtype=np.uint8, copy=True)
        with self._lock:
            started = time.perf_counter()
            self._notify(progress, 0.0, "Preparing input image")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            self.load_models(progress=progress)
            source = (torch.from_numpy(source_array).permute(2, 0, 1).unsqueeze(0).contiguous()
                      .to(self.device, torch.float32))
            self._notify(progress, 0.22, "Computing reference features")
            reference = self._prepare_reference(source, analysis_size)
            delta = torch.zeros_like(source, requires_grad=True)
            with torch.enable_grad():
                for index in range(iterations):
                    gradient = self._iteration_gradient(source, delta, reference, analysis_size)
                    with torch.no_grad():
                        update = smooth_step(step_size * gradient.sign())
                        delta.copy_(self._filter_delta(delta - update).clamp(-noise_clamp, noise_clamp))
                    del gradient, update
                    self._notify(progress, 0.30 + 0.68 * (index + 1) / iterations,
                                 f"Protecting image: {index + 1}/{iterations}")
            with torch.no_grad():
                protected = (source + delta).clamp(0, 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
            destination = Path(output_dir).resolve()
            destination.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem
            source_path = destination / f"{stem}_source.png"
            output_path = destination / f"{stem}_protected.png"
            Image.fromarray(source_array).save(source_path)
            Image.fromarray(protected).save(output_path)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            result = {"output_path": str(output_path), "source_path": str(source_path),
                      "width": size[0], "height": size[1], "elapsed_seconds": time.perf_counter() - started,
                      "peak_vram_gb": (torch.cuda.max_memory_allocated(self.device) / 1024**3
                                       if self.device.type == "cuda" else 0.0),
                      "method": "multi_scale_approximation", "analysis_size": analysis_size,
                      "iterations": iterations, "noise_clamp": noise_clamp,
                      "mtcnn_enabled": self.enable_mtcnn,
                      "max_pixel_change": int(np.abs(protected.astype(np.int16) - source_array.astype(np.int16)).max())}
            (destination / f"{stem}_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            del source, delta, reference
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._notify(progress, 1.0, "Protected PNG saved")
            return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="Input photo")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--analysis-size", type=int, default=512)
    parser.add_argument("--noise-clamp", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--image-encoder-path", default="h94/IP-Adapter")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    protector = FaceShieldProtector(model_path=args.model_path, image_encoder_path=args.image_encoder_path,
                                    cache_dir=args.cache_dir, device=args.device,
                                    local_files_only=args.local_files_only)
    result = protector.protect(args.input_path, args.output_dir, max_size=args.max_size,
                               iterations=args.iterations, analysis_size=args.analysis_size,
                               noise_clamp=args.noise_clamp, step_size=args.step_size,
                               progress=lambda value, description: print(f"{value:6.1%} {description}", flush=True))
    print(json.dumps(result, indent=2))
