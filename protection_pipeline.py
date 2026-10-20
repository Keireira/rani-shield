"""Sequential PhantomSeal -> FaceShield with a clean comparison baseline."""
from __future__ import annotations
import gc
import importlib.util
import json
import math
import threading
import time
from pathlib import Path
from asset_paths import weight_path
import numpy as np
from PIL import Image, ImageOps

MODES = ("phantomseal_faceshield", "phantomseal", "faceshield")


def readiness(root_dir):
    root = Path(root_dir).resolve()
    from phantomseal_assets import ensemble_readiness
    phantom = ensemble_readiness(root)
    if importlib.util.find_spec("facenet_pytorch") is None:
        phantom["ready"] = False
        phantom["message"] = "Missing: facenet-pytorch"
    required_fs = ["faceshield/arcface50_checkpoint.tar", "faceshield/arcface100_checkpoint.tar",
        "ip-adapter/ip-adapter_sd15.bin", "sd15/unet/config.json",
        "sd15/unet/diffusion_pytorch_model.fp16.safetensors",
        "sd15/vae/config.json", "sd15/vae/diffusion_pytorch_model.fp16.safetensors",
        "sd15/text_encoder/config.json", "sd15/text_encoder/model.fp16.safetensors",
        "sd15/tokenizer/vocab.json", "sd15/tokenizer/merges.txt",
        "ip-adapter/models/image_encoder/config.json", "ip-adapter/models/image_encoder/model.safetensors"]
    fs_missing = [str(weight_path(root, name).relative_to(root)) for name in required_fs if not weight_path(root, name).is_file()]
    return {"phantomseal": phantom,
            "faceshield": {"ready": not fs_missing, "message": "Ready" if not fs_missing else "Missing: " + ", ".join(fs_missing)}}


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ProtectionPipeline:
    def __init__(self, root_dir, phantom_factory=None, shield_factory=None):
        self.root = Path(root_dir)
        self._phantom_factory = phantom_factory
        self._shield_factory = shield_factory
        self._lock = threading.RLock()
        self._loaded = False

    def protect(self, input_path, output_dir, max_size=1280, iterations=30, noise_clamp=12,
                step_size=1, analysis_size=512, progress=None, mode="phantomseal_faceshield", phantom_iterations=1000):
        if mode not in MODES:
            raise ValueError("Unknown protection mode.")
        if not isinstance(max_size, int) or not 128 <= max_size <= 1280:
            raise ValueError("Maximum long edge must be between 128 and 1280.")
        if not isinstance(iterations, int) or not 1 <= iterations <= 200:
            raise ValueError("FaceShield steps must be between 1 and 200.")
        if not isinstance(phantom_iterations, int) or not 10 <= phantom_iterations <= 2000:
            raise ValueError("PhantomSeal steps must be between 10 and 2000.")
        if not math.isfinite(noise_clamp) or not 1 <= noise_clamp <= 32 or not math.isfinite(step_size) or not .25 <= step_size <= 4:
            raise ValueError("Invalid FaceShield noise limit or step size.")
        with self._lock:
            if not self._phantom_factory and mode != "faceshield" and not readiness(self.root)["phantomseal"]["ready"]:
                raise RuntimeError("PhantomSeal ensemble is not ready. " + readiness(self.root)["phantomseal"]["message"])
            started = time.perf_counter()
            folder = Path(output_dir).resolve()
            folder.mkdir(parents=True, exist_ok=True)
            with Image.open(input_path) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
                factor = min(1., max_size / max(original.size))
                size = (max(1, round(original.width * factor)), max(1, round(original.height * factor)))
                if original.size != size:
                    original = original.resize(size, Image.Resampling.LANCZOS)
            clean_path = folder / "original.png"
            original.save(clean_path)
            next_input = clean_path
            stages = []
            stage1_path = None
            def notify(value, message):
                if progress:
                    progress(value, message)
            self._loaded = True
            try:
                if mode != "faceshield":
                    from phantomseal_runtime import PhantomSealProtector
                    engine = self._phantom_factory() if self._phantom_factory else PhantomSealProtector(self.root)
                    end = .60 if mode == "phantomseal_faceshield" else 1.
                    try:
                        report = engine.protect(next_input, folder / "stage1", iterations=phantom_iterations,
                                                progress=lambda v, m: notify(float(v) * end, str(m)))
                        next_input = Path(report["output_path"])
                        with Image.open(next_input) as stage1:
                            if stage1.size != original.size:
                                raise RuntimeError("PhantomSeal changed the normalized output dimensions.")
                        stage1_path = str(next_input)
                        stages.append({"name": "PhantomSeal", **report})
                    finally:
                        engine.close()
                        del engine
                        release_gpu()
                    notify(end, "Releasing PhantomSeal GPU memory")
                if mode != "phantomseal":
                    from protector import FaceShieldProtector
                    engine = self._shield_factory() if self._shield_factory else FaceShieldProtector(root_dir=self.root)
                    start = .60 if mode == "phantomseal_faceshield" else 0.
                    try:
                        report = engine.protect(next_input, folder / "stage2", max_size=max_size,
                                                iterations=iterations, noise_clamp=noise_clamp, step_size=step_size,
                                                analysis_size=analysis_size,
                                                progress=lambda v, m: notify(start + (1-start) * float(v), "FaceShield: " + str(m)))
                        next_input = Path(report["output_path"])
                        stages.append({"name": "FaceShield", **report})
                    finally:
                        if hasattr(engine, "close"):
                            engine.close()
                        del engine
                        release_gpu()
                with Image.open(next_input) as opened:
                    final = opened.convert("RGB")
                    if final.size != original.size:
                        raise RuntimeError("Protection stages changed the normalized output dimensions.")
                output = folder / "protected.png"
                final.save(output)
                difference = np.abs(np.asarray(final).astype(np.int16) - np.asarray(original).astype(np.int16))
                result = {"source_path": str(clean_path), "output_path": str(output), "stage1_path": stage1_path,
                          "method": mode, "width": size[0], "height": size[1], "stages": stages,
                          "elapsed_seconds": time.perf_counter() - started,
                          "peak_vram_gb": max((s.get("peak_vram_gb", 0.) for s in stages), default=0.),
                          "max_pixel_change": int(difference.max()),
                          "max_pixel_change_rgb": [int(v) for v in difference.max(axis=(0, 1))],
                          "faceshield_noise_limit": noise_clamp if mode != "phantomseal" else None,
                          "note": "Sequential experimental defense; combined protection requires independent face-swap evaluation."}
                (folder / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                notify(1., "Protected PNG saved")
                return result
            finally:
                self._loaded = False
                release_gpu()
