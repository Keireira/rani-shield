"""Independent protection modes and a sequential chain, with a clean comparison baseline."""
from __future__ import annotations
import gc
import importlib.util
import json
import threading
import time
from pathlib import Path
from asset_paths import weight_path
import numpy as np
from PIL import Image
from image_io import fit_rgb
from runtime_contracts import MODES, analysis_size as validate_analysis_size, protection_parameters


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
    from nullswap_runtime import readiness as nullswap_readiness
    return {"phantomseal": phantom, "nullswap": nullswap_readiness(root),
            "faceshield": {"ready": not fs_missing, "message": "Ready" if not fs_missing else "Missing: " + ", ".join(fs_missing)}}


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ProtectionPipeline:
    def __init__(self, root_dir, phantom_factory=None, shield_factory=None, nullswap_factory=None):
        self.root = Path(root_dir).resolve()
        self._phantom_factory = phantom_factory
        self._shield_factory = shield_factory
        self._nullswap_factory = nullswap_factory
        self._lock = threading.RLock()
        self._loaded = False

    def protect(self, input_path, output_dir, max_size=1280, iterations=30, noise_clamp=12,
                step_size=1, analysis_size=512, progress=None, mode="phantomseal_faceshield", phantom_iterations=1000,
                nullswap_strength=1.0, nullswap_seed=0):
        params = protection_parameters({
            "mode": mode, "max_size": max_size, "iterations": iterations,
            "phantom_iterations": phantom_iterations, "noise_clamp": noise_clamp, "step_size": step_size,
            "nullswap_strength": nullswap_strength, "nullswap_seed": nullswap_seed,
        })
        max_size, iterations, phantom_iterations = (params[name] for name in ("max_size", "iterations", "phantom_iterations"))
        noise_clamp, step_size = params["noise_clamp"], params["step_size"]
        analysis_size = validate_analysis_size(analysis_size)
        with self._lock:
            if not self._phantom_factory and mode in ("phantomseal", "phantomseal_faceshield"):
                available = readiness(self.root)["phantomseal"]
                if not available["ready"]:
                    raise RuntimeError("PhantomSeal ensemble is not ready. " + available["message"])
            started = time.perf_counter()
            folder = Path(output_dir).resolve()
            folder.mkdir(parents=True, exist_ok=True)
            original = fit_rgb(input_path, max_size)
            size = original.size
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
                if mode == "nullswap":
                    from nullswap_runtime import NullSwapProtector
                    engine = self._nullswap_factory() if self._nullswap_factory else NullSwapProtector(self.root)
                    try:
                        report = engine.protect(next_input, folder / "nullswap", max_size=max_size, progress=notify,
                                                strength=params["nullswap_strength"], seed=params["nullswap_seed"])
                        next_input = Path(report["output_path"])
                        stages.append({"name": "NullSwap (PhantomSeal reconstruction)", **report})
                    finally:
                        engine.close()
                        del engine
                        release_gpu()
                if mode in ("phantomseal", "phantomseal_faceshield"):
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
                if mode in ("faceshield", "phantomseal_faceshield"):
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
                          "faceshield_noise_limit": noise_clamp if mode in ("faceshield", "phantomseal_faceshield") else None,
                          "note": ("NullSwap reconstruction distributed by PhantomSeal; not the original authors' implementation."
                                   if mode == "nullswap" else
                                   "Experimental defense; protection requires independent face-swap evaluation.")}
                (folder / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                notify(1., "Protected PNG saved")
                return result
            finally:
                self._loaded = False
                release_gpu()
