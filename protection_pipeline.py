"""Ordered protection stages sharing one normalized clean comparison baseline."""
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
from runtime_contracts import analysis_size as validate_analysis_size, protection_parameters


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

    def _engine(self, kind):
        if kind == "phantomseal":
            if self._phantom_factory:
                return self._phantom_factory()
            from phantomseal_runtime import PhantomSealProtector
            return PhantomSealProtector(self.root)
        if kind == "nullswap":
            if self._nullswap_factory:
                return self._nullswap_factory()
            from nullswap_runtime import NullSwapProtector
            return NullSwapProtector(self.root)
        if self._shield_factory:
            return self._shield_factory()
        from protector import FaceShieldProtector
        return FaceShieldProtector(root_dir=self.root)

    def protect(self, input_path, output_dir, max_size=1280, iterations=30, noise_clamp=12,
                step_size=1, analysis_size=512, progress=None, mode="phantomseal_faceshield", phantom_iterations=1000,
                nullswap_strength=1.0, nullswap_seed=0, stages=None):
        raw = {
            "mode": mode, "max_size": max_size, "iterations": iterations,
            "phantom_iterations": phantom_iterations, "noise_clamp": noise_clamp, "step_size": step_size,
            "nullswap_strength": nullswap_strength, "nullswap_seed": nullswap_seed,
        }
        if stages is not None:
            raw["stages"] = stages
        params = protection_parameters(raw)
        max_size = params["max_size"]
        analysis_size = validate_analysis_size(analysis_size)
        legacy_stages = {
            "nullswap": [{"type": "nullswap", "params": {
                "strength": params["nullswap_strength"], "seed": params["nullswap_seed"]}}],
            "phantomseal": [{"type": "phantomseal", "params": {"iterations": params["phantom_iterations"]}}],
            "faceshield": [{"type": "faceshield", "params": {
                "iterations": params["iterations"], "noise_clamp": params["noise_clamp"],
                "step_size": params["step_size"], "analysis_size": analysis_size}}],
        }
        legacy_stages["phantomseal_faceshield"] = legacy_stages["phantomseal"] + legacy_stages["faceshield"]
        recipe = params["stages"] if mode == "custom" else legacy_stages[mode]
        names = {"nullswap": "NullSwap (PhantomSeal reconstruction)",
                 "phantomseal": "PhantomSeal", "faceshield": "FaceShield"}
        with self._lock:
            if not self._phantom_factory and any(stage["type"] == "phantomseal" for stage in recipe):
                available = readiness(self.root)["phantomseal"]
                if not available["ready"]:
                    raise RuntimeError("PhantomSeal ensemble is not ready. " + available["message"])
            started = time.perf_counter()
            folder = Path(output_dir).resolve()
            original = fit_rgb(input_path, max_size)
            folder.mkdir(parents=True, exist_ok=True)
            size = original.size
            clean_path = folder / "original.png"
            original.save(clean_path)
            next_input = clean_path
            reports = []
            stage1_path = None
            last_progress = 0.

            def notify(value, message):
                nonlocal last_progress
                last_progress = max(last_progress, min(1., max(0., float(value))))
                if progress:
                    progress(last_progress, message)

            self._loaded = True
            try:
                for index, stage in enumerate(recipe, start=1):
                    kind, stage_params = stage["type"], stage["params"]
                    name = names[kind]
                    label = f"Stage {index}/{len(recipe)}: {name}"
                    start, end = (index - 1) / len(recipe), index / len(recipe)
                    if mode == "phantomseal_faceshield":
                        start, end = ((0., .60) if index == 1 else (.60, 1.))
                    notify(start, f"{label}: loading")
                    relative = (f"stage{index:02d}_{kind}" if mode == "custom" else
                                {"nullswap": "nullswap", "phantomseal": "stage1", "faceshield": "stage2"}[kind])
                    engine = self._engine(kind)
                    try:
                        arguments = dict(stage_params)
                        if kind in ("nullswap", "faceshield"):
                            arguments["max_size"] = max_size
                        report = engine.protect(
                            next_input, folder / relative, **arguments,
                            progress=lambda value, message: notify(
                                start + (end - start) * min(1., max(0., float(value))),
                                f"{label}: {message}"))
                        next_input = Path(report["output_path"])
                        with Image.open(next_input) as output_image:
                            if output_image.size != size:
                                raise RuntimeError(f"{label} changed the normalized output dimensions.")
                        reports.append({**report, "type": kind, "index": index, "params": dict(stage_params),
                                        "name": name, "output_path": str(next_input)})
                        if kind == "phantomseal" and mode != "custom":
                            stage1_path = str(next_input)
                    finally:
                        try:
                            if hasattr(engine, "close"):
                                engine.close()
                        finally:
                            del engine
                            release_gpu()
                    notify(end, f"{label}: complete")
                with Image.open(next_input) as opened:
                    final = opened.convert("RGB")
                output = folder / "protected.png"
                final.save(output)
                difference = np.abs(np.asarray(final).astype(np.int16) - np.asarray(original).astype(np.int16))
                result = {"source_path": str(clean_path), "output_path": str(output), "stage1_path": stage1_path,
                          "method": mode, "width": size[0], "height": size[1], "stages": reports,
                          "pipeline": recipe, "elapsed_seconds": time.perf_counter() - started,
                          "peak_vram_gb": max((s.get("peak_vram_gb", 0.) for s in reports), default=0.),
                          "max_pixel_change": int(difference.max()),
                          "max_pixel_change_rgb": [int(v) for v in difference.max(axis=(0, 1))],
                          "faceshield_noise_limit": params["noise_clamp"] if mode in ("faceshield", "phantomseal_faceshield") else None,
                          "note": ("NullSwap reconstruction distributed by PhantomSeal; not the original authors' implementation. "
                                   if any(stage["type"] == "nullswap" for stage in recipe) else "") +
                                  "Experimental defense; protection requires independent face-swap evaluation."}
                (folder / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                notify(1., "Protected PNG saved")
                return result
            finally:
                self._loaded = False
                release_gpu()
