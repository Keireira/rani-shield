"""Reproducible paired SimSwap comparison: clean, PhantomSeal, PhantomSeal+FaceShield.

Defaults: six public source portraits x three fixed targets x PNG/JPEG95.
Protection is computed once per source; cached outputs are reused only when
source, parameters, code, model/bank hashes and output hashes still match.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from asset_paths import MTCNN_FILES, weight_path

from PIL import Image

from benchmark_chain import atomic_json, benchmark, load_rgb, sha256_file, swap_models
from image_io import condition_image


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def baseline_cutoff(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("minimum_baseline must be a finite number between 0 and 1.")
    return float(value)


def read_manifest(path, root):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected a schema_version=1 manifest object.")
    for key in ("sources", "targets"):
        entries = manifest.get(key)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Manifest requires a non-empty {key} array.")
        seen = set()
        for entry in entries:
            name = entry.get("id", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
                raise ValueError(f"Invalid or duplicate {key} id: {name!r}")
            seen.add(name)
            entry["path"] = str((root / entry["path"]).resolve())
            if not Path(entry["path"]).is_file():
                raise ValueError(f"Missing {key} image: {entry['path']}")
            with Image.open(entry["path"]) as image:
                if key == "targets" and image.size != (224, 224):
                    raise ValueError("SimSwap targets must be prepared 224x224 crops.")
    conditions = manifest.get("conditions", ["png"])
    if not conditions or len(conditions) != len(set(conditions)) or not set(conditions) <= {"png", "jpeg95"}:
        raise ValueError("Conditions must be unique png and/or jpeg95.")
    manifest["conditions"] = conditions
    allowed = {"max_size", "phantom_iterations", "iterations", "noise_clamp", "step_size"}
    if not isinstance(manifest.get("params"), dict) or set(manifest["params"]) - allowed:
        raise ValueError("Unknown protection parameters.")
    return manifest


def check_editor_idle(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3) as response:
            state = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError):
        return
    if state.get("active_job"):
        raise RuntimeError("The editor is processing a photo. Finish that job before benchmarking.")


def environment_fingerprint(root):
    root = Path(root).resolve()
    from phantomseal_assets import configuration, bank_images
    config = configuration(root)
    checkpoint = config["checkpoint_dir"]
    weights = [weight_path(root, "simswap", "latest_net_G.pth"),
               checkpoint / "simswap/arcface_checkpoint.tar",
               weight_path(root, "facenet", "facenet_vggface2.pt"),
               checkpoint / "faceshifter/model_ir_se50.pth",
               checkpoint / "hififace/hififace_opensource_299999.ckpt",
               weight_path(root, "ip-adapter", "ip-adapter_sd15.bin")]
    weights += [weight_path(root, "mtcnn", name) for name in MTCNN_FILES]
    weights += list(weight_path(root, "faceshield").glob("arcface*_checkpoint.tar"))
    weights += [p for directory in (weight_path(root, "sd15"), weight_path(root, "ip-adapter", "models", "image_encoder"))
                for p in directory.rglob("*") if p.is_file()]
    code = [root / name for name in ("protection_pipeline.py", "protector.py", "phantomseal_runtime.py",
                                     "phantomseal_assets.py", "ensemble_encoders.py", "phantomseal_config.json", "asset_paths.py",
                                     "image_io.py", "runtime_contracts.py", "scripts/runtime_environment.py",
                                     "scripts/runtime_environment.json")]
    code += list((root / "utils").rglob("*.py")) + list((root / "third_party").rglob("*.py"))
    code += list((root / "models").glob("*.py"))
    files = sorted(set(weights + code + bank_images(config["cloak_dir"])))
    hashes = {str(p): sha256_file(p) for p in files}
    versions = {package: importlib.metadata.version(package) for package in
                ("torch", "torchvision", "facenet-pytorch", "diffusers", "transformers", "numpy", "Pillow")}
    return {"files": hashes, "versions": versions}


def reusable_preparation(folder, fingerprint):
    try:
        prepared = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
        if prepared["fingerprint"] != fingerprint:
            return None
        for name, digest in prepared["outputs"].items():
            path = folder / name
            if path.resolve().parent not in (folder.resolve(), (folder / "stage1").resolve()):
                return None
            if not path.is_file() or sha256_file(path) != digest:
                return None
        if set(prepared["outputs"]) != {"original.png", "stage1/phantomseal.png", "protected.png"}:
            return None
        return prepared
    except (OSError, ValueError, KeyError, TypeError):
        return None


def preflight_faces(manifest, out, root):
    """Reject inputs the application cannot process before a long optimization."""
    from phantomseal_runtime import load_mtcnn
    import numpy as np
    detector = load_mtcnn(root, keep_all=True, device="cpu")
    findings = []
    for role in ("sources", "targets"):
        for entry in manifest[role]:
            boxes, probabilities = detector.detect(load_rgb(Path(entry["path"])))
            valid = [] if boxes is None else [float(p) for p in probabilities if p is not None and np.isfinite(p) and p >= .9]
            findings.append({"role": role, "id": entry["id"], "path": entry["path"],
                             "sha256": sha256_file(Path(entry["path"])), "detected": bool(valid),
                             "best_confidence": max(valid) if valid else None})
    atomic_json(out / "input_preflight.json", {"inputs": findings, "minimum_confidence": .9})
    invalid = [item["id"] for item in findings if not item["detected"]]
    if invalid:
        raise ValueError("No sufficiently clear face in: " + ", ".join(invalid) + ". See input_preflight.json; no swap efficacy is inferred from these input failures.")


def run(args):
    # Reject invalid gates before input/model checks or importing the GPU runtime.
    minimum_baseline = baseline_cutoff(getattr(args, "minimum_baseline", .80))
    root = args.root.resolve()
    from scripts.runtime_environment import configure_environment
    configure_environment(root)
    manifest = read_manifest(args.manifest.resolve(), root)
    if args.max_sources is not None:
        if args.max_sources < 1:
            raise ValueError("--max-sources must be positive.")
        manifest["sources"] = manifest["sources"][:args.max_sources]
    check_editor_idle(args.port)
    from protection_pipeline import ProtectionPipeline, readiness
    ready = readiness(root)
    if not all(entry["ready"] for entry in ready.values()):
        raise RuntimeError("Models are not ready: " + json.dumps(ready))
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required; --allow-cpu explicitly permits a very slow CPU run.")
    started = time.perf_counter()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("Hashing installed models, code and full cloak bank for reproducible caching...", flush=True)
    environment = environment_fingerprint(root)
    atomic_json(out / "environment.json", environment)
    atomic_json(out / "manifest.json", manifest)
    preflight_faces(manifest, out, root)
    cases = []
    for index, source in enumerate(manifest["sources"], 1):
        print(f"[{index}/{len(manifest['sources'])}] Preparing {source['id']}", flush=True)
        folder = out / "cases" / source["id"]
        signature = canonical_hash({"source": sha256_file(Path(source["path"])),
                                    "params": manifest["params"], "environment": environment})
        cached = reusable_preparation(folder, signature) if args.resume or args.evaluate_only else None
        if cached is None:
            if args.evaluate_only:
                raise ValueError(f"No valid cached preparation for {source['id']}. Run without --evaluate-only.")
            check_editor_idle(args.port)
            torch.manual_seed(0)
            last_ten = [-1]
            def progress(value, message):
                percent = int(float(value) * 10)
                if percent != last_ten[0]:
                    last_ten[0] = percent
                    print(f"  {min(percent*10,100):3d}% {message}", flush=True)
            report = ProtectionPipeline(root).protect(Path(source["path"]), folder,
                        mode="phantomseal_faceshield", progress=progress, **manifest["params"])
            outputs = {name: sha256_file(folder / name) for name in
                       ("original.png", "stage1/phantomseal.png", "protected.png")}
            atomic_json(folder / "prepared.json", {"fingerprint": signature, "outputs": outputs,
                        "source": source, "params": manifest["params"], "elapsed_seconds": report["elapsed_seconds"]})
        else:
            print("  Reusing verified protection PNGs", flush=True)
        cases.append((source, folder))
    if args.prepare_only:
        print(f"Prepared cases: {out / 'cases'}", flush=True)
        return
    reports = []
    image_quality = []
    import numpy as np
    from gui import compare_arrays
    for source, folder in cases:
        with Image.open(folder / "original.png") as image:
            original = np.asarray(image.convert("RGBA"))
        quality = {"source_id": source["id"]}
        for name, relative in (("stage1", "stage1/phantomseal.png"), ("final", "protected.png")):
            with Image.open(folder / relative) as image:
                pixels = np.asarray(image.convert("RGBA"))
            metrics = compare_arrays(original, pixels)
            metrics["max_pixel_change"] = int(np.abs(original[..., :3].astype(np.int16) - pixels[..., :3].astype(np.int16)).max())
            quality[name] = metrics
        image_quality.append(quality)
    pairs_total = len(cases) * len(manifest["targets"]) * len(manifest["conditions"])
    # Keep one evaluator loaded, after all protection models have been released.
    with swap_models(root) as models:
        for source, folder in cases:
            for condition in manifest["conditions"]:
                transformed = out / "conditions" / source["id"] / condition
                transformed.mkdir(parents=True, exist_ok=True)
                variants = {"clean": folder / "original.png", "stage1": folder / "stage1/phantomseal.png",
                            "final": folder / "protected.png"}
                for name, path in list(variants.items()):
                    image = condition_image(load_rgb(path), condition)
                    destination = transformed / (name + ".png")
                    image.save(destination)
                    variants[name] = destination
                for target in manifest["targets"]:
                    index = len(reports) + 1
                    print(f"Swap [{index}/{pairs_total}] {source['id']} -> {target['id']} / {condition}", flush=True)
                    pair_out = out / "pairs" / f"{source['id']}--{target['id']}--{condition}.json"
                    report = benchmark(root, variants["clean"], variants["stage1"], variants["final"],
                                       Path(target["path"]), pair_out, models=models,
                                       reference_path=folder / "original.png", minimum_baseline=minimum_baseline)
                    report.update(source_id=source["id"], target_id=target["id"], condition=condition)
                    atomic_json(pair_out, report)
                    reports.append(report)
                    atomic_json(out / "progress.json", {"complete": len(reports), "total": pairs_total})
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from benchmark_report import write_report
    generated = write_report(out, reports, {"manifest": manifest, "environment_file": "environment.json",
                              "protected_role": "source_identity",
                              "reuses_exploratory_source": True, "exploratory_source_ids": ["portrait"],
                              "heldout_note": "portrait/cage was previously used as an exploratory single-pair check. Protection parameters were frozen before this suite; the other five source portraits were not used in that check. One proposed input (lcw.jpg) was rejected because the application's detector found no face, before any swap scoring; gy.jpg passed the input preflight and replaced it.",
                              "input_exclusions": manifest.get("input_exclusions", []),
                              "image_quality": image_quality,
                              "elapsed_seconds": time.perf_counter() - started,
                              "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"},
                              root=root, minimum_baseline=minimum_baseline)
    print(json.dumps(generated, indent=2), flush=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=root / "benchmark_manifest.json")
    parser.add_argument("--out", type=Path, default=root / "results/benchmarks/latest")
    parser.add_argument("--resume", action="store_true", help="Reuse protection PNGs only when all hashes still match")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--port", type=int, default=7876, help="Refuse to benchmark during an active editor job")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--minimum-baseline", type=float, default=.80,
                        help="Clean-swap baseline cosine quality gate, 0..1 (default: 0.80). Heuristic only; manual baseline inspection is required.")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true")
    phase.add_argument("--evaluate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
