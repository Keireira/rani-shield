"""Exercise installed GPU models and PNG/JPEG95 swaps without starting a server.

Uses only the public repository portrait, enlarged to exercise 1280-pixel output.
Reduced iterations verify execution and storage, not protection effectiveness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asset_paths import project_path
from scripts.runtime_environment import configure_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/verification")
    parser.add_argument("--mode", choices=("phantomseal_faceshield", "nullswap"), default="phantomseal_faceshield")
    parser.add_argument("--backends", nargs="+", choices=("simswap", "diffswap", "infoswap", "uniface"), default=["simswap"])
    args = parser.parse_args()
    configure_environment(ROOT)
    folder = project_path(ROOT, args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    outside_reads = set()
    weight_reads = set()
    network_attempts = []

    def audit(event, values):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            network_attempts.append(event)
            raise RuntimeError("The installed-model smoke check must run offline.")
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        raw = os.fsdecode(values[0])
        if raw in {os.devnull, "NUL", "<stdin>", "<stdout>", "<stderr>"}:
            return
        path = Path(raw).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            outside_reads.add(str(path))

    sys.addaudithook(audit)
    started = time.perf_counter()
    from PIL import Image
    import torch
    from protection_pipeline import ProtectionPipeline
    from protection_test import run_protection_test
    from protector import FaceShieldProtector
    from scripts.manage_checks import runtime_paths

    # Record actual tensor-loader inputs, not suffixes: Python's site .pth files
    # configure imports and are not checkpoints. safe_open may read through Rust
    # without a Python "open" event, so instrument that public loader explicitly.
    import safetensors
    import safetensors.torch

    def observe_loader(loader):
        def observed(*args, **kwargs):
            filename = args[0] if args else kwargs.get("f", kwargs.get("filename"))
            value = filename if isinstance(filename, (str, bytes, os.PathLike)) else getattr(filename, "name", None)
            if isinstance(value, (str, bytes, os.PathLike)):
                path = project_path(ROOT, os.fsdecode(value))
                weight_reads.add(path.relative_to(ROOT).as_posix())
            return loader(*args, **kwargs)
        return observed

    torch.load = observe_loader(torch.load)
    safetensors.safe_open = observe_loader(safetensors.safe_open)
    safetensors.torch.safe_open = observe_loader(safetensors.torch.safe_open)
    if "diffswap" in args.backends:
        import dlib
        dlib.shape_predictor = observe_loader(dlib.shape_predictor)

    isolation = runtime_paths(ROOT)
    if not isolation["ready"]:
        raise RuntimeError("Python paths escape this checkout: " + "; ".join(isolation["errors"]))
    if not torch.cuda.is_available():
        raise RuntimeError("This optional smoke check requires an NVIDIA CUDA GPU.")

    source = folder / "public-upscaled-input.png"
    with Image.open(ROOT / "data/test/image.png") as photo:
        photo.convert("RGB").resize((1280, 1280), Image.Resampling.LANCZOS).save(source)

    last_message = [None]
    def progress(fraction, message):
        if message != last_message[0]:
            print(f"{fraction:6.1%} {message}", flush=True)
            last_message[0] = message

    # Include the optional detector path whose former local-variable shadowing
    # prevented it from loading. Both engines release their models between stages.
    pipeline = ProtectionPipeline(
        ROOT, shield_factory=lambda: FaceShieldProtector(root_dir=ROOT, enable_mtcnn=True)
    )
    protected = pipeline.protect(
        source, folder / "protection", max_size=1280, phantom_iterations=10,
        iterations=1, progress=progress, mode=args.mode,
    )
    for name in ("source_path", "stage1_path", "output_path"):
        if not protected.get(name):
            continue
        with Image.open(protected[name]) as picture:
            if picture.size != (1280, 1280):
                raise AssertionError(f"{name} did not preserve 1280x1280 output")
    comparisons = {}
    rows = []
    for backend in args.backends:
        comparison = run_protection_test(
            root_dir=ROOT, output_dir=folder / "swaps" / backend, source_path=protected["source_path"],
            result_path=protected["output_path"], stage1_path=protected["stage1_path"],
            mode=args.mode, backend=backend, jpeg95=True, progress=progress,
        )
        comparisons[backend] = comparison
        rows.extend(row for condition in comparison["conditions"] for row in condition["rows"])
    expected = (6 if args.mode == "phantomseal_faceshield" else 4) * len(args.backends)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} swaps for the selected methods and PNG/JPEG95")
    for row in rows:
        with Image.open(row["swap_path"]) as picture:
            picture.verify()
    modules_outside = sorted({
        str(Path(module.__file__).resolve()) for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and not str(module.__file__).startswith("<")
        and not Path(module.__file__).resolve().is_relative_to(ROOT)
    })
    unexpected_weights = [path for path in weight_reads if not path.startswith("weights/")]
    report = {
        "application": "Rani Shield", "date_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reduced-step execution and local-dependency verification; not an efficacy evaluation.",
        "source": "Upscaled public repository portrait, not native 1280-resolution evidence.",
        "python": str(Path(sys.executable).relative_to(ROOT)),
        "python_base": str(Path(sys.base_prefix).relative_to(ROOT)),
        "gpu": torch.cuda.get_device_name(0), "elapsed_seconds": round(time.perf_counter() - started, 2),
        "dimensions": [1280, 1280], "phantomseal_steps": 10 if args.mode == "phantomseal_faceshield" else 0,
        "faceshield_steps": 1 if args.mode == "phantomseal_faceshield" else 0,
        "optional_mtcnn_enabled": args.mode == "phantomseal_faceshield", "swap_rows": len(rows),
        "mode": args.mode, "backends": args.backends,
        "comparisons": {backend: [{"condition": condition["name"], "scores": {
            row["key"]: row["score"] for row in condition["rows"]}} for condition in value["conditions"]]
            for backend, value in comparisons.items()},
        "conditions": [condition["name"] for condition in comparison["conditions"]],
        "outside_python_file_accesses": sorted(outside_reads),
        "outside_module_files": modules_outside, "network_attempts": network_attempts,
        "model_files": sorted(weight_reads), "models_outside_weights": unexpected_weights,
        "audit_scope": "Python open events, imported module files, torch.load, safetensors.safe_open and dlib.shape_predictor inputs. OS/driver native file access is not traced.",
    }
    report["passed"] = not (outside_reads or modules_outside or network_attempts or unexpected_weights)
    destination = folder / "report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {destination}", flush=True)
    if not report["passed"]:
        raise RuntimeError("Local-dependency checks failed; inspect the report.")
    print(f"PASS: 1280px {args.mode}, {len(rows)} real swaps and local-only Python/model files.", flush=True)


if __name__ == "__main__":
    main()
