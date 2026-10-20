"""Single-pair SimSwap A/B check for clean -> PhantomSeal -> FaceShield.

Example (run separately after the protection job has released its GPU models):
  python benchmark_chain.py --root . --clean results/job/original.png \
    --stage1 results/job/stage1/phantomseal.png --final results/job/protected.png \
    --target data/phantomseal_cloaks/cage.jpg --out results/job/simswap_ab.json

The target must be an already prepared 224x224 face crop, like SimSwap's public
crop_224 examples. Source images retain their full dimensions; each source face
is independently detected and aligned to 224 before ArcFace identity extraction.
Outputs are quantized PNGs and independently detected/cropped by MTCNN for a
FaceNet similarity to the CLEAN source. Detection failure produces a null score.
This is one source/target pair and one swapper, not an attack-success benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from asset_paths import weight_path

import numpy as np
from PIL import Image
from image_io import load_rgb


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_images(images: dict[str, Image.Image], target: Image.Image) -> None:
    sizes = {name: image.size for name, image in images.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Clean, stage1, and final images must have identical dimensions: {sizes}")
    if target.size != (224, 224):
        raise ValueError("Use an already prepared 224x224 target face crop; benchmark does not resize it.")


def comparison(rows: dict, minimum_baseline: float = .8) -> dict:
    if isinstance(minimum_baseline, bool) or not isinstance(minimum_baseline, (int, float)) or not math.isfinite(minimum_baseline) or not 0 <= minimum_baseline <= 1:
        raise ValueError("Minimum baseline cosine must be a finite number from 0 to 1.")
    clean = rows.get("clean", {})
    baseline_score = clean.get("facenet_cosine_to_clean_source")
    baseline_valid = (
        clean.get("source_face_detected") is True and clean.get("swap_face_detected") is True
        and isinstance(baseline_score, (int, float)) and not isinstance(baseline_score, bool)
        and math.isfinite(baseline_score) and baseline_score >= minimum_baseline
    )
    first = rows["stage1"].get("facenet_cosine_to_clean_source")
    final = rows["final"].get("facenet_cosine_to_clean_source")
    measurable = (
        isinstance(first, (int, float)) and not isinstance(first, bool)
        and isinstance(final, (int, float)) and not isinstance(final, bool)
        and math.isfinite(first) and math.isfinite(final)
        and all(rows[name].get("source_face_detected") is True and rows[name].get("swap_face_detected") is True for name in ("stage1", "final"))
    )
    raw_delta = float(final - first) if measurable else None
    delta = raw_delta if baseline_valid else None
    direction = None
    if delta is not None:
        direction = ("higher_clean_identity_similarity" if delta > 0 else
                     "lower_clean_identity_similarity" if delta < 0 else "unchanged")
    return {
        "final_minus_stage1_cosine": delta,
        "raw_final_minus_stage1_cosine": raw_delta,
        "direction": direction,
        "baseline_status": "numeric_gate_met_visual_review_required" if baseline_valid else "inconclusive_weak_baseline",
        "baseline_cosine": baseline_score if isinstance(baseline_score, (int, float)) and not isinstance(baseline_score, bool) and math.isfinite(baseline_score) else None,
        "minimum_baseline_cosine": minimum_baseline,
        "both_swap_faces_detected": all(rows[name].get("swap_face_detected", False) for name in ("stage1", "final")),
        "interpretation": (
            "The configurable baseline cutoff is a quality gate, not confidence or a calibrated verification threshold. "
            "If the unprotected swap is below it, no defense comparison is inferred. Even above it, "
            "inspect whether the unprotected swap actually resembles the source. "
            "For a usable baseline, a positive difference means the final swap resembles the clean source more "
            "according to FaceNet for this pair. It may indicate weaker identity protection "
            "for this example. Detection changes and a single pair do not establish general efficacy."
        ),
    }


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def detection(engine, image):
    try:
        box, landmarks, face_count = engine.detect(image)
        return {"detected": True, "face_count": int(face_count),
                "box": np.asarray(box).tolist(), "error": None}, box, landmarks
    except ValueError as error:
        return {"detected": False, "face_count": 0, "box": None, "error": str(error)}, None, None


@contextmanager
def swap_models(root: Path, device: str | None = None):
    """Load once for a series of real generator calls; always release GPU models."""
    import torch
    from phantomseal_runtime import load_generator_weights
    from face_evaluation import SimSwapEvaluator
    from third_party.phantomseal.simswap_networks import Generator_Adain_Upsample

    engine = SimSwapEvaluator(root, device=device)
    generator = None
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        print("Loading audited ArcFace, FaceNet, and SimSwap weights", flush=True)
        engine.load()
        generator = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512,
                                            n_blocks=9, deep=False)
        state = load_generator_weights(weight_path(root, "simswap", "latest_net_G.pth"))
        generator.load_state_dict(state, strict=True)
        del state
        generator.eval().requires_grad_(False).to(engine.device)
        yield engine, generator
    finally:
        generator = None
        engine.close()


def benchmark(root: Path, clean_path: Path, stage1_path: Path, final_path: Path,
              target_path: Path, output_path: Path, device: str | None = None,
              *, models=None, reference_path: Path | None = None, minimum_baseline: float = .8) -> dict:
    # Imports do not run when this file is inspected or its pure helpers are tested.
    import torch
    import torch.nn.functional as F

    root = root.resolve()
    from phantomseal_runtime import image_tensor
    from benchmark_alignment import aligned_crop

    paths = {"clean": clean_path.resolve(), "stage1": stage1_path.resolve(), "final": final_path.resolve()}
    images = {name: load_rgb(path) for name, path in paths.items()}
    target_path = target_path.resolve()
    target = load_rgb(target_path)
    reference_path = (reference_path or clean_path).resolve()
    reference_image = load_rgb(reference_path)
    validate_images(images, target)
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".json":
        output_path = output_path / "benchmark.json"
    image_dir = output_path.parent / (output_path.stem + "_images")
    image_dir.mkdir(parents=True, exist_ok=True)
    target.save(image_dir / "target.png")

    started = time.perf_counter()
    with (swap_models(root, device) if models is None else nullcontext(models)) as (engine, generator):
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
        target_tensor = image_tensor(target, engine.device).contiguous()
        source_detection, clean_box, clean_landmarks = detection(engine, reference_image)
        if not source_detection["detected"]:
            raise ValueError("Clean source face is not detected; cannot establish the reference identity.")
        with torch.inference_mode():
            reference = engine.face_embedding(reference_image, clean_box)
            reference_arcface = engine.identity(image_tensor(aligned_crop(reference_image, clean_landmarks), engine.device))
            target_detection, target_box, _ = detection(engine, target)
            target_reference = engine.face_embedding(target, target_box) if target_detection["detected"] else None

        def cosine(embedding, other) -> float:
            value = float(F.cosine_similarity(embedding, other, dim=1).item())
            if not math.isfinite(value):
                raise RuntimeError("Non-finite FaceNet/ArcFace similarity")
            return value

        rows = {}
        for name in ("clean", "stage1", "final"):
            print(f"Running fixed-target SimSwap: {name}", flush=True)
            source = images[name]
            found, box, landmarks = detection(engine, source)
            row = {
                "source_path": str(paths[name]),
                "source_sha256": sha256_file(paths[name]),
                "source_face_detected": found["detected"],
                "source_detection": found,
                "source_arcface_cosine_to_clean": None,
                "swap_face_detected": False,
                "swap_detection": None,
                "facenet_cosine_to_clean_source": None,
                "facenet_cosine_to_target": None,
                "output_path": None,
                "aligned_source_path": None,
            }
            if not found["detected"]:
                row["status"] = "source_detection_failed"
                rows[name] = row
                continue
            with torch.inference_mode():
                face = image_tensor(aligned_crop(source, landmarks), engine.device).contiguous()
                source_identity = engine.identity(face)
                row["source_arcface_cosine_to_clean"] = cosine(source_identity, reference_arcface)
                generated = generator(target_tensor, source_identity).clamp(0, 1)
                if not torch.isfinite(generated).all():
                    raise RuntimeError(f"SimSwap produced non-finite pixels for {name}")
                output_array = (generated[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
                aligned_array = (face[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
            aligned_path = image_dir / f"{name}_source_aligned.png"
            Image.fromarray(aligned_array).save(aligned_path)
            swap_path = image_dir / f"{name}_swap.png"
            swap_image = Image.fromarray(output_array)
            swap_image.save(swap_path)
            row["output_path"] = str(swap_path)
            row["aligned_source_path"] = str(aligned_path)
            swapped, swap_box, _ = detection(engine, swap_image)
            row["swap_detection"] = swapped
            row["swap_face_detected"] = swapped["detected"]
            if swapped["detected"]:
                with torch.inference_mode():
                    embedding = engine.face_embedding(swap_image, swap_box)
                    row["facenet_cosine_to_clean_source"] = cosine(embedding, reference)
                    if target_reference is not None:
                        row["facenet_cosine_to_target"] = cosine(embedding, target_reference)
                row["status"] = "ok"
            else:
                row["status"] = "swap_detection_failed"
            rows[name] = row

        report = {
            "schema_version": 1,
            "benchmark": "single_pair_simswap224_identity_ab",
            "scope": "One public source/target pair; no ASR, thresholded success rate, robustness, or general protection claim.",
            "method": {
                "source": "Independent MTCNN detection, largest face >=0.9 confidence; official SimSwap five-pose minimum-error similarity crop224, OpenCV bilinear/constant-black border; ImageNet-normalized nearest112 ArcFace and L2 identity",
                "target": "Same pre-cropped224 RGB target tensor in all three generator calls; no resizing",
                "generator": "SimSwap Generator_Adain_Upsample n_blocks=9 deep=False, eval float32",
                "metric": "MTCNN detection/extraction plus independent FaceNet-vggface2 normalized cosine to CLEAN source",
                "metric_limitation": "FaceNet is independent of the optimized ArcFace encoder, but it was also used for cloak selection; this is not an entirely unseen evaluator.",
                "output_detection_failure": "Reported explicitly with null cosine; no reuse of clean landmarks or forced output crop",
                "quantization": "Swapped output is rounded to 8-bit PNG before face detection and scoring",
                "seed": 0,
            },
            "source_dimensions": list(images["clean"].size),
            "reference_source": {"path": str(reference_path), "sha256": sha256_file(reference_path), "dimensions": list(reference_image.size)},
            "target": {"path": str(target_path), "sha256": sha256_file(target_path), "dimensions": list(target.size), "detection": target_detection},
            "rows": rows,
            "final_vs_stage1": comparison(rows, minimum_baseline),
            "weights": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in (("latest_net_G.pth", weight_path(root, "simswap", "latest_net_G.pth")),
                    ("arcface_checkpoint.tar", Path(engine.branches["simswap"].weight_path)),
                    ("facenet_vggface2.pt", weight_path(root, "facenet", "facenet_vggface2.pt")))
            },
            "runtime": {"device": str(engine.device), "torch": torch.__version__,
                        "python": platform.python_version(), "platform": platform.platform(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_vram_gb": torch.cuda.max_memory_allocated(engine.device) / 1024**3 if engine.device.type == "cuda" else 0.0},
        }
        atomic_json(output_path, report)
        print(json.dumps({"report": str(output_path), "rows": {
            name: {"source_detected": row["source_face_detected"], "swap_detected": row["swap_face_detected"],
                   "cosine_to_clean": row["facenet_cosine_to_clean_source"]}
            for name, row in rows.items()}, "final_vs_stage1": report["final_vs_stage1"]}, indent=2), flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--minimum-baseline", type=float, default=.8, help="Heuristic minimum clean-swap cosine; not confidence. Visual baseline review is still required.")
    args = parser.parse_args()
    from scripts.runtime_environment import configure_environment
    configure_environment(args.root)
    target = args.target or args.root / "data" / "phantomseal_cloaks" / "cage.jpg"
    benchmark(args.root, args.clean, args.stage1, args.final, target, args.out, args.device, minimum_baseline=args.minimum_baseline)


if __name__ == "__main__":
    main()
