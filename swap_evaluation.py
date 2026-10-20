"""Common PNG/JPEG benchmark contract for DiffSwap, InfoSwap and UniFace."""
from contextlib import contextmanager, nullcontext
import math
from pathlib import Path
import time

from image_io import load_rgb
from swap_adapters import BACKENDS, FaceAlignmentError, backend_key, create_adapter


@contextmanager
def swap_models(root, backend, device=None):
    from face_evaluation import FaceEvaluator
    evaluator = FaceEvaluator(root, device=device)
    adapter = None
    try:
        evaluator.load()
        adapter = create_adapter(root, backend, device=device)
        adapter.load()
        yield evaluator, adapter
    finally:
        if adapter is not None:
            adapter.close()
        evaluator.close()
        adapter = evaluator = None
        from protection_pipeline import release_gpu
        release_gpu()


def benchmark(root, clean_path, stage1_path, final_path, target_path, output_path,
              device=None, *, backend, models=None, reference_path=None, minimum_baseline=.8):
    """Score quantized outputs against the same uncompressed original identity."""
    import torch
    import torch.nn.functional as F
    from benchmark_chain import atomic_json, comparison, detection, sha256_file

    backend_key(backend)
    # Validate the public quality-gate argument before any expensive work.
    comparison({"clean": {}, "stage1": {}, "final": {}}, minimum_baseline)
    paths = {"clean": Path(clean_path).resolve(), "stage1": Path(stage1_path).resolve(),
             "final": Path(final_path).resolve()}
    images = {key: load_rgb(path) for key, path in paths.items()}
    if len({image.size for image in images.values()}) != 1:
        raise ValueError("All compared sources must have matching dimensions.")
    reference_path = Path(reference_path or clean_path).resolve()
    reference_image, target = load_rgb(reference_path), load_rgb(target_path)
    if target.size != (224, 224):
        raise ValueError("Use the installed prepared 224x224 target crop.")
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_dir = destination.parent / (destination.stem + "_images")
    image_dir.mkdir(exist_ok=True)
    started = time.perf_counter()

    def cosine(first, second):
        value = float(F.cosine_similarity(first, second, dim=1).item())
        if not math.isfinite(value):
            raise RuntimeError("Non-finite FaceNet similarity.")
        return value

    with (swap_models(root, backend, device) if models is None else nullcontext(models)) as (evaluator, adapter):
        found, box, _ = detection(evaluator, reference_image)
        if not found["detected"]:
            raise ValueError("The clean original has no detectable face; no reference can be established.")
        reference = evaluator.face_embedding(reference_image, box)
        target_detection, target_box, _ = detection(evaluator, target)
        target_embedding = evaluator.face_embedding(target, target_box) if target_detection["detected"] else None
        rows = {}
        generated_cache = {}
        for name, source in images.items():
            found, _, _ = detection(evaluator, source)
            row = {"source_path": str(paths[name]), "source_sha256": sha256_file(paths[name]),
                   "source_face_detected": found["detected"], "source_detection": found,
                   "swap_face_detected": False, "swap_detection": None,
                   "facenet_cosine_to_clean_source": None, "facenet_cosine_to_target": None,
                   "output_path": None, "status": "source_detection_failed"}
            rows[name] = row
            if not found["detected"]:
                continue
            # Single-stage tests pass the final file twice through the legacy
            # three-row contract. Reuse identical bytes instead of a second run.
            key = row["source_sha256"]
            if key not in generated_cache:
                try:
                    with torch.inference_mode():
                        generated = adapter.swap(source.copy(), target.copy(), seed=0)
                except FaceAlignmentError as error:
                    row.update(status="backend_alignment_failed", error=str(error))
                    continue
                if generated.mode != "RGB" or generated.width < 1 or generated.height < 1:
                    raise ValueError(f"{BACKENDS[backend]['label']} returned an invalid RGB image.")
                generated_cache[key] = generated
            output = generated_cache[key]
            path = image_dir / f"{name}_swap.png"
            output.save(path, format="PNG")
            quantized = load_rgb(path)
            detected, swap_box, _ = detection(evaluator, quantized)
            row.update(output_path=str(path), swap_face_detected=detected["detected"],
                       swap_detection=detected, status="ok" if detected["detected"] else "swap_detection_failed")
            if detected["detected"]:
                embedding = evaluator.face_embedding(quantized, swap_box)
                row["facenet_cosine_to_clean_source"] = cosine(embedding, reference)
                if target_embedding is not None:
                    row["facenet_cosine_to_target"] = cosine(embedding, target_embedding)
        report = {
            "schema_version": 1, "backend": backend, "benchmark": f"single_pair_{backend}_identity_ab",
            "rows": rows, "final_vs_stage1": comparison(rows, minimum_baseline),
            "reference_source": {"path": str(reference_path), "sha256": sha256_file(reference_path)},
            "target": {"path": str(Path(target_path).resolve()), "dimensions": list(target.size)},
            "method": {"generator": BACKENDS[backend]["label"],
                       "metric": "MTCNN + FaceNet cosine to the original clean PNG",
                       "generator_input": getattr(adapter, "preprocessing",
                           "Backend-specific preprocessing at native resolution; identical target and random seed for all variants."),
                       "seed": 0, "output_detection_failure": "Null score; no forced crop",
                       "metric_limitation": "FaceNet participates in PhantomSeal cloak selection; no universal protection claim."},
            "runtime": {"elapsed_seconds": time.perf_counter() - started},
        }
        atomic_json(destination, report)
        return report
