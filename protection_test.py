"""One-result SimSwap check. Imports and readiness never create GPU models."""
from __future__ import annotations

import gc
import io
import math
from pathlib import Path
from asset_paths import weight_path

from PIL import Image, ImageOps


TARGETS = {"cage": {"label": "Demo face 1", "file": "cage.jpg"},
           "wlh": {"label": "Demo face 2", "file": "wlh.jpg"},
           "zxy": {"label": "Demo face 3", "file": "zxy.jpg"}}
METRIC = "FaceNet cosine to clean source"
NOTE = ("Lower similarity means less copying of the clean source identity in this example. "
        "This checks one SimSwap model in the source-identity role, not general protection or "
        "destination-face replacement. FaceNet was used for cloak selection and SimSwap was an "
        "optimization branch. Detection failures have no similarity score. JPEG95 is applied "
        "equally to each input after protection; the identity reference stays the original clean PNG.")


def target_options():
    return [{"key": key, "label": value["label"]} for key, value in TARGETS.items()]


def testing_readiness(root_dir):
    """File-only checks for the existing actual SimSwap benchmark runtime."""
    root = Path(root_dir).resolve()
    from phantomseal_assets import ensemble_readiness
    ensemble = ensemble_readiness(root)
    required = ["benchmark_chain.py", "benchmark_alignment.py",
                "third_party/phantomseal/simswap_networks.py"]
    required += ["data/phantomseal_cloaks/" + value["file"] for value in TARGETS.values()]
    missing = [name for name in required if not (root / name).is_file()]
    generator = weight_path(root, "simswap", "latest_net_G.pth")
    if not generator.is_file():
        missing.append(str(generator.relative_to(root)))
    ready = ensemble["ready"] and not missing
    message = "Ready · real SimSwap comparison" if ready else "; ".join(
        (["Missing test files: " + ", ".join(missing)] if missing else []) +
        ([ensemble["message"]] if not ensemble["ready"] else []))
    return {"ready": ready, "message": message, "targets": target_options()}


def _rgb(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _condition(image, name):
    if name == "png":
        return image
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95, subsampling=2)
    stream.seek(0)
    with Image.open(stream) as opened:
        return opened.convert("RGB")


def finite_score(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) else None)


def run_protection_test(*, root_dir, output_dir, source_path, result_path,
                        stage1_path=None, mode="phantomseal_faceshield", target_key="cage",
                        jpeg95=True, progress=None, benchmark_runner=None, models_context=None):
    """Return trusted local paths; Studio registers them before returning the API job.

    Injectable benchmark/context functions permit meaningful CPU lifecycle tests.
    Production uses benchmark_chain.benchmark and its shared swap_models context.
    """
    if target_key not in TARGETS:
        raise ValueError("Unknown SimSwap test target.")
    if mode not in ("phantomseal_faceshield", "phantomseal", "faceshield"):
        raise ValueError("Unknown protection method.")
    if type(jpeg95) is not bool:
        raise ValueError("jpeg95 must be a boolean.")
    chain = mode == "phantomseal_faceshield"
    if chain and not stage1_path:
        raise ValueError("The completed chain has no PhantomSeal intermediate image.")
    root, folder = Path(root_dir).resolve(), Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    source_path, result_path = Path(source_path).resolve(), Path(result_path).resolve()
    paths = {"clean": source_path, "final": result_path}
    if chain:
        paths["stage1"] = Path(stage1_path).resolve()
    target = _rgb(root / "data/phantomseal_cloaks" / TARGETS[target_key]["file"])
    if target.size != (224, 224):
        raise ValueError("The installed SimSwap target must be a prepared 224x224 crop.")
    target_path = folder / "target.png"
    target.save(target_path)
    modes = ["png", "jpeg95"] if jpeg95 else ["png"]
    labels = {"clean": "Clean source", "stage1": "PhantomSeal",
              "final": {"phantomseal_faceshield": "PhantomSeal + FaceShield",
                        "phantomseal": "PhantomSeal", "faceshield": "FaceShield"}[mode]}
    if benchmark_runner is None or models_context is None:
        from benchmark_chain import benchmark, swap_models
        benchmark_runner = benchmark_runner or benchmark
        context = models_context or swap_models
    else:
        context = models_context
    production_context = models_context is None
    models = None
    conditions = []
    def notify(fraction, message):
        if progress:
            progress(fraction, message)
    try:
        notify(.05, "Loading the SimSwap test models")
        with context(root) as models:
            for index, name in enumerate(modes):
                notify(.12 + .8 * index / len(modes), "Running SimSwap: " + ("PNG" if name == "png" else "JPEG95"))
                inputs = folder / "inputs" / name
                inputs.mkdir(parents=True, exist_ok=True)
                transformed = {}
                for key, path in paths.items():
                    destination = inputs / (key + ".png")
                    _condition(_rgb(path), name).save(destination)
                    transformed[key] = destination
                # The benchmark's three-input implementation can evaluate a single
                # method with its final image twice; only two distinct rows are published.
                report = benchmark_runner(root, transformed["clean"],
                    transformed.get("stage1", transformed["final"]), transformed["final"],
                    target_path, folder / (name + ".json"), models=models, reference_path=source_path)
                rows = []
                for key in (("clean", "stage1", "final") if chain else ("clean", "final")):
                    row = report["rows"][key]
                    detected = row.get("source_face_detected") is True and row.get("swap_face_detected") is True
                    rows.append({"key": key, "label": labels[key], "swap_path": row.get("output_path"),
                        "score": finite_score(row.get("facenet_cosine_to_clean_source")) if detected else None,
                        "source_detected": row.get("source_face_detected") is True,
                        "swap_detected": row.get("swap_face_detected") is True,
                        "status": str(row.get("status", "unknown"))})
                delta = None
                if chain and all(row["source_detected"] and row["swap_detected"] and row["score"] is not None for row in rows):
                    delta = rows[-1]["score"] - rows[1]["score"]
                conditions.append({"name": name, "rows": rows, "delta_final_minus_stage1": delta})
                notify(.12 + .8 * (index + 1) / len(modes), "SimSwap " + name.upper() + " complete")
    finally:
        # The caller tuple owns the generator too: drop it before empty_cache.
        models = None
        gc.collect()
        if production_context:
            from protection_pipeline import release_gpu
            release_gpu()
    notify(.98, "Preparing the test comparison")
    return {"target_path": str(target_path), "conditions": conditions, "metric": METRIC, "note": NOTE}
