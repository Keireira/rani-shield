"""Lazy registry of real, independently selected face-swap backends."""
from importlib import import_module
from pathlib import Path

from asset_paths import MTCNN_FILES, weight_path


BACKENDS = {
    "simswap": {"label": "SimSwap", "size": 224},
    "diffswap": {"label": "DiffSwap", "size": 256},
    "infoswap": {"label": "InfoSwap", "size": 512},
    "uniface": {"label": "UniFace", "size": 256},
}
ADAPTER_CLASSES = {
    "diffswap": "DiffSwapAdapter", "infoswap": "InfoSwapAdapter", "uniface": "UniFaceAdapter",
}


class FaceAlignmentError(ValueError):
    """A backend's own detector/alignment could not prepare this face pair."""


def backend_key(value):
    if not isinstance(value, str) or value not in BACKENDS:
        raise ValueError("Choose SimSwap, DiffSwap, InfoSwap or UniFace.")
    return value


def evaluator_readiness(root_dir):
    required = [weight_path(root_dir, "facenet", "facenet_vggface2.pt"),
                *(weight_path(root_dir, "mtcnn", name) for name in MTCNN_FILES)]
    missing = [str(path.relative_to(Path(root_dir).resolve())) for path in required
               if not path.is_file() or not path.stat().st_size]
    return {"ready": not missing, "message": "Missing evaluator: " + ", ".join(missing) if missing else "Ready"}


def backend_readiness(root_dir, backend):
    backend_key(backend)
    evaluator = evaluator_readiness(root_dir)
    if backend == "simswap":
        required = [weight_path(root_dir, "simswap", name)
                    for name in ("latest_net_G.pth", "arcface_checkpoint.tar")]
        missing = [path.name for path in required if not path.is_file() or not path.stat().st_size]
        status = {"ready": not missing, "message": "Missing: " + ", ".join(missing) if missing else "Ready"}
    else:
        try:
            status = import_module(f"swap_adapters.{backend}").readiness(root_dir)
        except (ImportError, OSError, ValueError) as error:
            status = {"ready": False, "message": str(error)}
    messages = [item["message"] for item in (status, evaluator) if not item["ready"]]
    return {"key": backend, **BACKENDS[backend],
            "ready": status["ready"] and evaluator["ready"],
            "message": "; ".join(messages) if messages else "Ready"}


def create_adapter(root_dir, backend, device=None):
    backend_key(backend)
    if backend == "simswap":
        raise ValueError("SimSwap uses its existing audited benchmark implementation.")
    module = import_module(f"swap_adapters.{backend}")
    return getattr(module, ADAPTER_CLASSES[backend])(root_dir, device=device)
