"""Shared, standard-library-only contract for local model and data storage."""
from pathlib import Path


WEIGHTS_DIR = "weights"
DEFAULT_CHECKPOINT_DIR = WEIGHTS_DIR
DEFAULT_CLOAK_DIR = "weights/phantomseal/cloaks/mix_full"
MTCNN_FILES = ("pnet.pt", "rnet.pt", "onet.pt")


def weights_dir(root_dir):
    return project_path(root_dir, WEIGHTS_DIR)


def project_path(root_dir, value):
    """Resolve a runtime asset inside this project, including junction targets."""
    root = Path(root_dir).resolve()
    value = Path(value)
    target = value if value.is_absolute() else root / value
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Runtime paths must stay inside the project: {value}")
    return resolved


def weight_path(root_dir, *relative_parts):
    relative = Path(*relative_parts)
    if relative.anchor or ".." in relative.parts:
        raise ValueError("Weight paths must be relative to this project's weights directory.")
    return project_path(weights_dir(root_dir), relative)
