"""Shared, standard-library-only contract for local model and data storage."""
from pathlib import Path


WEIGHTS_DIR = "weights"
DEFAULT_CHECKPOINT_DIR = WEIGHTS_DIR
DEFAULT_CLOAK_DIR = "weights/phantomseal/cloaks/mix_full"
MTCNN_FILES = ("pnet.pt", "rnet.pt", "onet.pt")


def weights_dir(root_dir):
    return Path(root_dir).resolve() / WEIGHTS_DIR


def weight_path(root_dir, *relative_parts):
    relative = Path(*relative_parts)
    if relative.anchor or ".." in relative.parts:
        raise ValueError("Weight paths must be relative to this project's weights directory.")
    return weights_dir(root_dir) / relative
