"""Keep library downloads, caches and temporary files inside the project."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from asset_paths import project_path


CONTRACT = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
CACHE_PATHS = CONTRACT["cache_paths"]


def environment_values(root):
    """Return absolute local paths without creating directories or mutating env."""
    values = {name: str(project_path(root, relative)) for name, relative in CACHE_PATHS.items()}
    values.update(CONTRACT["settings"])
    values["PIP_CONFIG_FILE"] = os.devnull
    return values


def configure_environment(root):
    """Call before importing ML/UI packages so their cache constants are local."""
    values = environment_values(root)
    for name in CACHE_PATHS:
        Path(values[name]).mkdir(parents=True, exist_ok=True)
    os.environ.update(values)
    # A deprecated Transformers override can otherwise bypass HF_HOME/HF_HUB_CACHE.
    for name in CONTRACT["unset"]:
        os.environ.pop(name, None)
    tempfile.tempdir = values["TEMP"]
    return values
