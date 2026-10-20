"""Import optional protection/swap weights from an explicitly supplied bundle.

Accepts an extracted checkpoints directory (or its parent), a unified weights
directory, or a ZIP containing those allowlisted files. Models are copied into
this project's weights/; nothing is loaded or executed. InfoSwap's missing
readout statistics are downloaded from a pinned author revision and checked
against both SHA-256 and byte count. Runtime inference never calls this script.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Callable
from urllib.request import urlopen
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.weight_assets import (
    atomic_json, checked_destination, is_link, sha256_file,
)


MODEL_FILES = {
    "nullswap": ("nullswap/epoch_0001_step_012000.pth",),
    "diffswap": (
        "diffswap/diffswap.pth", "diffswap/glint360k_r100.pth",
        "diffswap/shape_predictor_68_face_landmarks.dat",
    ),
    "infoswap": (
        "infoswap/model_128_ir_se50.pth",
        "infoswap/w_kernel_smooth/ckpt_ks_E.pth",
        "infoswap/w_kernel_smooth/ckpt_ks_G.pth",
        "infoswap/w_kernel_smooth/ckpt_ks_I.pth",
    ),
    "uniface": ("uniface/500000.pt",),
}
READOUT_REVISION = "0484ef5fcb35bf811f2c3ec52ecf77ccade822db"
READOUT_HASHES = (
    ("4b6ee5478df151a2a5085ed8c14c77cce40cefff979c96ede33cc811443eaa44", 16777987),
    ("cfb6fef47c03bdc93b365a16ae4112d2063d4c4359550db7873fb65718be9499", 4195075),
    ("84322cc2fb155eb11b63197f67a514f91bd3ff852a0a0a303953cbf4632425a4", 4195075),
    ("d96c1ffe34aacb1d7a874e5b7301f2b4eb1c01319ccab06debafa4732bdcd60f", 4195075),
    ("2dd2d187cedc95d51bb12c48acecb45aa004cddc3894dfc2b604408a5c994581", 2097923),
    ("6f75ba23f6a2b5655ebd31bc41bb05f3db49655f787e6bb46849d7d0e8f2265f", 2097923),
    ("69878c10741fb41a53f7528cdb9b102be6986b409dca61631c614d071b8d7c21", 2097923),
    ("84368a506d8c3f32d2dffb11a691df485d962cc0703240e7a1fb33d6f2c1d40a", 2097923),
    ("ddb580d225f878b0250c3ea70dd98545887c548c3e4a20635c469f6281af8f2d", 1049350),
    ("d671e5c8b781e50a77beee7218634133cfcc8ac57e508686610964af349a251f", 1049350),
    ("68702548c72f19b79112e10d18b6710d8d861dfe781f7a7269f338e3d5ea380c", 1049350),
)
MANIFEST = "weights/manifests/optional-models.json"


@dataclass
class Source:
    origin: str
    size: int
    open: Callable


class CheckpointBundle:
    def __init__(self, path, stack):
        self.path = Path(path).expanduser()
        if not self.path.exists() or is_link(self.path):
            raise ValueError("Specify an existing plain checkpoints directory or ZIP file.")
        self.path = self.path.resolve()
        self.archive = None
        if self.path.is_file():
            if not zipfile.is_zipfile(self.path):
                raise ValueError("Only ZIP bundles are supported; extract other archives first.")
            self.archive = stack.enter_context(zipfile.ZipFile(self.path))
        elif (self.path / "checkpoints").is_dir():
            self.path = checked_destination(self.path, "checkpoints")

    def find(self, relative, *, optional=False):
        if self.archive is None:
            path = checked_destination(self.path, relative)
            if not path.exists() and optional:
                return None
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty checkpoint: {relative}")
            return Source(f"local:{self.path.name}/{relative}", path.stat().st_size,
                          lambda: path.open("rb"))
        aliases = {relative, f"checkpoints/{relative}", f"weights/{relative}"}
        matches = [info for info in self.archive.infolist() if info.filename in aliases]
        if not matches and optional:
            return None
        if len(matches) != 1:
            raise ValueError(f"ZIP must contain exactly one allowlisted member for {relative}.")
        info = matches[0]
        if info.is_dir() or not info.file_size or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Empty, directory or linked ZIP checkpoint: {relative}")
        return Source(f"zip:{self.path.name}!/{info.filename}", info.file_size,
                      lambda: self.archive.open(info))


def _hash_source(source):
    digest, size = hashlib.sha256(), 0
    with source.open() as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size != source.size:
        raise ValueError(f"Checkpoint changed while hashing: {source.origin}")
    return digest.hexdigest()


def _publish_source(source, target, expected_sha):
    """Publish only complete verified bytes; never replace a differing file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest, count = hashlib.sha256(), 0
        with source.open() as stream, tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=target.name + ".",
            suffix=".part", delete=False,
        ) as output:
            temporary = Path(output.name)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(chunk)
                if count > source.size:
                    raise ValueError(f"Unexpected asset byte count: {source.origin}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if count != source.size or digest.hexdigest() != expected_sha:
            raise ValueError(f"Asset checksum/size mismatch: {source.origin}")
        try:
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or is_link(target) or sha256_file(target) != expected_sha:
                raise FileExistsError(f"Refusing to replace a different model: {target}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _models(value):
    values = [value] if isinstance(value, str) else list(value)
    if values == ["all"]:
        return list(MODEL_FILES)
    if not values or any(name not in MODEL_FILES for name in values):
        raise ValueError("Select nullswap, diffswap, infoswap, uniface, or all by itself.")
    return list(dict.fromkeys(values))


def import_assets(checkpoints, root=PROJECT, *, models=("all",), dry_run=False, download=True):
    root = Path(root).expanduser()
    if not root.is_dir() or is_link(root):
        raise ValueError("Destination must be an existing plain project directory.")
    root = root.resolve()
    selected = _models(models)
    manifest_path = checked_destination(root, MANIFEST)
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not isinstance(previous, dict) or not isinstance(previous.get("assets", []), list):
        raise ValueError("Invalid existing optional-models provenance manifest.")
    plan = []
    with ExitStack() as stack:
        bundle = CheckpointBundle(checkpoints, stack)
        for model in selected:
            for relative in MODEL_FILES[model]:
                source = bundle.find(relative)
                plan.append((model, relative, source, _hash_source(source), "copy"))
        if "infoswap" in selected:
            for index, (expected_sha, size) in enumerate(READOUT_HASHES):
                relative = f"infoswap/readout/readout_layer{index}.pth"
                target = checked_destination(root, f"weights/{relative}")
                url = (f"https://raw.githubusercontent.com/GGGHSL/InfoSwap-master/"
                       f"{READOUT_REVISION}/modules/weights128/readout_layer{index}.pth")
                if target.is_file() and target.stat().st_size == size and sha256_file(target) == expected_sha:
                    source = Source(url, size, lambda: None)
                else:
                    source = bundle.find(relative, optional=True)
                    if source is not None:
                        if source.size != size or _hash_source(source) != expected_sha:
                            raise ValueError(f"InfoSwap readout differs from pinned official release: {relative}")
                    else:
                        if not download:
                            raise FileNotFoundError(f"Missing {relative}; allow downloads or provide it in the bundle.")
                        source = Source(url, size, lambda url=url: urlopen(url, timeout=60))
                plan.append(("infoswap", relative, source, expected_sha,
                             "download" if source.origin.startswith("https:") else "copy"))
        # Validate every destination/conflict before writing any model.
        rows = []
        for model, relative, source, sha, action in plan:
            target = checked_destination(root, f"weights/{relative}")
            if target.exists():
                if not target.is_file() or target.stat().st_size != source.size or sha256_file(target) != sha:
                    raise FileExistsError(f"Different existing model; move it aside before importing: {target}")
                action = "skip"
            rows.append({"model": model, "path": f"weights/{relative}", "sha256": sha,
                         "bytes": source.size, "source": source.origin, "action": action})
        report = {"schema_version": 1, "layout": "weights", "models": selected,
                  "dry_run": dry_run, "assets": rows,
                  "copied": sum(row["action"] == "copy" for row in rows),
                  "downloaded": sum(row["action"] == "download" for row in rows),
                  "skipped": sum(row["action"] == "skip" for row in rows)}
        if dry_run:
            return report
        for row, (_, _, source, _, _) in zip(rows, plan):
            if row["action"] != "skip":
                target = checked_destination(root, row["path"])
                _publish_source(source, target, row["sha256"])
        merged = {record["path"]: record for record in previous.get("assets", [])}
        merged.update({row["path"]: {key: value for key, value in row.items() if key != "action"}
                       for row in rows})
        atomic_json(manifest_path, {"schema_version": 1, "layout": "weights",
                                   "assets": [merged[key] for key in sorted(merged)]})
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="Explicit extracted checkpoints folder, its parent, unified weights folder, or ZIP.")
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--models", nargs="+", default=["all"], choices=[*MODEL_FILES, "all"])
    parser.add_argument("--dry-run", action="store_true", help="Hash/check and show the plan without writes or downloads.")
    parser.add_argument("--no-download", action="store_true", help="Require InfoSwap readout statistics locally.")
    args = parser.parse_args()
    report = import_assets(args.checkpoints, args.root, models=args.models,
                           dry_run=args.dry_run, download=not args.no_download)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Optional model import failed: {error}")
