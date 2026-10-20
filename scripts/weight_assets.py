"""Pure-standard-library asset layout and migration helpers; no model imports."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile


CHECKPOINT_DEST = "weights"
CLOAK_DEST = "weights/phantomseal/cloaks/mix_full"
MODEL_MANIFEST = "weights/manifests/phantomseal.json"
IMPORT_MANIFEST = "weights/manifests/imported-assets.json"
RELATIVE_CONFIG = {"checkpoint_dir": CHECKPOINT_DEST, "cloak_dir": CLOAK_DEST, "expected_cloaks": 214}
OLD_CONFIG = {"checkpoint_dir": ".cache/phantomseal-downloads/checkpoints", "cloak_dir": ".cache/phantomseal-downloads/data/vggface2_cloak/mix_full", "expected_cloaks": 214}
SD_FILES = ("tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
            "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors")
RUNTIME_RECORDS = {"generator": "weights/simswap/latest_net_G.pth", "arcface": "weights/simswap/arcface_checkpoint.tar", "facenet_vggface2": "weights/facenet/facenet_vggface2.pt"}
MTCNN_RECORDS = {name: f"weights/mtcnn/{name}.pt" for name in ("pnet", "rnet", "onet")}
PATH_MAP = (
    ("models/arcface50_checkpoint.tar", "weights/faceshield/arcface50_checkpoint.tar"),
    ("models/arcface100_checkpoint.tar", "weights/faceshield/arcface100_checkpoint.tar"),
    ("models/phantomseal/latest_net_G.pth", RUNTIME_RECORDS["generator"]),
    ("models/phantomseal/arcface_checkpoint.tar", RUNTIME_RECORDS["arcface"]),
    ("models/phantomseal/facenet_vggface2.pt", RUNTIME_RECORDS["facenet_vggface2"]),
    ("models/phantomseal/manifest.json", MODEL_MANIFEST),
    ("utils/unet/ip_adapter/ip-adapter_sd15.bin", "weights/ip-adapter/ip-adapter_sd15.bin"),
    *((".cache/models/sd15/" + name, "weights/sd15/" + name) for name in SD_FILES),
    *((".cache/models/ip-adapter/models/image_encoder/" + name, "weights/ip-adapter/models/image_encoder/" + name) for name in ("config.json", "model.safetensors")),
)
LEGACY_FILE = "utils/faceparser/FaceParser.pth"
LEGACY_DEST = "weights/faceparser/FaceParser.pth"
ATTACK_SOURCE = ".cache/torch/hub/checkpoints/pnet-6b6ef92b.pth"
ATTACK_DEST = "weights/mtcnn-attack/pnet-6b6ef92b.pth"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def is_link(path):
    details = Path(path).lstat()
    return stat.S_ISLNK(details.st_mode) or bool(getattr(details, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024))


def checked_destination(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Path must stay inside the project: {relative}")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            linked = is_link(current)
        except FileNotFoundError:
            continue
        if linked:
            raise ValueError(f"Refusing a link/junction: {current}")
    if not current.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes the project: {current}")
    return current


def local_path(root, value):
    root, value = Path(root).resolve(), Path(value)
    target = value if value.is_absolute() else root / value
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Migration cannot read or move assets outside this project: {target}")
    return checked_destination(root, target.relative_to(root))


def source_file(root, legacy, unified):
    return root / unified if (root / unified).is_file() else root / legacy


def bundled_mtcnn(root, name, *, installed_fallback=True, prefer_central=True):
    candidates = [root / f".venv/Lib/site-packages/facenet_pytorch/data/{name}.pt"]
    if prefer_central:
        candidates.insert(0, root / f"weights/mtcnn/{name}.pt")
    for path in candidates:
        if path.is_file():
            return path
    if installed_fallback:
        spec = importlib.util.find_spec("facenet_pytorch")
        if spec and spec.submodule_search_locations:
            for folder in spec.submodule_search_locations:
                path = Path(folder) / "data" / f"{name}.pt"
                if path.is_file():
                    return path
    raise ValueError(f"Missing bundled MTCNN {name}.pt. Run with the prepared project Python.")


def bank_images(folder):
    if not folder.is_dir() or is_link(folder):
        raise ValueError(f"Missing plain full cloak-bank directory: {folder}")
    images = []
    for path in sorted(folder.rglob("*")):
        if is_link(path):
            raise ValueError(f"Refusing a cloak-bank link/junction: {path}")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.append(path)
    if len(images) != 214:
        raise ValueError(f"Expected exactly 214 cloak images, found {len(images)} in {folder}.")
    return images


def rewrite_model_manifest(value, mtcnn=None):
    result = copy.deepcopy(value)
    if not isinstance(result, dict) or not isinstance(result.get("assets"), dict):
        raise ValueError("Invalid PhantomSeal model provenance manifest.")
    for name, relative in RUNTIME_RECORDS.items():
        if name not in result["assets"]:
            raise ValueError(f"Missing provenance record: {name}")
        record = result["assets"][name]
        if record.get("path") and record["path"] != relative:
            record.setdefault("previous_path", record["path"])
        record["path"] = relative
    for name, record in (mtcnn or {}).items():
        result["assets"][f"mtcnn_{name}"] = {**result["assets"].get(f"mtcnn_{name}", {}), **record,
            "path": MTCNN_RECORDS[name], "format": "facenet-pytorch bundled tensor state; copied without deserialization"}
    result["layout"] = "weights"
    return result


def atomic_copy(source, target, expected_sha, *, overwrite=False, payload=None):
    """Publish a complete verified copy; default publication never replaces."""
    import io
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        input_stream = io.BytesIO(payload) if payload is not None else Path(source).open("rb")
        with input_stream as stream, tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                output.write(chunk); digest.update(chunk)
            output.flush(); os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha:
            raise ValueError(f"Source changed while copying: {source}")
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                if os.name == "nt":
                    os.rename(temporary, target)
                else:
                    os.link(temporary, target)
            except FileExistsError:
                if not target.is_file() or is_link(target) or sha256_file(target) != expected_sha:
                    raise FileExistsError(f"Refusing to replace different existing asset: {target}")
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def atomic_json(target, value):
    encoded = payload_bytes(value)
    atomic_copy(target, target, hashlib.sha256(encoded).hexdigest(), overwrite=True, payload=encoded)


def translated_path(relative):
    relative = relative.replace("\\", "/")
    exact = dict((*PATH_MAP, (LEGACY_FILE, LEGACY_DEST), (ATTACK_SOURCE, ATTACK_DEST),
                  (OLD_CONFIG["checkpoint_dir"] + "/simswap/arcface_checkpoint.tar", RUNTIME_RECORDS["arcface"]),
                  (OLD_CONFIG["checkpoint_dir"] + "/faceshifter/model_ir_se50.pth", "weights/faceshifter/model_ir_se50.pth"),
                  (OLD_CONFIG["checkpoint_dir"] + "/hififace/hififace_opensource_299999.ckpt", "weights/hififace/hififace_opensource_299999.ckpt")))
    if relative in exact:
        return exact[relative]
    prefix = OLD_CONFIG["cloak_dir"] + "/"
    return CLOAK_DEST + "/" + relative[len(prefix):] if relative.startswith(prefix) else relative


def rewrite_import_manifest(value, current):
    result = copy.deepcopy(value)
    if not isinstance(result, dict):
        raise ValueError("Invalid imported-assets provenance manifest.")
    unique = {}
    for original in result.get("assets", []):
        row = dict(original)
        old = row["path"]
        row["path"] = translated_path(old)
        if row["path"] != old:
            row.setdefault("previous_paths", []).append(old)
        if row["path"] in current:
            row.update({key: current[row["path"]][key] for key in ("sha256", "bytes")})
        if row["path"] in unique:
            unique[row["path"]].setdefault("previous_paths", []).extend(row.get("previous_paths", []))
            unique[row["path"]].setdefault("source_aliases", []).append(row.get("source_path"))
        else:
            unique[row["path"]] = row
    for relative, record in current.items():
        if relative not in unique:
            unique[relative] = dict(record)
    result.update(layout="weights", assets=list(unique.values()), configuration=dict(RELATIVE_CONFIG), source_model_manifest=MODEL_MANIFEST)
    result["total_bytes"] = sum(row["bytes"] for row in result["assets"])
    return result
