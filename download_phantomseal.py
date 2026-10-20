"""Fetch the minimal public PhantomSeal/SimSwap assets, without loading models.

Run with the project's Python: python download_phantomseal.py [--root PROJECT].
The default root is the directory containing this script. Downloads are HTTPS
with certificate verification; SHA-256 records transfer integrity, not an
independently published upstream signature. No pickle or Torch loader is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import requests
from asset_paths import weight_path
from scripts.weight_assets import atomic_copy, bundled_mtcnn


SIMSWAP_RELEASE = "https://github.com/neuralchen/SimSwap/releases/download/1.0/"
GENERATOR_URL = SIMSWAP_RELEASE + "checkpoints.zip"
ARCFACE_URL = SIMSWAP_RELEASE + "arcface_checkpoint.tar"
FACENET_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/"
    "20180402-114759-vggface2.pt"
)
SIMSWAP_COMMIT = "bd7b7686a17f41dd11cfcd5d82f7e4c5eb94b780"
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
MAX_GENERATOR_BYTES = 512 * 1024 * 1024
GENERATOR_MEMBERS = frozenset(
    {"people/latest_net_G.pth", "checkpoints/people/latest_net_G.pth"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path, **metadata: object) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=path.name + ".", suffix=".part", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download_file(url: str, destination: Path, *, retries: int = 3) -> dict:
    """Stream an HTTPS download into a sibling temporary file, then replace."""
    if urlsplit(url).scheme != "https":
        raise ValueError("Only HTTPS download URLs are accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary: Path | None = None
        try:
            with requests.get(
                url, stream=True, timeout=(20, 90), verify=True,
                headers={"User-Agent": "PhantomSeal-minimal-assets/1.0"},
            ) as response:
                response.raise_for_status()
                if urlsplit(response.url).scheme != "https":
                    raise ValueError("Refusing a redirect to non-HTTPS")
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise ValueError("The server returned HTML instead of a model")
                total = int(response.headers.get("Content-Length", "0"))
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Download exceeds the 1 GiB safety limit")
                digest = hashlib.sha256()
                downloaded = 0
                next_progress = 16 * 1024 * 1024
                print(f"Downloading {destination.name} (attempt {attempt}/{retries})", flush=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent,
                    prefix=destination.name + ".", suffix=".part", delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > MAX_DOWNLOAD_BYTES:
                            raise ValueError("Download exceeds the 1 GiB safety limit")
                        stream.write(chunk)
                        digest.update(chunk)
                        if downloaded >= next_progress:
                            suffix = f" / {total / 1048576:.1f} MiB" if total else ""
                            print(f"  {downloaded / 1048576:.1f} MiB{suffix}", flush=True)
                            next_progress += 16 * 1024 * 1024
                    stream.flush()
                    os.fsync(stream.fileno())
                if not downloaded or (total and downloaded != total):
                    raise ValueError(f"Incomplete download: {downloaded} of {total} bytes")
                os.replace(temporary, destination)
                print(f"Saved {destination.name}: {downloaded:,} bytes", flush=True)
                return {"url": url, "bytes": downloaded, "sha256": digest.hexdigest()}
        except (OSError, ValueError, requests.RequestException) as error:
            last_error = error
            print(f"Download attempt {attempt} failed: {error}", flush=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def extract_generator(archive: Path, destination: Path) -> str:
    """Extract exactly one allowlisted regular file; never use extractall()."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with zipfile.ZipFile(archive) as zipped:
            candidates = [info for info in zipped.infolist() if info.filename in GENERATOR_MEMBERS]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one allowlisted SimSwap generator in archive")
            info = candidates[0]
            member = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                info.is_dir() or member.is_absolute() or ".." in member.parts
                or "\\" in info.filename or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                or not 0 < info.file_size <= MAX_GENERATOR_BYTES
                or bool(info.flag_bits & 1)
            ):
                raise ValueError("Invalid generator ZIP member")
            written = 0
            with zipped.open(info, "r") as source, tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent,
                prefix=destination.name + ".", suffix=".part", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    written += len(chunk)
                    if written > MAX_GENERATOR_BYTES:
                        raise ValueError("Uncompressed generator exceeds safety limit")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if written != info.file_size:
                raise ValueError("Incomplete generator extraction")
            os.replace(temporary, destination)
            return info.filename
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reusable(path: Path, previous: dict, *, force: bool) -> bool:
    if force or not path.is_file() or not isinstance(previous, dict):
        return False
    if previous.get("bytes") != path.stat().st_size:
        return False
    if previous.get("sha256") != sha256_file(path):
        return False
    print(f"Verified existing {path.name}: {path.stat().st_size:,} bytes", flush=True)
    return True


def cloak_provenance(root: Path) -> dict:
    bank = root / "data" / "phantomseal_cloaks"
    files = sorted(p for p in bank.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    record = {
        "description": "Public SimSwap crop_224 demonstration portraits used as a small local cloak bank.",
        "adaptation_notice": (
            "This is not PhantomSeal's VGGFace2 experimental cloak bank. "
            "Paper protection/tracing scores do not apply to this replacement."
        ),
        "source_repository": "https://github.com/neuralchen/SimSwap",
        "source_commit": SIMSWAP_COMMIT,
        "source_directory": f"https://github.com/neuralchen/SimSwap/tree/{SIMSWAP_COMMIT}/crop_224",
        "upstream_license": "SimSwap repository: CC-BY-NC-4.0; images retain applicable original rights.",
        "image_count": len(files),
        "files": [file_record(p, root, source_url=f"https://raw.githubusercontent.com/neuralchen/SimSwap/{SIMSWAP_COMMIT}/crop_224/{p.name}") for p in files],
    }
    if files:
        atomic_json(bank / "PROVENANCE.json", record)
    return record


def install(root: Path, *, force: bool = False) -> dict:
    root = root.expanduser().resolve()
    model_dir = weight_path(root, "simswap")
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = weight_path(root, "manifests", "phantomseal.json")
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    previous_assets = previous.get("assets", {})
    if not isinstance(previous_assets, dict):
        previous_assets = {}
    manifest = {
        **previous,
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verification": "HTTPS certificate verification and locally calculated SHA-256; no upstream published hashes available.",
        "model_loading": "No models were deserialized or executed by this installer.",
        "sources": {**previous.get("sources", {}),
            "simswap": "https://github.com/neuralchen/SimSwap/blob/main/download-weights.sh",
            "facenet": "https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py",
            "phantomseal": "https://github.com/LiangqinRen/PhantomSeal",
        },
        "assets": dict(previous_assets),
    }
    generator = model_dir / "latest_net_G.pth"
    if reusable(generator, previous_assets.get("generator", {}), force=force):
        manifest["assets"]["generator"] = {**previous_assets["generator"], "path": generator.relative_to(root).as_posix()}
    else:
        archive = weight_path(root, "downloads", "checkpoints.zip")
        archive_info = download_file(GENERATOR_URL, archive)
        try:
            member = extract_generator(archive, generator)
            manifest["assets"]["generator"] = file_record(
                generator, root, url=GENERATOR_URL, archive_member=member,
                archive=archive_info, format="torch state_dict; not deserialized",
            )
        finally:
            archive.unlink(missing_ok=True)
    atomic_json(manifest_path, manifest)
    for key, filename, url, format_name in (
        ("arcface", "arcface_checkpoint.tar", ARCFACE_URL, "torch serialized nn.Module; not deserialized"),
        ("facenet_vggface2", "facenet_vggface2.pt", FACENET_URL, "torch state_dict; not deserialized"),
    ):
        path = (weight_path(root, "facenet", filename) if key == "facenet_vggface2" else model_dir / filename)
        if reusable(path, previous_assets.get(key, {}), force=force):
            manifest["assets"][key] = {**previous_assets[key], "path": path.relative_to(root).as_posix()}
        else:
            info = download_file(url, path)
            manifest["assets"][key] = file_record(path, root, url=url, format=format_name)
            if manifest["assets"][key]["sha256"] != info["sha256"]:
                raise RuntimeError(f"Post-download hash mismatch for {filename}")
        atomic_json(manifest_path, manifest)
    for name in ("pnet", "rnet", "onet"):
        path = weight_path(root, "mtcnn", name + ".pt")
        key = "mtcnn_" + name
        if reusable(path, previous_assets.get(key, {}), force=force):
            manifest["assets"][key] = {**previous_assets[key], "path": path.relative_to(root).as_posix()}
        else:
            packaged = bundled_mtcnn(root, name, prefer_central=False)
            digest = sha256_file(packaged)
            atomic_copy(packaged, path, digest, overwrite=True)
            manifest["assets"][key] = file_record(path, root, source_package="facenet-pytorch",
                format="bundled tensor state; copied without deserialization")
        atomic_json(manifest_path, manifest)
    manifest["layout"] = "weights"
    manifest["cloak_bank"] = cloak_provenance(root)
    atomic_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Cloak portraits: {manifest['cloak_bank']['image_count']}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force", action="store_true", help="Redownload even if manifest hashes match")
    arguments = parser.parse_args()
    install(arguments.root, force=arguments.force)


if __name__ == "__main__":
    main()
