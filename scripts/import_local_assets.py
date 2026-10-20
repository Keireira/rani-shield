"""Import a verified original or unified installation into weights/ only.

No downloads, model loading, environment copying, editor data or old-repo links.
Only the three bundled MTCNN states may be read from a source Python package.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.weight_assets import (ATTACK_DEST, ATTACK_SOURCE, CHECKPOINT_DEST, CLOAK_DEST, IMPORT_MANIFEST,
    LEGACY_DEST, LEGACY_FILE, MODEL_MANIFEST, MTCNN_RECORDS, OLD_CONFIG, PATH_MAP, RELATIVE_CONFIG,
    RUNTIME_RECORDS, SD_FILES, atomic_copy, atomic_json, bank_images, bundled_mtcnn,
    checked_destination, is_link, payload_bytes, rewrite_model_manifest, sha256_file, source_file)

FIXED_FILES = tuple(unified for _, unified in PATH_MAP)


def build_plan(source, destination, include_legacy=False):
    config_path = source / "phantomseal_config.json"
    default = RELATIVE_CONFIG if (source / "weights").is_dir() else OLD_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else default
    expected = config.get("expected_cloaks", 214) if isinstance(config, dict) else None
    if type(expected) is not int or expected != 214:
        raise ValueError("Source configuration must use the full 214-image cloak bank.")
    def configured(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (source / path).resolve()
    checkpoints = configured(config.get("checkpoint_dir", default["checkpoint_dir"]))
    cloaks = configured(config.get("cloak_dir", default["cloak_dir"]))
    if not cloaks.is_dir() and config.get("cloak_dir", default["cloak_dir"]) in (CLOAK_DEST, OLD_CONFIG["cloak_dir"]):
        other = OLD_CONFIG["cloak_dir"] if config.get("cloak_dir", default["cloak_dir"]) == CLOAK_DEST else CLOAK_DEST
        cloaks = source / other
    plan = [(source_file(source, old, new), new, "runtime") for old, new in PATH_MAP]
    for relative, role in (("simswap/arcface_checkpoint.tar", "ensemble_simswap"),
                           ("faceshifter/model_ir_se50.pth", "ensemble_faceshifter"),
                           ("hififace/hififace_opensource_299999.ckpt", "ensemble_hififace_id_and_3d")):
        path = checkpoints / relative
        if not path.is_file():
            path = source_file(source, OLD_CONFIG["checkpoint_dir"] + "/" + relative, "weights/" + relative)
        if not path.is_file() and role == "ensemble_simswap":
            path = source_file(source, "models/phantomseal/arcface_checkpoint.tar", RUNTIME_RECORDS["arcface"])
        plan.append((path, "weights/" + relative, role))
    plan.extend((image, CLOAK_DEST + "/" + image.relative_to(cloaks).as_posix(), "cloak_bank") for image in bank_images(cloaks))
    plan.extend((bundled_mtcnn(source, name), relative, "mtcnn_package_state") for name, relative in MTCNN_RECORDS.items())
    if include_legacy:
        legacy = source_file(source, LEGACY_FILE, LEGACY_DEST)
        if legacy.is_file():
            plan.append((legacy, LEGACY_DEST, "optional_legacy_faceparser"))
    attack = source_file(source, ATTACK_SOURCE, ATTACK_DEST)
    if attack.is_file():
        plan.append((attack, ATTACK_DEST, "optional_mtcnn_attack"))
    for path, relative, _ in plan:
        if not path.is_file() or is_link(path) or path.stat().st_size == 0:
            raise ValueError(f"Missing, empty or linked asset: {path}")
        checked_destination(destination, relative)
    return plan, checkpoints, cloaks


def import_assets(source, destination, *, overwrite=False, include_legacy=False, dry_run=False):
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if not source.is_dir() or not destination.is_dir() or is_link(destination):
        raise ValueError("Choose an existing source installation and a plain destination directory.")
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("Use migrate_weights.py for an in-place migration.")
    config_target = checked_destination(destination, "phantomseal_config.json")
    if config_target.exists() and not overwrite and json.loads(config_target.read_text(encoding="utf-8-sig")) != RELATIVE_CONFIG:
        raise FileExistsError("Destination configuration differs; review it or explicitly use --overwrite.")
    record_target = checked_destination(destination, IMPORT_MANIFEST)
    plan, checkpoints, cloaks = build_plan(source, destination, include_legacy)
    model_source = next(path for path, relative, _ in plan if relative == MODEL_MANIFEST)
    original_manifest = json.loads(model_source.read_text(encoding="utf-8-sig"))
    mtcnn = {name: {"bytes": path.stat().st_size, "sha256": sha256_file(path), "source_package": "facenet-pytorch"}
             for name, relative in MTCNN_RECORDS.items() for path, output, _ in plan if output == relative}
    manifest_payload = payload_bytes(rewrite_model_manifest(original_manifest, mtcnn))
    records, payloads, source_hashes = {}, {MODEL_MANIFEST: manifest_payload}, {}
    for path, relative, role in plan:
        before = path.stat()
        source_sha = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while hashing: {path}")
        source_hashes.setdefault(relative, source_sha)
        payload = payloads.get(relative)
        digest = hashlib.sha256(payload).hexdigest() if payload is not None else source_sha
        size = len(payload) if payload is not None else after.st_size
        if relative in records:
            if records[relative]["sha256"] != digest:
                raise ValueError(f"Source aliases disagree for {relative}")
            if str(path) != records[relative]["source_path"]:
                records[relative].setdefault("source_aliases", []).append(str(path))
            continue
        target = checked_destination(destination, relative)
        identical = target.is_file() and sha256_file(target) == digest
        if target.exists() and not identical and not overwrite:
            raise FileExistsError(f"Refusing to replace different existing asset: {target}")
        records[relative] = {"path": relative, "source_path": str(path), "role": role, "bytes": size,
                             "sha256": digest, "source_sha256": source_sha, "action": "skip" if identical else "copy"}
    for name, relative in RUNTIME_RECORDS.items():
        expected = original_manifest.get("assets", {}).get(name, {})
        if expected.get("sha256") != source_hashes[relative] or expected.get("bytes") != records[relative]["bytes"]:
            raise ValueError(f"Source asset does not match its provenance manifest: {name}")
    rows = list(records.values())
    report = {"schema_version": 2, "layout": "weights", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_installation": str(source), "source_checkpoint_dir": str(checkpoints), "source_cloak_dir": str(cloaks),
              "provenance_note": "Source paths are historical only, never runtime dependencies. SHA-256 verifies copy integrity, not independently signed upstream authenticity. Original model/data terms remain applicable.",
              "source_model_manifest": MODEL_MANIFEST, "configuration": dict(RELATIVE_CONFIG), "assets": rows,
              "copied": sum(row["action"] == "copy" for row in rows), "skipped": sum(row["action"] == "skip" for row in rows),
              "total_bytes": sum(row["bytes"] for row in rows), "dry_run": dry_run}
    if not dry_run:
        for row in rows:
            if row["action"] == "copy":
                atomic_copy(Path(row["source_path"]), checked_destination(destination, row["path"]), row["sha256"], overwrite=overwrite, payload=payloads.get(row["path"]))
        if not config_target.exists() or json.loads(config_target.read_text(encoding="utf-8-sig")) != RELATIVE_CONFIG:
            atomic_json(config_target, RELATIVE_CONFIG)
        atomic_json(record_target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace differing allowed assets/configuration")
    parser.add_argument("--include-legacy", action="store_true", help="Also copy optional legacy FaceParser.pth")
    parser.add_argument("--dry-run", action="store_true", help="Hash and validate without writing")
    args = parser.parse_args()
    report = import_assets(args.source, args.root, overwrite=args.overwrite, include_legacy=args.include_legacy, dry_run=args.dry_run)
    print(json.dumps({key: report[key] for key in ("copied", "skipped", "total_bytes", "dry_run")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"Asset import failed: {error}")
