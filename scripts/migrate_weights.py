"""Plan an in-project migration to weights/. Use --apply to perform it.

Only known model assets and model caches are considered. All paths, hashes and
destination conflicts are checked before any mutation. No downloads or models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.weight_assets import (ATTACK_DEST, ATTACK_SOURCE, CLOAK_DEST, IMPORT_MANIFEST, LEGACY_DEST,
    LEGACY_FILE, MODEL_MANIFEST, MTCNN_RECORDS, OLD_CONFIG, PATH_MAP, RELATIVE_CONFIG,
    atomic_copy, atomic_json, bank_images, bundled_mtcnn, checked_destination, is_link,
    local_path, payload_bytes, rewrite_import_manifest, rewrite_model_manifest, sha256_file)


def migrate(root, *, apply=False):
    raw_root = Path(root).expanduser()
    if not raw_root.is_dir() or is_link(raw_root):
        raise ValueError("Migration needs a plain project directory.")
    root = raw_root.resolve()
    config_path = checked_destination(root, "phantomseal_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else OLD_CONFIG
    if not isinstance(config, dict) or type(config.get("expected_cloaks", 214)) is not int or config.get("expected_cloaks", 214) != 214:
        raise ValueError("Expected a full 214-image cloak-bank configuration.")
    # No configured outside-installation path can be read, moved or erased.
    checkpoint = local_path(root, config.get("checkpoint_dir", OLD_CONFIG["checkpoint_dir"]))
    cloak = local_path(root, config.get("cloak_dir", OLD_CONFIG["cloak_dir"]))
    entries = []
    def add(source, relative, *, preserve_source=False, payload=None):
        source, target = local_path(root, source), checked_destination(root, relative)
        if not source.is_file():
            return
        if source.stat().st_size == 0:
            raise ValueError(f"Empty source asset: {source}")
        if any(row["source"] == source and row["target"] == target for row in entries):
            return
        entries.append({"source": source, "target": target, "relative": relative, "preserve_source": preserve_source, "payload": payload})

    for old, new in PATH_MAP:
        if new != MODEL_MANIFEST:
            add(root / old, new)
    add(root / LEGACY_FILE, LEGACY_DEST)
    add(root / ATTACK_SOURCE, ATTACK_DEST)
    add(root / ".cache/models/arcface-models.zip", "weights/downloads/arcface-models.zip")
    for base in dict.fromkeys((checkpoint, root / OLD_CONFIG["checkpoint_dir"])):
        for relative in ("simswap/arcface_checkpoint.tar", "faceshifter/model_ir_se50.pth", "hififace/hififace_opensource_299999.ckpt"):
            add(base / relative, "weights/" + relative)
    # A stopped migration may have already moved part of the bank. Validate the
    # union before resuming; per-name hashes are checked with every other alias.
    bank_names = set()
    for bank in dict.fromkeys((cloak, root / OLD_CONFIG["cloak_dir"], root / CLOAK_DEST)):
        if bank.is_dir():
            local_path(root, bank)
            for image in sorted(bank.rglob("*")):
                local_path(root, image)
                if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    name = image.relative_to(bank).as_posix()
                    bank_names.add(name)
                    add(image, CLOAK_DEST + "/" + name)
    if len(bank_names) != 214:
        raise ValueError(f"Expected exactly 214 cloak images across old/new banks, found {len(bank_names)}.")
    for name, relative in MTCNN_RECORDS.items():
        # Exactly three package states are copied, never moved out of .venv.
        resource = bundled_mtcnn(root, name, installed_fallback=False)
        add(resource, relative, preserve_source=True)
    add(root / ".cache/phantomseal/full-cloak-index.json", "weights/cache/phantomseal/full-cloak-index.json")
    hf = local_path(root, ".cache/huggingface")
    if hf.is_dir():
        for path in sorted(hf.rglob("*")):
            local_path(root, path)
            if path.is_file():
                add(path, "weights/cache/huggingface/" + path.relative_to(hf).as_posix())

    mtcnn = {}
    for name, relative in MTCNN_RECORDS.items():
        path = bundled_mtcnn(root, name, installed_fallback=False)
        mtcnn[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "source_package": "facenet-pytorch"}
    manifests = [root / "models/phantomseal/manifest.json", root / MODEL_MANIFEST]
    for path in manifests:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            add(path, MODEL_MANIFEST, payload=payload_bytes(rewrite_model_manifest(value, mtcnn)))

    groups = {}
    def preflight(row):
        source = row["source"]
        before = source.stat()
        row["source_sha256"] = sha256_file(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while hashing: {source}")
        row["sha256"] = hashlib.sha256(row["payload"]).hexdigest() if row["payload"] is not None else row["source_sha256"]
        row["bytes"] = len(row["payload"]) if row["payload"] is not None else after.st_size
        group = groups.setdefault(row["relative"], [])
        if group and group[0]["sha256"] != row["sha256"]:
            raise FileExistsError(f"Different source aliases for {row['relative']}")
        target = row["target"]
        if target.exists():
            if not target.is_file():
                raise FileExistsError(f"Destination is not a file: {target}")
            current_sha = sha256_file(target)
            # A manifest already at its final name may need only its path metadata rewritten.
            transforming_in_place = source == target and row["payload"] is not None
            if current_sha != row["sha256"] and not transforming_in_place:
                raise FileExistsError(f"Different existing destination: {target}")
        group.append(row)
    for row in entries:
        preflight(row)

    current = {relative: {"path": relative, "bytes": rows[0]["bytes"], "sha256": rows[0]["sha256"],
                         "source_path": str(rows[0]["source"])} for relative, rows in groups.items()}
    imported_sources = (root / ".cache/imported-assets.json", root / IMPORT_MANIFEST)
    for path in imported_sources:
        if path.is_file():
            value = rewrite_import_manifest(json.loads(path.read_text(encoding="utf-8-sig")), current)
            before_count = len(entries)
            add(path, IMPORT_MANIFEST, payload=payload_bytes(value))
            if len(entries) > before_count:
                preflight(entries[-1])

    report_rows = []
    for row in entries:
        already = row["source"] == row["target"] and row["source_sha256"] == row["sha256"]
        action = ("already_unified" if already else "update_manifest" if row["source"] == row["target"] and row["payload"] is not None
                  else "copy_package_state" if row["preserve_source"] else "move_or_deduplicate")
        report_rows.append({"source": row["source"].relative_to(root).as_posix(), "path": row["relative"],
                            "bytes": row["bytes"], "sha256": row["sha256"], "source_sha256": row["source_sha256"], "action": action})
    report = {"schema_version": 1, "layout": "weights", "apply": apply,
              "created_utc": datetime.now(timezone.utc).isoformat(), "assets": report_rows,
              "configuration": dict(RELATIVE_CONFIG), "note": "Only project-local allowlisted assets are moved. Package MTCNN originals are preserved. All conflicts were checked before mutation."}
    if not apply:
        return report
    for relative, rows in groups.items():
        target = rows[0]["target"]
        # Recheck all aliases before changing the first name in this group.
        for row in rows:
            if sha256_file(row["source"]) != row["source_sha256"]:
                raise ValueError(f"Source changed after preflight: {row['source']}")
        desired_sha = rows[0]["sha256"]
        if not target.exists() or sha256_file(target) != desired_sha:
            first = rows[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            if first["payload"] is not None:
                atomic_copy(first["source"], target, desired_sha, payload=first["payload"], overwrite=first["source"] == target)
            elif first["preserve_source"]:
                atomic_copy(first["source"], target, desired_sha)
            else:
                if target.exists():
                    raise FileExistsError(f"Destination changed after preflight: {target}")
                os.rename(first["source"], target)
        if sha256_file(target) != desired_sha:
            raise ValueError(f"Destination hash verification failed: {target}")
        for row in rows:
            source = row["source"]
            if source != target and source.exists() and not row["preserve_source"]:
                # Removal is limited to the exact alias whose preflight hash still matches.
                if sha256_file(source) != row["source_sha256"]:
                    raise ValueError(f"Refusing to remove a changed alias: {source}")
                source.unlink()
    if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8-sig")) != RELATIVE_CONFIG:
        atomic_json(config_path, RELATIVE_CONFIG)
    atomic_json(checked_destination(root, "weights/manifests/migration.json"), report)
    # Remove only empty old asset directories, with no recursive deletion.
    parents = {parent for row in entries for parent in row["source"].parents
               if parent != root and parent.is_relative_to(root) and not parent.is_relative_to(root / "weights") and not parent.is_relative_to(root / ".venv")}
    for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            pass
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="Perform the fully preflighted in-project moves")
    action.add_argument("--dry-run", action="store_true", help="Plan only; this is the default")
    args = parser.parse_args()
    report = migrate(args.root, apply=args.apply)
    counts = {}
    for row in report["assets"]:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    print(json.dumps({"apply": args.apply, "actions": counts}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"Weight migration failed: {error}")
