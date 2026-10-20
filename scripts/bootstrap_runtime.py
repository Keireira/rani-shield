"""Copy an installed CPython 3.11 runtime into the project without its packages.

The source installation is needed only during setup. Subsequent execution uses
.runtime/python and .venv inside the project. This helper never downloads files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import stat
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.weight_assets import atomic_json, checked_destination, is_link, sha256_file


RUNTIME_DIRECTORIES = ("DLLs", "Lib", "include", "libs", "tcl")
RUNTIME_FILES = ("python.exe", "pythonw.exe", "python3.dll", "python311.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt")
REQUIRED_FILES = ("python.exe", "python311.dll", "LICENSE.txt", "Lib/os.py", "Lib/venv/__init__.py")


def source_files(source):
    """Select CPython files, excluding unrelated installed third-party packages."""
    selected = []
    for name in RUNTIME_FILES:
        path = source / name
        if path.is_file():
            selected.append(path)
    for directory in RUNTIME_DIRECTORIES:
        top = source / directory
        if not top.exists():
            continue
        if is_link(top):
            raise ValueError(f"Runtime source cannot contain links: {top}")
        # os.walk prunes site-packages before entering it, even on large installs.
        import os
        for current, directories, files in os.walk(top, followlinks=False):
            current = Path(current)
            directories[:] = [name for name in directories if name not in ("site-packages", "__pycache__")]
            for name in directories:
                if is_link(current / name):
                    raise ValueError(f"Runtime source cannot contain links: {current / name}")
            selected.extend(current / name for name in files if not name.endswith((".pyc", ".pyo")))
    for path in selected:
        if is_link(path) or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"Runtime source must be regular files: {path}")
    return sorted(selected)


def install_python(source, root):
    root, source = Path(root).resolve(), Path(source).resolve()
    destination = checked_destination(root, ".runtime/python")
    if source == destination:
        for name in REQUIRED_FILES:
            if not (destination / name).is_file():
                raise ValueError(f"Incomplete local Python runtime: {name}")
        return {"ready": True, "copied": 0, "runtime": str(destination)}
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError("Expected an installed CPython 3.11 directory; missing: " + ", ".join(missing))
    plan = []
    for path in source_files(source):
        relative = path.relative_to(source)
        target = checked_destination(root, Path(".runtime/python") / relative)
        digest = sha256_file(path)
        if target.exists() and (not target.is_file() or sha256_file(target) != digest):
            raise FileExistsError(f"Local runtime differs; refusing to overwrite: {target}")
        plan.append((path, target, digest))
    copied = 0
    for path, target, digest in plan:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
        if sha256_file(target) != digest:
            raise ValueError(f"Runtime copy verification failed: {target}")
    report = {
        "schema_version": 1,
        "source_installation": str(source),
        "provenance_note": "Source path is historical only. Runtime files are independent local copies.",
        "files": [{"path": target.relative_to(root).as_posix(), "sha256": digest, "bytes": target.stat().st_size}
                  for _, target, digest in plan],
    }
    atomic_json(checked_destination(root, ".runtime/python/manifest.json"), report)
    return {"ready": True, "copied": copied, "runtime": str(destination)}


def install_node(source, root, license_path=None):
    """Optionally vendor Node for frontend checks, retaining its license."""
    source, root = Path(source).resolve(), Path(root).resolve()
    destination = checked_destination(root, ".runtime/node/node.exe")
    if source == destination:
        return {"ready": True, "copied": 0}
    license_path = Path(license_path).resolve() if license_path else next(
        (source.parent / name for name in ("LICENSE", "LICENSE.txt")
         if (source.parent / name).is_file()), None)
    if not source.is_file() or is_link(source) or license_path is None or not license_path.is_file() or is_link(license_path):
        raise ValueError("Node source must provide node.exe and its adjacent LICENSE or LICENSE.txt.")
    planned = [(source, destination), (license_path, checked_destination(root, ".runtime/node/LICENSE"))]
    for old, new in planned:
        if new.exists() and sha256_file(old) != sha256_file(new):
            raise FileExistsError(f"Local Node runtime differs; refusing to overwrite: {new}")
    copied = 0
    for old, new in planned:
        new.parent.mkdir(parents=True, exist_ok=True)
        if not new.exists():
            shutil.copy2(old, new)
            copied += 1
        if sha256_file(old) != sha256_file(new):
            raise ValueError(f"Node copy verification failed: {new}")
    return {"ready": True, "copied": copied}


def refresh_scripts(root):
    """Rebase console launchers after moving a project while preserving packages."""
    from importlib.metadata import distributions
    from pip._vendor.distlib.scripts import ScriptMaker
    root = Path(root).resolve()
    executable = checked_destination(root, ".venv/Scripts/python.exe")
    if Path(sys.executable).resolve() != executable:
        raise ValueError("Refresh entry points with this project's .venv Python.")
    scripts = checked_destination(root, ".venv/Scripts")
    package_dir = checked_destination(root, ".venv/Lib/site-packages")
    maker = ScriptMaker(None, str(scripts))
    maker.executable = str(executable)
    maker.clobber = True
    count = 0
    for distribution in distributions(path=[str(package_dir)]):
        for entry in distribution.entry_points:
            if entry.group not in ("console_scripts", "gui_scripts"):
                continue
            maker.variants = {"", "X", "X.Y"} if entry.name == "pip" else {""}
            maker.make(f"{entry.name} = {entry.value}", options={"gui": entry.group == "gui_scripts"})
            count += 1
    return {"refreshed": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--source", type=Path, help="Installed CPython 3.11 base directory")
    parser.add_argument("--node-source", type=Path, help="Optional Node executable with adjacent license")
    parser.add_argument("--node-license", type=Path, help="Explicit license file for this Node distribution")
    parser.add_argument("--refresh-scripts", action="store_true", help="Rebase installed entry points using local .venv Python")
    args = parser.parse_args()
    if args.source is None and args.node_source is None and not args.refresh_scripts:
        parser.error("Provide --source, --node-source or --refresh-scripts")
    result = {}
    if args.source is not None:
        result["python"] = install_python(args.source, args.root)
    if args.node_source is not None:
        result["node"] = install_node(args.node_source, args.root, args.node_license)
    if args.refresh_scripts:
        result["scripts"] = refresh_scripts(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Runtime setup failed: {error}")
