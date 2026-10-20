"""Small metadata/file preflights for manage.ps1; never loads model weights."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import tempfile


def environment(root, version=None, version_info=None):
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    get_version = version or metadata.version
    python = version_info or sys.version_info
    errors = []
    torch_ready = True
    for name, expected in (("torch", "2.7.1+cu128"), ("torchvision", "0.22.1+cu128")):
        try:
            actual = get_version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            torch_ready = False
            errors.append(f"{name}: {actual}; expected {expected}")
    requirements_ready = True
    for line in (Path(root) / "requirements-windows.txt").read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            actual = get_version(requirement.name)
        except metadata.PackageNotFoundError:
            actual = None
        if actual is None or (requirement.specifier and actual not in requirement.specifier):
            requirements_ready = False
            errors.append(f"{requirement.name}: {actual or 'missing'}; required {requirement.specifier}")
    python_ready = tuple(python[:2]) == (3, 11)
    if not python_ready:
        errors.append(f"Python {python[0]}.{python[1]}; expected 3.11")
    return dict(ready=python_ready and torch_ready and requirements_ready,
                python_ready=python_ready, torch_ready=torch_ready,
                requirements_ready=requirements_ready, errors=errors)


def configure(root, checkpoints, cloaks):
    root, checkpoints, cloaks = (Path(p).expanduser().resolve() for p in (root, checkpoints, cloaks))
    if not checkpoints.is_dir() or not cloaks.is_dir():
        raise ValueError("Choose the extracted checkpoints and data/vggface2_cloak/mix_full directories.")
    required = [checkpoints / "hififace/hififace_opensource_299999.ckpt",
                checkpoints / "faceshifter/model_ir_se50.pth"]
    simswap = checkpoints / "simswap/arcface_checkpoint.tar"
    if not simswap.is_file():
        simswap = root / "weights/simswap/arcface_checkpoint.tar"
    missing = [str(p) for p in [*required, simswap] if not p.is_file() or p.stat().st_size == 0]
    images = [p for p in cloaks.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if missing:
        raise ValueError("Missing ensemble checkpoints: " + ", ".join(missing))
    if len(images) < 214:
        raise ValueError(f"The full mix_full bank needs at least 214 images; found {len(images)}.")
    config = dict(checkpoint_dir=str(checkpoints), cloak_dir=str(cloaks), expected_cloaks=214)
    target = root / "phantomseal_config.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, prefix=".phantomseal-config-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(config, output, indent=2)
            output.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    print(f"Configured {target}\nCheckpoints: {checkpoints}\nCloak bank: {cloaks} ({len(images)} images)")


def doctor(root):
    root = Path(root).resolve()
    print(f"Model storage: {root / 'weights'}")
    report = environment(root)
    good = report["ready"]
    print(f"{'OK' if good else 'FAIL'} Environment: Python 3.11, torch 2.7.1+cu128, torchvision 0.22.1+cu128")
    for error in report["errors"]:
        print(f"  {error}")
    sys.path.insert(0, str(root))
    try:
        from protection_pipeline import readiness
        engines = readiness(root)
        for name, result in engines.items():
            message = result['message'].replace('\u00b7', '|')
            print(f"{'OK' if result['ready'] else 'FAIL'} {name}: {message}")
            good = good and result["ready"]
        phantom = engines["phantomseal"]
        branches = {"simswap", "faceshifter", "hififace_id", "hififace_3d"}
        complete = set(phantom.get("branches", [])) == branches and phantom.get("cloak_count", 0) >= 214
        print(f"{'OK' if complete else 'FAIL'} PhantomSeal: all 4 branches; {phantom.get('cloak_count', 0)}/214 cloak images")
        good = good and complete
    except Exception as error:
        good = False
        print(f"FAIL Model files/configuration: {error}")
    try:
        import torch
        cuda = torch.cuda.is_available()
        details = f"torch CUDA {torch.version.cuda}"
        if cuda:
            details += f", {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}"
        print(f"{'OK' if cuda else 'FAIL'} CUDA: {details}")
        good = good and cuda and torch.version.cuda == "12.8"
    except Exception as error:
        good = False
        print(f"FAIL CUDA: {error}")
    print("No model weights were loaded. File readiness does not measure defense strength.")
    return 0 if good else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("environment", "configure", "doctor"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--cloaks", type=Path)
    args = parser.parse_args()
    if args.command == "environment":
        print(json.dumps(environment(args.root)))
        return 0
    if args.command == "doctor":
        return doctor(args.root)
    if args.checkpoints is None or args.cloaks is None:
        parser.error("configure needs --checkpoints and --cloaks")
    configure(args.root, args.checkpoints, args.cloaks)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
