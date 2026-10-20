"""File-only configuration/readiness for the full PhantomSeal identity ensemble."""
import json
from pathlib import Path
from asset_paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_CLOAK_DIR, MTCNN_FILES, project_path, weight_path


def configuration(root_dir):
    root = Path(root_dir).resolve()
    path = project_path(root, "phantomseal_config.json")
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    if not isinstance(config, dict):
        raise ValueError("phantomseal_config.json must contain an object.")
    expected = config.get("expected_cloaks", 214)
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("expected_cloaks must be a positive integer.")
    def resolve(value):
        return project_path(root, value)
    return {"checkpoint_dir": resolve(config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
            "cloak_dir": resolve(config.get("cloak_dir", DEFAULT_CLOAK_DIR)),
            "expected_cloaks": expected,
            "variant": "ensemble", "bank_name": "PhantomSeal VGGFace2 mixed full"}


def bank_images(folder):
    folder = Path(folder).resolve()
    return sorted(project_path(folder, path) for path in folder.rglob("*")
                  if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))


def ensemble_readiness(root_dir):
    root = Path(root_dir)
    try:
        config = configuration(root)
        checkpoint = config["checkpoint_dir"]
        simswap = checkpoint / "simswap/arcface_checkpoint.tar"
        required = [simswap,
                    weight_path(root, "facenet", "facenet_vggface2.pt"),
                    checkpoint / "hififace/hififace_opensource_299999.ckpt",
                    checkpoint / "faceshifter/model_ir_se50.pth"]
        required += [weight_path(root, "mtcnn", name) for name in MTCNN_FILES]
        required = [project_path(root, path) for path in required]
        missing = [path.name for path in required if not path.is_file()]
        for source in ("ensemble_encoders.py", "third_party/phantomseal/full/faceshifter.py",
                       "third_party/phantomseal/full/iresnet.py", "third_party/phantomseal/full/deep3d.py"):
            if not (root / source).is_file():
                missing.append(source)
        images = bank_images(config["cloak_dir"])
        if len(images) < config["expected_cloaks"]:
            missing.append(f"full cloak bank ({len(images)}/{config['expected_cloaks']} images)")
        return {"ready": not missing,
                "message": "Missing: " + ", ".join(missing) if missing else f"Ready · PhantomSeal ensemble · {len(images)} cloak faces",
                "variant": "ensemble", "branches": ["simswap", "hififace_3d", "hififace_id", "faceshifter"],
                "cloak_count": len(images), "cloak_bank": config["bank_name"]}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"ready": False, "message": f"PhantomSeal configuration error: {error}", "variant": "ensemble"}
