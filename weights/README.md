# Local weights and resources

All active model files, the complete PhantomSeal cloak bank, resource manifests and optional Hugging Face download cache belong under this **`weights/`** directory. Only this README is tracked in Git. Python source files remain in their code directories; moving weights does not move or rename the network architectures.

## What goes where

Paths below are relative to the repository root.

| Path | Required for | Contents or source |
|---|---|---|
| `weights/faceshield/arcface50_checkpoint.tar` | FaceShield | ArcFace entry from the official FaceShield model archive |
| `weights/faceshield/arcface100_checkpoint.tar` | FaceShield | Second archive entry; the local release copies were byte-identical, so two independent networks are not established |
| `weights/simswap/arcface_checkpoint.tar` | PhantomSeal and real SimSwap tests | Official SimSwap ArcFace identity backbone |
| `weights/simswap/latest_net_G.pth` | Real SimSwap tests and benchmarks | Official 224 px SimSwap generator; not needed for the feature-only PhantomSeal optimization objective |
| `weights/facenet/facenet_vggface2.pt` | PhantomSeal cloak selection and swap scoring | Official facenet-pytorch VGGFace2 release |
| `weights/mtcnn/pnet.pt`, `weights/mtcnn/rnet.pt`, `weights/mtcnn/onet.pt` | PhantomSeal and real swap tests | All three facenet-pytorch detector states; prepared centrally by the resource tools |
| `weights/faceshifter/model_ir_se50.pth` | Full PhantomSeal objective | Recognition checkpoint from the full PhantomSeal checkpoint archive |
| `weights/hififace/hififace_opensource_299999.ckpt` | Full PhantomSeal objective | Final HifiFace checkpoint; the adapter extracts its identity and 3D feature heads |
| `weights/sd15/unet/config.json` and `diffusion_pytorch_model.fp16.safetensors` | FaceShield | SD 1.5 UNet |
| `weights/sd15/vae/config.json` and `diffusion_pytorch_model.fp16.safetensors` | FaceShield | SD 1.5 VAE |
| `weights/sd15/text_encoder/config.json` and `model.fp16.safetensors` | FaceShield | SD 1.5 text encoder |
| `weights/sd15/tokenizer/` | FaceShield | `vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json` |
| `weights/ip-adapter/ip-adapter_sd15.bin` | FaceShield | Released SD 1.5 IP-Adapter parameters |
| `weights/ip-adapter/models/image_encoder/config.json` and `model.safetensors` | FaceShield | Released IP-Adapter CLIP image encoder |
| `weights/faceparser/FaceParser.pth` | Optional legacy path | Face parser; not required by the default editor's FaceShield branch |
| `weights/mtcnn-attack/pnet-6b6ef92b.pth` | Optional FaceShield detector path | Separate detector resource; the default editor branch does not require it |
| `weights/phantomseal/cloaks/mix_full/` | Full PhantomSeal objective | **All 214** original mixed-bank photos from the official data archive, retaining their filenames |
| `weights/manifests/*.json` | Local provenance | Download, import and migration source records, sizes and observed hashes |
| `weights/cache/huggingface/` | Optional library cache | `HF_HOME` used by managed launchers; separate from the explicit `weights/sd15/` and `weights/ip-adapter/` model directories |

Keep the `.fp16.safetensors` names intact. The loaders request that variant. Do not put another `checkpoints/` level under `weights/`: the default ensemble paths are `weights/simswap/...`, `weights/faceshifter/...` and `weights/hififace/...`.

The default `phantomseal_config.json` is:

```json
{
  "checkpoint_dir": "weights",
  "cloak_dir": "weights/phantomseal/cloaks/mix_full",
  "expected_cloaks": 214
}
```

## Prepare a fresh installation

```powershell
.\manage.cmd setup
.\manage.cmd models
```

`models` downloads the public FaceShield base files, SD 1.5, IP-Adapter and smaller SimSwap/ArcFace/FaceNet releases into the unified layout. It also copies the three detector states bundled with installed facenet-pytorch into `weights/mtcnn/`; active processing loads those central files. It **does not fetch the full PhantomSeal checkpoint and data archives**.

Obtain the full archives from PhantomSeal's [official release folder](https://drive.google.com/drive/folders/1caHioBnA1478FR15W3JzNuHu36zxopJv): [data archive](https://drive.google.com/file/d/1feXBOP0WGemPMpsjeNLKe1j0l1A3Cxjj/view) and [checkpoint archive](https://drive.google.com/file/d/1p-_w1OMEYtph7R3yI188mE-o-xmJ6ya_/view). Extract them to a temporary or external location, then copy only the required resources into the table's locations:

- `checkpoints/faceshifter/model_ir_se50.pth` → `weights/faceshifter/model_ir_se50.pth`.
- `checkpoints/hififace/hififace_opensource_299999.ckpt` → `weights/hififace/hififace_opensource_299999.ckpt`.
- `data/vggface2_cloak/mix_full/` → `weights/phantomseal/cloaks/mix_full/`, with all 214 photos.

The smaller downloader supplies the SimSwap checkpoint and generator. Keep an archived SimSwap identity checkpoint only if it matches the intended release; the runtime has a single canonical `weights/simswap/arcface_checkpoint.tar` path. The large archives contain unrelated synthesis/training models that this encoder-only objective does not load.

```powershell
.\manage.cmd configure -Checkpoints "weights" -Cloaks "weights\phantomseal\cloaks\mix_full"
.\manage.cmd doctor
```

`configure` records paths and does not copy files. Explicit external resource overrides are supported, but the default self-contained installation uses `weights/`. `doctor` checks environment and resource readiness without loading the model tensors or proving protection efficacy.

## Import an existing complete installation

After `setup`, preview or run the local importer:

```powershell
.\.venv\Scripts\python.exe scripts/import_local_assets.py --source "F:\Git\iccv25_faceshield" --dry-run
.\.venv\Scripts\python.exe scripts/import_local_assets.py --source "F:\Git\iccv25_faceshield"
.\manage.cmd doctor
```

The importer copies selected model files and the full bank, hashes the copied resources, and writes relative configuration. The detector states come from central source files or the old environment's fixed bundled detector files; the package originals are retained. It excludes private uploads, generated editor jobs, process records and the source virtual environment itself. Different existing files/configuration are rejected by default; review them before explicitly using `--overwrite`. Optional legacy parser import uses `--include-legacy`.

## Migrate the previous Studio resource layout

Stop only this checkout's managed server before changing resource paths:

```powershell
.\manage.cmd stop
.\.venv\Scripts\python.exe scripts/migrate_weights.py
.\.venv\Scripts\python.exe scripts/migrate_weights.py --apply
.\manage.cmd doctor
.\manage.cmd run
```

Bare invocation and `--dry-run` only show a plan. `--apply` performs the hash-checked in-project moves and writes the migration record. An obsolete duplicate SimSwap ArcFace file is removed only after verifying an identical canonical destination. The utility does not move `.venv`, generated results or outside-project files. Historical numerical evidence inventories remain unchanged; their old paths describe the earlier layout, while the migration manifest records the relocation.

## Terms and interpretation

These are released pretrained resources, not newly trained checkpoints. Their code, model and dataset terms remain applicable. Source URLs, retained notices and the duplicate FaceShield ArcFace caveat are documented in [Provenance](../docs/PROVENANCE.md) and [Validation](../docs/VALIDATION.md). Hashes verify local integrity; they are not publisher signatures. Neither complete model readiness nor a generated output establishes successful face-swap protection.
