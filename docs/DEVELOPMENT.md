# Development

FaceShield Studio is a local research application, with a browser canvas served by FastAPI and one serialized GPU worker. The default server binds to `127.0.0.1:7876`, separate from the earlier installation's 7875; an alternate checkout can use 7877. No cloud inference service is required by the integrated processing path.

## Components

| File or directory | Responsibility |
|---|---|
| `gui.py`, `app.py` | Local HTTP server, asset registry, job queue; `app.py` is a compatibility entry point |
| `web/` | Canvas comparison, image export, settings and baseline-gated SimSwap test modal |
| `protection_pipeline.py` | Input normalization, three processing modes, clean baseline and stage metadata |
| `phantomseal_runtime.py` | Cloak selection, face alignment and full-resolution PhantomSeal optimization |
| `ensemble_encoders.py` | Strict checkpoint loading and exact four-branch preprocessing wrappers |
| `phantomseal_assets.py` | Unified resource configuration and readiness checks |
| `protector.py` | Memory-conscious FaceShield adapter and standalone CLI |
| `protection_test.py` | Real SimSwap comparisons for completed editor results |
| `benchmark_chain.py`, `benchmark_alignment.py` | Paired SimSwap synthesis, alignment and FaceNet measurement |
| `benchmark_suite.py`, `benchmark_report.py` | Public manifest runner, resume verification and condition/source-aware reports |
| `download_models.py`, `download_phantomseal.py` | Public base-weight installers |
| `manage.cmd`, `manage.ps1`, `scripts/manage_checks.py` | Windows environment, model checks and owned server lifecycle |
| `scripts/import_local_assets.py` | Copy verified local runtime assets and full bank into this installation with a hash manifest |
| `scripts/weight_assets.py` | Shared resource mapping, hash and manifest helpers without model loading |
| `scripts/migrate_weights.py` | Preview or apply an in-project migration from the earlier resource layout to `weights/` |
| `third_party/phantomseal/` | Minimal inference architectures and retained source/license notices |

## Environment and commands

Use Python 3.11. `manage.cmd setup` pins CUDA 12.8 PyTorch wheels before installing `requirements-windows.txt`. The manager verifies package versions and CUDA build metadata rather than treating any installed `torch` as suitable. Existing environment files are reused when ready. Some non-core dependency ranges remain ranges; this is not a fully locked transitive environment.

```powershell
.\manage.cmd setup
.\manage.cmd models
.\manage.cmd configure -Checkpoints "weights" -Cloaks "weights\phantomseal\cloaks\mix_full"
.\manage.cmd doctor
.\manage.cmd run
.\manage.cmd status
.\manage.cmd stop
```

`setup -Python311 "C:\Path\python.exe"` selects an explicit interpreter when the Windows `py -3.11` launcher is unavailable. `-DryRun` prints the management plan without installing, writing, starting or stopping. GNU Make mirrors the same commands, for example `make run PORT=7877`; it is not needed to use the application.

For an interactive terminal server instead of the managed background process:

```powershell
.\.venv\Scripts\python.exe gui.py --host 127.0.0.1 --port 7876 --no-browser
```

Do not start both launchers on the same port. The manager records the executable, command line, PID and exact process creation time, verifies the listening process, and stops only its own identified server. Logs are written to `.cache/manage/`.

## Model and data layout

Every active downloaded model and the full cloak bank live in **`weights/`**.
The complete placement guide is [weights/README.md](../weights/README.md).
Full PhantomSeal paths are configured in `phantomseal_config.json`:

```json
{
  "checkpoint_dir": "weights",
  "cloak_dir": "weights/phantomseal/cloaks/mix_full",
  "expected_cloaks": 214
}
```

`configure` validates the resource directories and updates this configuration; it does not copy archive contents. Explicit external paths remain supported as an override, but the portable default places everything in this repository's `weights/`. The default chain requires the complete bank and all four branches. Missing files produce a readiness message, rather than an automatic fallback to the old 24-photo demo bank or a partial objective.

To migrate a complete installation after preparing the new environment, use:

```powershell
.\.venv\Scripts\python.exe scripts/import_local_assets.py --source "F:\Git\iccv25_faceshield"
.\manage.cmd doctor
```

The importer selects the minimal complete runtime files and all 214 bank images, copies them into ignored `weights/` storage, records hashes under `weights/manifests/` and writes relative configuration. It excludes private editor results, uploads, process records and the old virtual environment. It does not create symlinks or require the old tree after import. `--dry-run` previews the selection. Different existing destination files/configuration are rejected by default; `--overwrite` is an explicit replacement option. The import manifest preserves historical source paths and file hashes for provenance, rather than creating a runtime dependency on that source directory. Public source and test data are tracked separately from these large assets.

For resources already in an earlier Studio layout, stop that checkout's server and run the migration utility:

```powershell
.\manage.cmd stop
.\.venv\Scripts\python.exe scripts/migrate_weights.py
.\.venv\Scripts\python.exe scripts/migrate_weights.py --apply
.\manage.cmd doctor
```

Bare invocation and `--dry-run` only show the plan. `--apply` performs hash-checked moves within the project. The old duplicate SimSwap ArcFace alias can be removed only after the identical unified destination is verified. The utility does not move `.venv`, generated results or files outside the project.

| Location | Use |
|---|---|
| `weights/simswap/arcface_checkpoint.tar` | PhantomSeal SimSwap identity backbone |
| `weights/faceshifter/model_ir_se50.pth` | FaceShifter recognition backbone |
| `weights/hififace/hififace_opensource_299999.ckpt` | Final HifiFace identity and 3D feature-network parameters |
| `weights/phantomseal/cloaks/mix_full/*` | Full original 214-photo bank |
| `weights/facenet/facenet_vggface2.pt` | FaceNet for cloak selection and comparison scoring |
| `weights/mtcnn/pnet.pt`, `weights/mtcnn/rnet.pt`, `weights/mtcnn/onet.pt` | Mandatory detector weights for PhantomSeal and real swap tests; copied from installed facenet-pytorch by the resource tools |
| `weights/simswap/latest_net_G.pth` | SimSwap generator for real swap tests and benchmarks |
| `weights/faceshield/arcface50_checkpoint.tar`, `weights/faceshield/arcface100_checkpoint.tar` | FaceShield identity inputs; see the duplicate-weight caveat in [Validation](VALIDATION.md#faceshield-weight-caveat) |
| `weights/sd15/` | SD 1.5 UNet, VAE, text encoder, tokenizer and configuration |
| `weights/ip-adapter/models/image_encoder/` | IP-Adapter CLIP image encoder |
| `weights/ip-adapter/ip-adapter_sd15.bin` | IP-Adapter SD 1.5 parameters |
| `weights/faceparser/FaceParser.pth` | Optional legacy face-parser resource; not used by the default editor path |
| `weights/mtcnn-attack/pnet-6b6ef92b.pth` | Optional FaceShield detection resource; not required by the default editor branch |
| `weights/manifests/` | Resource download/import/migration records and observed hashes |
| `weights/cache/huggingface/` | Optional Hugging Face library download cache; managed launchers use this `HF_HOME` |

SD UNet, VAE and text encoder retain their `.fp16.safetensors` filenames for variant-aware loading. `manage.cmd models` prepares the base files and small public PhantomSeal/SimSwap components. The full author archives remain a separate download; sources and terms are listed in [Provenance](PROVENANCE.md).

## Processing and memory

The pipeline applies EXIF orientation, converts the input to RGB and normalizes dimensions once using `min(1, max_size / long_edge)`. `max_size` ranges from 128 to 1280. The original normalized PNG is saved before either defense and remains the image-quality and identity reference. There is no resize between protection stages.

PhantomSeal maintains the output-resolution perturbation and differentiably samples a fixed aligned 256 × 256 view of the largest detected face. Every optimization step uses these frozen branches:

| Branch | Preprocessing and output |
|---|---|
| SimSwap identity | ImageNet normalization, nearest-neighbor 112 px resize, L2-normalized identity features |
| FaceShifter identity | Raw RGB in `[0,1]`, bilinear 256 px view, crop `19:237`, bilinear 112 px with `align_corners=True`; backbone-normalized features |
| HifiFace identity | RGB mapped to `[-1,1]`, bilinear 112 px, L2-normalized features |
| HifiFace shape | Raw RGB in `[0,1]`, bilinear 256 px, first 80 reconstruction coefficients without L2 normalization |

The HifiFace heads come from the full final checkpoint, matching upstream's overwrite of standalone initializations. Upstream `unify` loss weights, per-channel caps and 0.003 update scale are retained. Branch gradients are computed sequentially, accumulated, and used in one sign update; this does not turn the algorithm into four independent sequential perturbation updates. The earlier SimSwap-only context loss is not added to the full `unify` objective.

Cloak selection uses a CPU FaceNet index of all detectable images in the 214-photo bank. It selects a distance nearest 1.20 among candidates above 1.04. Image, weight and preprocessing fingerprints invalidate the index when those inputs change. A cached index reduces repeated startup work; it does not replace the original bank's provenance.

FaceShield uses frozen evaluation models, sequential perturbation gradients and CUDA half precision where applicable. After reference computations it moves the VAE back to CPU and discards the text encoder. Its diffusion analysis runs on a differentiable 512 px view while the delta remains at output resolution. The default follows the original `run.sh` non-MTCNN branch. These adaptations and the 256 px face view make the local application different from a literal reproduction of either paper's evaluation protocol.

Only feature encoders needed by PhantomSeal's objective are loaded. FaceShifter/HifiFace synthesis generators, discriminators, Lightning training modules and the BFM renderer are unnecessary for those feature losses. The two defense engines release GPU models between stages. The SimSwap test generator is loaded for testing, not retained alongside a running protection job.

The default PhantomSeal RGB bounds are approximately `19.1 / 7.7 / 19.1` out of 255. FaceShield's default limit of 12 applies relative to the stage-one image. The chain can therefore reach approximately `31 / 20 / 31` relative to the clean image. Comparing the defaults is a product-settings comparison, **not a matched-distortion ablation**. Metadata records measured total changes and stage timings.

## Local API and job state

| Route | Contract |
|---|---|
| `GET /health` | Lightweight server liveness |
| `GET /api/state` | Readiness, current assets, active job, eligible test-result IDs and latest test-job IDs |
| `POST /api/upload` | JSON `{name, data}` with an image data URL; returns a registered PNG asset |
| `POST /api/process` | `{source_id, params}`; returns a queued protection job with HTTP 202 |
| `GET /api/job/{id}` | Protection or test status, progress, stage, elapsed time and final fields |
| `POST /api/test` | `{result_id, target_key, jpeg95}`; queues a real SimSwap test with HTTP 202 |
| `POST /api/compare` | Image-quality measurements for registered source/result assets |
| `GET /api/image/{id}` | Registered PNG; `/assets/{id}` is an alias |
| `GET /api/pixels/{id}` | Registered image as raw RGBA bytes for the browser's exact difference view |

`GET /api/status` and `/api/jobs/{id}` remain compatibility aliases. Errors use `{error: "..."}`. Concurrent GPU work returns HTTP 409. Asset URLs resolve only registered IDs; there is no arbitrary-path file endpoint.

Protection completion returns the clean normalized `source`, optional `stage1`, final `result` and metrics. Test jobs have `kind: "test"` and `result_id`, and never overwrite the main editor assets. Startup checks the active job kind before resuming polling. A test may continue while its modal is closed; its result belongs to the exact completed protection result, not whichever settings are currently selected.

The 0.80 baseline cutoff and per-condition visual confirmations are browser interpretation controls. The cutoff is not added to the `/api/test` request. New photos, results, test jobs or cutoff changes invalidate confirmations; cached raw test data is gated again when displayed. See [Validation](VALIDATION.md#interactive-baseline-gate).

The server accepts loopback hosts on its configured port, rejects foreign origins/cross-site requests, bounds JSON/image input, and serves a fixed set of static files. These are concrete local safeguards, not a formal security certification. Assets and job registries are in memory. Files remain under `results/editor/`, but restarting the server does not guarantee restoration of the prior browser session.

## Checks and benchmarks

```powershell
.\manage.cmd test
.\manage.cmd benchmark-help
.\manage.cmd benchmark -Sources 1 -OutputDir "results\benchmarks\quick"
```

The management test command hides CUDA, runs PowerShell lifecycle tests and Python `unittest` discovery, then runs `tests/frontend-check.cjs` if Node.js is available. It verifies dimensions/no upscale, perturbation bounds, preprocessing gradients, sequential loss equivalence, safe loader contracts, route validation, jobs, baseline failures, source weighting, benchmark resume integrity and UI state/export behavior. Many tests use dummy engines; their success does not establish defense efficacy.

The full benchmark manifest is `benchmark_manifest.json`. Protection outputs are reused only after resume integrity checks. A comparison is keyed by `(source_id, target_id, condition)`. Reports keep PNG and JPEG95 separate, report invalid/detection/weak-baseline counts, and average target-level deltas within each source before equal weighting across sources. They do not report p-values or an attack-success rate from uncalibrated cosine values.

Keep `weights/**` out of commits except its tracked placement guide `weights/README.md`. The full bank, binary models, resource manifests and Hugging Face resource cache share that ignored tree. Keep external archives, `.venv`, other caches, private photos and generated editor jobs out of commits as well. Versioned sanitized numerical evidence belongs under `docs/evidence/`; full public HTML/image bundles can be retained locally under ignored `results/benchmarks/`.
