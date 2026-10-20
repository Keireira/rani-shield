# Development

Rani Shield is a local research application, with a browser canvas served by FastAPI and one serialized GPU worker. The default server binds to `127.0.0.1:7876`; an alternate instance can use 7877. No cloud inference service is required by the integrated processing path.

## Components

| File or directory | Responsibility |
|---|---|
| `gui.py`, `app.py` | Local HTTP server, asset registry, job queue; `app.py` is a compatibility entry point |
| `web/` | Canvas comparison, image export, settings and baseline-gated selectable face-swap test modal |
| `protection_pipeline.py` | Input normalization, four processing modes, clean baseline and stage metadata |
| `runtime_contracts.py`, `image_io.py` | Shared parameter validation, output dimensions, image decoding and PNG/JPEG95 transformations |
| `phantomseal_runtime.py` | Cloak selection, face alignment and full-resolution PhantomSeal optimization |
| `ensemble_encoders.py` | Strict checkpoint loading and exact four-branch preprocessing wrappers |
| `asset_paths.py`, `phantomseal_assets.py` | Project-contained resource paths, configuration and readiness checks |
| `protector.py` | Memory-conscious FaceShield adapter and standalone CLI |
| `nullswap_runtime.py` | Standalone labelled NullSwap reconstruction; bounded native residual mapped to full output |
| `swap_adapters/`, `swap_evaluation.py`, `face_evaluation.py` | Lazy swapper registry, native model adapters, independent face measurement and detection gates |
| `protection_test.py` | Shared PNG/JPEG95 orchestration for four real swap backends |
| `benchmark_chain.py`, `benchmark_alignment.py` | Paired SimSwap synthesis, alignment and FaceNet measurement |
| `benchmark_suite.py`, `benchmark_report.py` | Public manifest runner, resume verification and condition/source-aware reports |
| `download_models.py`, `download_phantomseal.py` | Public base-weight installers |
| `manage.cmd`, `manage.ps1`, `scripts/manage_checks.py` | Windows environment, model checks, foreground server launch and legacy process cleanup |
| `scripts/bootstrap_runtime.py` | Copy and verify the project-local CPython/Node runtimes, retaining their notices |
| `scripts/runtime_environment.json`, `scripts/runtime_environment.py` | Shared local cache/temp environment contract for Python and management scripts |
| `scripts/smoke_check.py` | Offline GPU execution probe with model/Python file provenance; no editor server |
| `scripts/import_swap_assets.py` | Explicit optional archive import; streaming hash verification, safe paths and pinned InfoSwap readout downloads |
| `scripts/import_local_assets.py` | Copy verified local runtime assets and full bank into this installation with a hash manifest |
| `scripts/weight_assets.py` | Shared resource mapping, hash and manifest helpers without model loading |
| `scripts/migrate_weights.py` | Preview or apply an in-project migration from the earlier resource layout to `weights/` |
| `third_party/phantomseal/` | Minimal inference architectures and retained source/license notices |

## Environment and commands

`manage.cmd setup` prepares a project-local Python 3.11 installation at `.runtime/python/` and a `.venv` that uses it. The Python executable, standard library and DLLs are inside the repository, alongside the packages. Bootstrap may copy an installed Python 3.11 once; the original installation is not needed after preparation. Node.js can similarly be copied to `.runtime/node/` for frontend checks.

The manager pins CUDA 12.8 PyTorch wheels before installing `requirements-windows.txt` and verifies package versions/build metadata. Ready package installations are reused. Some non-core dependency ranges remain ranges; this is not a fully locked transitive environment. Windows and the NVIDIA driver remain host requirements; management uses Windows PowerShell.

```powershell
.\manage.cmd setup
.\manage.cmd models
.\manage.cmd configure -Checkpoints "weights" -Cloaks "weights\phantomseal\cloaks\mix_full"
.\manage.cmd doctor
.\manage.cmd run
```

`setup -Python311 "C:\Path\python.exe"` selects an explicit bootstrap interpreter when automatic discovery is unavailable. It is a setup source, not a runtime path. `-DryRun` prints the management plan without installing, writing, starting or stopping. GNU Make mirrors the same commands, for example `make run PORT=7877`; it is not needed to use the application. A bare `make` prints help. Checks and setup do not start the editor.

`manage.cmd run` and `make run` invoke the server synchronously in the current terminal, with live stdout/stderr. They return when the server exits; **Ctrl+C** stops it. They create no background server, PID record or server log files. `launch.ps1` delegates to `manage.ps1 run`, so both launch paths use the same validation and foreground behavior without opening a browser.

Inspect port occupancy from another terminal:

```powershell
.\manage.cmd status
```

`run` rejects an occupied port or an active recorded legacy server without starting, stopping or reusing a process. `stop` is retained only for older background launches with a verified saved executable, command line, PID and exact process creation time. It does not control foreground servers; use **Ctrl+C** in their terminal.

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

`configure` validates the resource directories and records paths relative to the project; it does not copy archive contents. Runtime resources must resolve inside the project. External overrides and links escaping the root are rejected. The default chain requires the complete bank and all four branches. Missing files produce a readiness message, rather than an automatic fallback to the old 24-photo demo bank or a partial objective.

To migrate a complete installation after preparing the new environment, use:

```powershell
.\.venv\Scripts\python.exe scripts/import_local_assets.py --source "D:\path\to\old-installation"
.\manage.cmd doctor
```

The importer selects the minimal complete runtime files and all 214 bank images, copies them into ignored `weights/` storage, records hashes under `weights/manifests/` and writes relative configuration. It excludes private editor results, uploads, process records and the old virtual environment. It does not create symlinks or require the old tree after import. `--dry-run` previews the selection. Different existing destination files/configuration are rejected by default; `--overwrite` is an explicit replacement option. The import manifest preserves historical source paths and file hashes for provenance, rather than creating a runtime dependency on that source directory. Public source and test data are tracked separately from these large assets.

For resources already in an earlier local layout, stop that checkout's server with **Ctrl+C** first. If it was launched in the background by an older version, use that checkout's `manage.cmd stop` to stop the verified recorded process. Then run the migration utility:

```powershell
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
| `weights/faceparser/FaceParser.pth` | Optional historical resource; the unused face-parser code is outside the active application |
| `weights/mtcnn-attack/pnet-6b6ef92b.pth` | Optional FaceShield detection resource; not required by the default editor branch |
| `weights/manifests/` | Resource download/import/migration records and observed hashes |
| `weights/cache/huggingface/` | Optional Hugging Face library download cache; managed launchers use this `HF_HOME` |

SD UNet, VAE and text encoder retain their `.fp16.safetensors` filenames for variant-aware loading. `manage.cmd models` prepares the base files and small public PhantomSeal/SimSwap components. The full author archives remain a separate download; sources and terms are listed in [Provenance](PROVENANCE.md).

The integrated runtime uses explicit local weight paths. Resource caches belong under `weights/cache/` and application temporary files under `.cache/tmp/`. Import manifests may contain the old source location as a historical record, but the loader never uses that location. A Git clone alone excludes `.runtime/`, `.venv/`, weights and generated assets; run setup and prepare the resources before using it.

## Processing and memory

The pipeline applies EXIF orientation, converts the input to RGB and normalizes dimensions once using `min(1, max_size / long_edge)`. `max_size` ranges from 128 to 1280. The original normalized PNG is saved before the first defense and remains the image-quality and identity reference. There is no resize between protection stages.

The editor submits `mode: "custom"` with an ordered `stages` recipe of 1–8 entries. Repeated types are supported; each entry owns its parameters and a distinct output directory. Every stage is validated before work begins, runs against the previous output, and releases its model before the next stage loads. Geometry is checked after every stage. Progress describes the stage number and filter. The legacy fixed modes and flat parameters remain accepted for older clients and benchmarks.

PhantomSeal maintains the output-resolution perturbation and differentiably samples a fixed aligned 256 × 256 view of the largest detected face. Every optimization step uses these frozen branches:

| Branch | Preprocessing and output |
|---|---|
| SimSwap identity | ImageNet normalization, nearest-neighbor 112 px resize, L2-normalized identity features |
| FaceShifter identity | Raw RGB in `[0,1]`, bilinear 256 px view, crop `19:237`, bilinear 112 px with `align_corners=True`; backbone-normalized features |
| HifiFace identity | RGB mapped to `[-1,1]`, bilinear 112 px, L2-normalized features |
| HifiFace shape | Raw RGB in `[0,1]`, bilinear 256 px, first 80 reconstruction coefficients without L2 normalization |

The HifiFace heads come from the full final checkpoint, matching upstream's overwrite of standalone initializations. Upstream `unify` loss weights, per-channel caps and 0.003 update scale are retained. Branch gradients are computed sequentially, accumulated, and used in one sign update; this does not turn the algorithm into four independent sequential perturbation updates. The earlier SimSwap-only context loss is not added to the full `unify` objective.

Cloak selection uses a CPU FaceNet index of all detectable images in the 214-photo bank. It selects a distance nearest 1.20 among candidates above 1.04. Image, weight and preprocessing fingerprints invalidate the index when those inputs change. A cached index reduces repeated startup work; it does not replace the original bank's provenance.

FaceShield uses frozen evaluation models, sequential perturbation gradients and CUDA half precision where applicable. After reference computations it moves the VAE back to CPU and discards the text encoder. Its diffusion analysis runs on a differentiable 512 px view while the delta remains at output resolution. The default follows the original upstream non-MTCNN configuration. These adaptations and the 256 px face view make the local application different from a literal reproduction of either paper's evaluation protocol.

`protector.py` is the maintained FaceShield runtime and CLI. The original Linux/distributed training-era launch scripts are outside the active application; consult the [historical upstream README](UPSTREAM_README.md) and upstream repository for that research setup.

Only feature encoders needed by PhantomSeal's objective are loaded. FaceShifter/HifiFace synthesis generators, discriminators, Lightning training modules and the BFM renderer are unnecessary for those feature losses. Defense engines release GPU models between stages. The SimSwap test generator is loaded for testing, not retained alongside a running protection job.

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

A custom processing request uses this shape:

```json
{
  "source_id": "registered-upload-id",
  "params": {
    "mode": "custom",
    "max_size": 1280,
    "stages": [
      {"type": "nullswap", "params": {"strength": 1, "seed": 0}},
      {"type": "faceshield", "params": {"iterations": 30, "noise_clamp": 12, "step_size": 1, "analysis_size": 512}},
      {"type": "nullswap", "params": {"strength": 0.5, "seed": 42}}
    ]
  }
}
```

PhantomSeal stages use `{"type": "phantomseal", "params": {"iterations": 1000}}`. Omitted stage parameters receive that filter's defaults. Empty/oversized recipes, unknown types or stage fields, invalid numeric values and parameters belonging to another filter are rejected. A legacy mode cannot also supply `stages`.

Custom completion includes the normalized `pipeline` recipe in metadata and `intermediate_assets` entries with `index`, `type`, `label`, registered `asset` and `params`. These allow stage downloads without exposing arbitrary file paths. A custom result test publishes clean/final comparisons for the complete recipe; legacy chain tests retain their separate PhantomSeal row.

Protection completion returns the clean normalized `source`, optional `stage1`, final `result` and metrics. Test jobs have `kind: "test"` and `result_id`, and never overwrite the main editor assets. Startup checks the active job kind before resuming polling. A test may continue while its modal is closed; its result belongs to the exact completed protection result, not whichever settings are currently selected.

The 0.80 baseline cutoff and per-condition visual confirmations are browser interpretation controls. The cutoff is not added to the `/api/test` request. New photos, results, test jobs or cutoff changes invalidate confirmations; cached raw test data is gated again when displayed. See [Validation](VALIDATION.md#interactive-baseline-gate).

The server accepts loopback hosts on its configured port, rejects foreign origins/cross-site requests, bounds JSON/image input, and serves a fixed set of static files. These are concrete local safeguards, not a formal security certification. Assets and job registries are in memory. Files remain under `results/editor/`, but restarting the server does not guarantee restoration of the prior browser session.

## Checks and benchmarks

```powershell
.\manage.cmd test
.\manage.cmd smoke
.\manage.cmd benchmark-help
.\manage.cmd benchmark -Sources 1 -OutputDir "results\benchmarks\quick"
```

The management test command hides CUDA, runs PowerShell lifecycle tests and Python `unittest` discovery, then runs `tests/frontend-check.cjs` when the project-local Node.js runtime is available. It verifies dimensions/no upscale, perturbation bounds, preprocessing gradients, sequential loss equivalence, safe loader contracts, route validation, jobs, baseline failures, source weighting, benchmark resume integrity and UI state/export behavior. Many tests use dummy engines; their success does not establish defense efficacy.

`smoke` executes the full installed model path without an HTTP server: a public source enlarged to 1280 px, 10 PhantomSeal steps, 1 FaceShield step including the optional attack detector, then six real SimSwap outputs across PNG/JPEG95. The probe blocks Python socket resolution/connection attempts and records file access, loaded module paths, model paths and interpreter isolation. Its JSON report lives under `results/verification/`. The audit covers observed Python-level access and imports; it is not an OS sandbox or proof about every native DLL system call. Host Windows/NVIDIA components remain necessary.

For the lightweight Python path audit alone:

```powershell
.\.venv\Scripts\python.exe -I scripts/manage_checks.py runtime --root .
```

The full benchmark manifest is `benchmark_manifest.json`. Protection outputs are reused only after resume integrity checks. A comparison is keyed by `(source_id, target_id, condition)`. Reports keep PNG and JPEG95 separate, report invalid/detection/weak-baseline counts, and average target-level deltas within each source before equal weighting across sources. They do not report p-values or an attack-success rate from uncalibrated cosine values.

Keep `weights/**` out of commits except its tracked placement guide `weights/README.md`. The full bank, binary models, resource manifests and Hugging Face resource cache share that ignored tree. Keep archives, `.runtime/`, `.venv/`, other caches, private photos and generated editor jobs out of commits as well. Versioned sanitized numerical evidence belongs under `docs/evidence/`; full public HTML/image bundles can be retained locally under ignored `results/benchmarks/`.

## Additional model adapters

The `nullswap` mode instantiates only its own generator and detector. SimSwap evaluation loads only its identity encoder, generator and evaluator, independently of the PhantomSeal cloak bank or other ensemble heads. Each new swap adapter owns its preprocessing and native output resolution; the common evaluator compares quantized outputs to the same original identity and catches typed native-alignment failures as null-score rows. Other runtime errors remain job failures. One shared job lock prevents overlapping protection and testing. See [NEW_MODELS.md](NEW_MODELS.md) for implementation provenance and differences.
