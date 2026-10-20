# Rani Shield

A local Windows photo editor for creating and inspecting experimental protection against face swaps. Upload a portrait, run **PhantomSeal**, **FaceShield**, or **PhantomSeal → FaceShield**, inspect the changes, export a PNG, and test the result with a real SimSwap comparison. The interface is in English and supports an output long edge up to **1280 px**.

**Project credits:** this project grew out of work with [**FaceShield — Why0912/FaceShield**](https://github.com/Why0912/FaceShield) and is directly built on the protection implementation in [**iccv25_faceshield — KUAICV FaceShield**](https://github.com/kuai-lab/iccv25_faceshield). Their roles are described in [Credits and licenses](#credits-and-licenses), alongside PhantomSeal and the other upstream projects.

**Reliable face-swap protection has not been demonstrated by the current local tests.** A completed PNG and a working comparison are software results; they do not establish that a photo is safe from face swaps. The [validation record](docs/VALIDATION.md) preserves the measurements and their limitations.

## Start here

The supported configuration is **64-bit Windows with an NVIDIA GPU and compatible driver**. The full pipeline has been exercised on an **RTX 5070 Ti with 16 GB VRAM**. That is a tested system, not a minimum-memory guarantee.

Open PowerShell in `F:\Git\rani-shield`. If this installation already contains its environment and model files:

```powershell
.\manage.cmd doctor
.\manage.cmd run
```

Open [http://127.0.0.1:7876](http://127.0.0.1:7876). `run` keeps the server and its live logs in the current terminal until it exits. Press **Ctrl+C** in that terminal to stop it.

Starting a server is an explicit action. Setup, model preparation, checks and the default help command do not require the editor to be running. `run` creates no background server, PID record or server log files, and refuses to start if the port or a recorded legacy server is already in use. It does not reuse or stop that process. For a different port, use `.\manage.cmd run -Port 7877`. `launch.ps1` delegates to the same foreground command; neither launcher opens a browser automatically.

Use `status` from another terminal to inspect port occupancy. `stop` is retained only for background servers started by an older version, and requires a verified ownership record; foreground servers are stopped with **Ctrl+C**.

For a new source checkout:

```powershell
.\manage.cmd setup
.\manage.cmd models
```

`setup` prepares local Python 3.11 under `.runtime/python/`, the project's `.venv`, PyTorch **2.7.1+cu128**, torchvision **0.22.1+cu128**, and [requirements-windows.txt](requirements-windows.txt). Bootstrap can use an installed Python 3.11 and Node.js to prepare these local runtimes. After preparation, processing uses the interpreter, standard library and packages inside this repository; the original Python installation and sibling projects are not runtime dependencies. Node.js under `.runtime/node/` is used for frontend checks.

If automatic Python discovery is unavailable, pass `.\manage.cmd setup -Python311 "D:\path\to\Python311\python.exe"`. Node.js is optional for the application: setup copies `node.exe` and its adjacent `LICENSE` when available. Otherwise place both files from the same Node.js distribution in `.runtime/node/` to enable frontend checks. Keep the runtime license notices.

The full installation includes large local resources that Git does not contain. `models` downloads the public base weights and smaller PhantomSeal/SimSwap resources; **the full PhantomSeal checkpoint and data archives require a separate download**. Place those resources using [weights/README.md](weights/README.md), then run `doctor`.

A [Russian quick start](QUICKSTART.md) and [Windows troubleshooting guide](WINDOWS.md) are available.

## Protect and inspect a photo

1. Use **Open image** or drag in a photo.
2. Select a method and settings, then choose **Protect photo**.
3. Inspect **Original**, **Slider**, **Side by side**, **Result** or **Diff**. Pan, zoom and **Fit** affect only the view.
4. Choose **Export PNG**. Original mode exports the original; Diff exports the full-resolution pixel difference; the other modes export the protected result. The PhantomSeal intermediate has a separate export action.

| Method | Processing |
|---|---|
| PhantomSeal → FaceShield | Default: 1,000 PhantomSeal steps, followed by 30 FaceShield steps |
| PhantomSeal only | Four identity/shape feature branches and the full 214-photo cloak bank |
| FaceShield only | Adapted diffusion, image-prompt and identity objectives from KUAICV FaceShield |
| NullSwap · reconstruction | Feed-forward PhantomSeal reconstruction; native aligned 256px residual mapped to the full image, adjustable strength and seed; defaults remain gain 1, seed 0 and at most 8/255 pixel change |

The output preserves aspect ratio and limits the long edge to the selected size, up to **1280 px**. Smaller inputs are not enlarged. PhantomSeal and FaceShield optimize their perturbations at the output resolution, with a differentiable 256 px aligned face view for PhantomSeal and a 512 px diffusion view for FaceShield. The two stages do not resize the image between them.

The editor serializes GPU jobs and releases model resources between stages. Uploads and generated files stay in `results/editor/`. Its asset registry is held in memory, so restarting the server may require uploading the photo again even when old PNG files remain on disk.

## Test the protection result

**Test protection** copies the source identity into one of three public demo target faces with the selected **SimSwap, DiffSwap, InfoSwap or UniFace** model. The target supplies the destination face and pose; it is not the uploaded person. The panel shows the original, the target, the **Unprotected swap**, and swaps made from the protected variants. JPEG95 optionally repeats the comparison after compression.

Check the **Unprotected swap** first. If that face does not visibly resemble the original, it is a weak test of protection. For each condition:

- The default numeric baseline cutoff is **0.80 cosine similarity**, not 80% confidence and not a universal identity threshold.
- Confirm **The unprotected swap looks like me** only when the baseline is visibly recognizable.
- A weak, undetected or unconfirmed baseline remains **Inconclusive**. Comparative effects stay hidden.

Lower FaceNet cosine describes lower similarity under that evaluator for that comparison. It is not a probability of protection. SSIM, PSNR and changed-pixel measurements describe image changes, not resistance to face swapping.

This test covers using the uploaded portrait as an **identity source**. It does not measure prevention of replacing the face in that portrait as a destination. Each run uses one selected swapper. SimSwap/FaceNet also participate in the optimization or cloak selection. The existing six-source benchmark leaves only **8 of 36 comparisons above the numeric baseline cutoff, from two source identities**, before visual acceptance. The new models also require a usable unprotected baseline; their availability does not imply stronger protection. See [Validation](docs/VALIDATION.md) before drawing conclusions.

## Commands

GNU Make is optional; every operation is available through `manage.cmd`. A bare `make` or `manage.cmd` prints help.

| Purpose | Windows command | Optional Make command |
|---|---|---|
| Prepare environment | `.\manage.cmd setup` | `make setup` |
| Prepare public base models | `.\manage.cmd models` | `make models` |
| Import optional model archive | `.\manage.cmd import-models -Checkpoints "PATH\TO\checkpoints"` | `make import-models CHECKPOINTS="PATH/TO/checkpoints"` |
| Check environment, CUDA and assets | `.\manage.cmd doctor` | `make doctor` |
| Run the editor in this terminal (Ctrl+C to stop) | `.\manage.cmd run` | `make run` |
| Inspect port occupancy | `.\manage.cmd status` | `make status` |
| Stop a verified legacy background server | `.\manage.cmd stop` | `make stop` |
| Run automated checks | `.\manage.cmd test` | `make test` |
| Exercise installed models without a server | `.\manage.cmd smoke` | `make smoke` |
| Exercise NullSwap and all four swappers | `.\manage.cmd smoke-extra` | `make smoke-extra` |
| Benchmark options | `.\manage.cmd benchmark-help` | `make benchmark-help` |
| Full real-swap benchmark | `.\manage.cmd benchmark` | `make benchmark` |

`test` runs CPU Python and PowerShell checks, plus frontend checks when the local Node.js runtime is available. It does not evaluate defense efficacy or need an editor server. The full benchmark uses six public sources, three targets and PNG/JPEG95: **36 paired cases and 108 swaps**. It uses the GPU and writes JSON, CSV, HTML and image evidence under `results/benchmarks/`; do not run it alongside another GPU job. A smaller run is:

```powershell
.\manage.cmd benchmark -Sources 1 -OutputDir "results\benchmarks\quick"
```

`doctor` checks readiness without loading the model tensors. An all-green readiness report cannot replace a processing test or establish protection quality. Current and historical execution checks are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

`smoke` uses the installed GPU models directly, with 10 PhantomSeal steps and 1 FaceShield step on a public portrait enlarged to 1280 px, then generates six PNG/JPEG95 SimSwap outputs. It starts no server, blocks network attempts and records Python file/module access and loaded model paths. This is a bounded execution check, not a defense benchmark or native high-resolution efficacy evidence. Reports go under `results/verification/`.

`smoke-extra` exercises NullSwap on the same public 1280px fixture and all four real swappers for PNG/JPEG95, with the same offline and local-path audit. It generates 16 published outputs; single-stage duplicate inputs are reused by the new adapters. DiffSwap uses 200 DDIM steps per distinct image. Neither smoke command starts an HTTP server.

NullSwap here is explicitly **PhantomSeal's best-effort reconstruction**, not an original author checkpoint or an exact reproduction of the paper. See [New models](docs/NEW_MODELS.md) for sources, native resolutions, preprocessing differences and model-specific limitations.

## Files and isolation

All application dependencies and data are kept under the repository root. The host still supplies Windows, the NVIDIA driver and a browser; PowerShell is used by the management scripts and GNU Make is optional. Active model files, full cloak data and model caches belong under **`weights/`**. [phantomseal_config.json](phantomseal_config.json) uses repository-relative paths. Runtime paths that escape the project are rejected. The separate import command may read an old installation once to copy resources in; it does not leave a runtime link to that installation.

| Location | Responsibility |
|---|---|
| `.runtime/python/` | Project-local Python executable, standard library and DLLs |
| `.runtime/node/` | Project-local Node.js for frontend checks |
| `.venv/` | Project Python packages, using the local base interpreter |
| `web/` | English editor, comparison canvas and baseline-gated test panel |
| `gui.py`, `app.py` | Local API, assets and job lifecycle |
| `swap_adapters/`, `swap_evaluation.py`, `face_evaluation.py` | Independent swapper loading, native preprocessing and common baseline measurement |
| `nullswap_runtime.py` | Standalone bounded NullSwap reconstruction |
| `protection_pipeline.py` | Image normalization and four processing modes |
| `runtime_contracts.py`, `image_io.py` | Shared validation, image dimensions and decoding/compression helpers |
| `phantomseal_runtime.py`, `ensemble_encoders.py` | Full PhantomSeal feature ensemble and optimization |
| `protector.py` | FaceShield runtime adapter |
| `protection_test.py`, `benchmark_*.py` | Real SimSwap tests, scoring and reproducible reports |
| `scripts/`, `manage.cmd`, `manage.ps1`, `Makefile` | Environment, resource preparation and management |
| `weights/` | All model checkpoints, tokenizer/configuration files, full cloak bank and model manifests |
| `third_party/` | Vendored inference code and upstream notices |
| `results/`, `.cache/` | Generated outputs, logs and local temporary files |
| `docs/`, `tests/` | Architecture, provenance, evidence and regression checks |

Weights, environments, caches, private uploads and generated results are excluded from Git. Historical evidence and import manifests can mention old paths as provenance; those records do not provide runtime resources. Details are in [Development](docs/DEVELOPMENT.md), the [weights placement guide](weights/README.md) and [PhantomSeal notes](PHANTOMSEAL.md).

## Credits and licenses

Special credit to both projects that led to this application:

- **[FaceShield — Why0912/FaceShield](https://github.com/Why0912/FaceShield)**, *FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models*. This Bunny-based project was the starting point of the original investigation and development effort. Its task is face anti-spoofing; Rani Shield's current protection pipeline does not run its RGB/SAV multimodal model. This acknowledgment identifies the project lineage without claiming that an unavailable trained FAS checkpoint is integrated.
- **[iccv25_faceshield — kuai-lab/iccv25_faceshield](https://github.com/kuai-lab/iccv25_faceshield)**, *FaceShield: Defending Facial Image against Deepfake Threats* (ICCV 2025), by Jaehwan Jeong, Sumin In, Sieun Kim, Hannie Shin, Jongheon Jeong, Sang Ho Yoon, Jaewook Chung and Sangpil Kim. This is the direct source-code and algorithmic foundation of the FaceShield protection stage. Rani Shield adapts it for the Windows editor, local models, GPU memory use and full-resolution output perturbations. [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Jeong_FaceShield_Defending_Facial_Image_against_Deepfake_Threats_ICCV_2025_paper.html).

Required attribution: **Korea University Artificial Intelligence Computer Vision Lab (KUAICV), Department of Artificial Intelligence, Korea University — [kuaicv.com](https://kuaicv.com/)**. The original [KUAICV Research-Only License](LICENSE) is retained unchanged. Its restrictions apply to this derivative, including the requirement for written permission for commercial use. The MIT license of a separate component does not relicense the combined application, pretrained models or datasets.

Additional credits:

- **[PhantomSeal](https://github.com/LiangqinRen/PhantomSeal)** — the four-branch `unify` objective, released checkpoints and full mixed cloak bank.
- **[SimSwap](https://github.com/neuralchen/SimSwap)** — identity encoder, real swap generator and public demonstration images.
- **FaceShifter, HifiFace, Deep3D, InsightFace/ArcFace and [facenet-pytorch](https://github.com/timesler/facenet-pytorch)** — feature encoders, face detection and measurement; exact sources and retained notices are listed in [Provenance](docs/PROVENANCE.md).
- **Stable Diffusion, IP-Adapter and CLIP** — the diffusion and image-conditioning components of FaceShield.
- **DeSynth** — the local editor interaction design that informed pan/zoom, comparison views and PNG export.

See [docs/PROVENANCE.md](docs/PROVENANCE.md) for component licenses, model/data sources, preserved notices and hashes, and [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md) for the historical ICCV FaceShield documentation. No upstream affiliation or endorsement is implied.
