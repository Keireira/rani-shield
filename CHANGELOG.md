# Changelog

## Ordered protection pipelines — 2026-10-21

- Add an editor pipeline builder with 1–8 ordered NullSwap, PhantomSeal and FaceShield stages, independent parameters, repeated filters and per-stage PNG downloads.
- Save complete recipes in presets and migrate existing fixed-mode settings without losing their values.
- Validate recipes before processing, execute stages sequentially with model cleanup and preserve the clean original as the comparison baseline.
- Keep legacy API modes compatible; evaluate custom pipelines against their final result in real face-swap checks.

## Foreground server launch — 2026-10-21

- Make `manage.cmd run` and `make run` keep the server and live logs in the current terminal until exit; use Ctrl+C to stop it.
- Route `launch.ps1` through the same foreground command without opening a browser. New launches create no background server, PID record or server log files.
- Reject occupied ports and active recorded legacy servers without reusing or stopping them. Keep `status` for port inspection and `stop` only for verified background servers launched by an older version.

## NullSwap parameter controls — 2026-10-21

- Add NullSwap strength (0–4) and seed (0–4294967295) to the sidebar, Settings and presets, with defaults unchanged at strength 1 and seed 0.
- Validate both parameters in the API and pipeline, record them in result metadata, and scale the mapped residual before PNG rounding.
- Verify unchanged default pixels, reproducible seeds and strength bounds on the real model at 1280×1280 without starting a server.

## NullSwap and selectable real-swap checks

- Add standalone NullSwap using PhantomSeal's explicitly labelled reconstruction; bounded 256px residual mapped to output up to 1280px.
- Add DiffSwap, InfoSwap and UniFace test adapters alongside SimSwap, with strict local checkpoint loading, native preprocessing and shared baseline gates.
- Separate evaluation from the PhantomSeal ensemble so tests load only their required models.
- Add verified optional archive import and `import-models` / `smoke-extra` management and Make targets.
- Retain source/license notices and document preprocessing adaptations and weak-baseline limitations in `docs/NEW_MODELS.md`.


## Unreleased

- Rename the application to **Rani Shield**, keeping FaceShield and PhantomSeal
  as the names of the upstream algorithms.
- Share parameter validation and image transformations across the API, pipeline
  and benchmarks; release partially loaded models after failures.
- Fix the optional FaceShield MTCNN loader and Windows short-path test assertions.
- Add serverless `make smoke` with real GPU processing, PNG/JPEG95 swaps,
  network blocking and Python/model file provenance checks.
- Prepare project-local Python and Node runtimes under `.runtime/`, and bind
  `.venv` to the local Python installation.
- Enforce project-contained runtime resource paths, relative configuration and
  local model caches/temporary files. Imports from older installations remain
  explicit one-time copies.
- Remove unused upstream Linux/distributed launch and face-parser modules from
  the active application. The supported FaceShield implementation remains
  `protector.py`; original research documentation and notices are retained.
- Rewrite the README and Windows guides with explicit credits to both
  **Why0912/FaceShield** and **kuai-lab/iccv25_faceshield**, their distinct roles,
  the required model subset and the limits of the protection evidence.
- Consolidate model weights, tokenizer/configuration files, the complete cloak
  bank, model manifests and model caches under `weights/`.
- Update runtime, benchmark, download, import and launch paths to use the unified
  layout. Keep Python network definitions in their source directories.
- Add an in-project migration with hash/conflict checks, repeatable execution
  and verified cleanup of the duplicate SimSwap identity checkpoint.
- Load the three MTCNN detector states explicitly from `weights/mtcnn/`, with
  CPU parity checks against the pinned packaged detector. The optional
  FaceShield attack detector also uses an explicit local checkpoint.

See [weights/README.md](weights/README.md) for the directory map and migration
commands. Large local resources remain outside Git history.

## 0.1.0 — 2026-10-20

Initial standalone local application assembled from the implementation and
validation work in this project:

- English photo editor with comparison, difference map, settings and PNG export.
- PhantomSeal, FaceShield, and sequential PhantomSeal → FaceShield processing.
- Full four-branch PhantomSeal objective and 214-image mixed cloak bank.
- Aspect-preserving output up to a 1280 px long edge, without enlarging inputs.
- Real SimSwap diagnostics for PNG and JPEG95 with separate baseline quality
  and manual likeness checks. Weak or unconfirmed baselines are inconclusive.
- Windows setup, model preparation, configuration, doctor, process management,
  CPU tests, and paired benchmark commands through Make or manage.cmd.
- Corrected benchmark reporting, upstream attribution and model provenance.

The application is complete as a local research workflow. Reliable protection
against arbitrary face-swapping tools has not been demonstrated; see
[validation findings](docs/VALIDATION.md).
