# Implementation decisions

This records the choices behind the local workflow assembled during development.

| Topic | Implemented decision |
|---|---|
| User workflow | Upload a photograph, choose a method, generate a perturbed PNG, inspect it and optionally run a real swap diagnostic. The interface is English. |
| Protection model setup | Use released pretrained components and optimize the input image for each job. No model training from scratch or private training dataset is required for this runtime. |
| Combined method | Run PhantomSeal first, save that exact intermediate image, then apply FaceShield. Single-method modes remain available. |
| PhantomSeal completeness | Use the original 214-photo mixed bank and all four feature branches of the selected `unify` objective. This adapter does not implement every context-protection or forensic-tracing feature of the original framework. |
| Other research directions | NullSwap was investigated as a possible alternative but was not integrated or validated here. Its checkpoint and unrelated attackers from the large archive are not required by this application. |
| Resolution | Keep a full-resolution perturbation up to a 1280 px long edge, preserve aspect ratio and avoid enlarging uploads. Compute expensive losses through smaller differentiable views. |
| Interface interaction | Follow the local DeSynth comparison workflow: pan/zoom, slider, side-by-side, pixel difference and PNG export. |
| Evaluation | Compare actual SimSwap outputs from the clean, intermediate and final images, with an unchanged clean identity reference. Also test JPEG95. |
| Baseline validity | Require a configurable clean-swap cosine cutoff and a separate manual likeness check. Weak or unconfirmed baselines are inconclusive. The earlier unfiltered 28/36 efficacy conclusion was withdrawn. |
| High-resolution evidence | Treat the upscaled public example as an operational resolution check, not a new native high-resolution identity or proof of protection. |
| Installation | Provide Windows management commands with optional Make wrappers, deterministic process ownership checks, local model import and explicit full-archive configuration. |
| Resource layout | Keep all active weights, the full 214-photo bank, resource manifests and optional Hugging Face downloads under `weights/`. Configure `checkpoint_dir` as `weights` and `cloak_dir` as `weights/phantomseal/cloaks/mix_full`; a reviewed external override remains available. |
| Existing installations | Import a complete old installation into the unified tree, or preview the in-project migration with `scripts/migrate_weights.py` and apply it explicitly with `--apply`. Verify hashes and avoid moving environments, results or outside-project files. |
| Source control | Keep code, public demo examples, notices, tests, sanitized numerical evidence and `weights/README.md` in Git. Ignore the other `weights/**` resources, private images, generated sessions and environments. Historical evidence inventories retain their original resource paths. |

See [Validation](VALIDATION.md) for the evidence and remaining uncertainty,
[Development](DEVELOPMENT.md) for implementation details, and
[Provenance](PROVENANCE.md) for component sources and retained terms.
