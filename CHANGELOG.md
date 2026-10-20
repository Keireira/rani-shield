# Changelog

## Unreleased

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
