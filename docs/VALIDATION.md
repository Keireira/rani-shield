# Validation and interpretation

**The current local evidence does not establish reliable face-swap protection, and does not establish that chaining FaceShield improves PhantomSeal.** The application successfully generates images and real SimSwap comparisons. Those operational facts are separate from whether a swapper first copied the intended identity and whether a defense prevented that copying.

The evidence below was collected on 20 October 2026. Versioned, sanitized measurements are in [six-source.json](evidence/six-source.json) and [resolution1280.json](evidence/resolution1280.json). Full public reports, when retained locally, are under `results/benchmarks/2026-10-20-full/` and its `resolution1280/` subdirectory; those HTML/image bundles are ignored runtime artifacts. Private user-photo tests are not part of this evidence set.

## Scope of the measurement

The protected photo is an **identity source**: SimSwap extracts its identity and inserts it into a fixed target face. The test does **not** measure whether the protected photo resists having its own face replaced when used as the destination.

For each source/target/condition, the generator creates three actual swaps:

1. Clean, unprotected source → fixed target.
2. PhantomSeal intermediate → the same target.
3. PhantomSeal + FaceShield final → the same target.

The paired SimSwap generator and its own identity backbone are used for synthesis. Attack sources are independently detected and aligned with SimSwap's five-pose template. Generated output faces are independently detected and cropped/resized by MTCNN for FaceNet scoring. FaceNet cosine is measured against the **original clean PNG source** in both PNG and JPEG95 conditions. JPEG95 transforms the attack inputs, including the clean baseline, using quality 95 and 4:2:0 subsampling; it does not change the reference identity.

A lower cosine means the output resembles the reference less **according to FaceNet**. It is not a probability, a calibrated attack-success threshold or a guarantee that a human sees another identity. FaceNet participates in cloak selection, and SimSwap is one optimization branch, so neither constitutes a completely unseen evaluator.

## Baseline correction

The initial raw ranking showed a lower chain score in 28/36 comparisons. **The unqualified efficacy interpretation of that ranking is withdrawn.** Many unprotected swaps were already poor likenesses. A further decrease from a failed clean swap cannot establish that protection defeated a successful attack.

A subsequent numerical screen requires a complete paired triple, successful source/output face detection, finite scores and clean-swap cosine **at least 0.80**. It retains only **8 of 36 comparisons, from two source identities**. Each condition retains four numerical comparisons. These eight still require visual baseline inspection; they are not eight successful defenses and do not define a success rate.

The 0.80 cutoff is a local, adjustable quality heuristic. It is **not 80% confidence**, a universally validated FaceNet identity threshold, or evidence that a clean output visibly resembles the person. Reports retain weak and invalid cases with explicit counts instead of silently treating them as successful protection. A source-versus-target baseline sanity check is also reported separately; it does not substitute for checking likeness.

## Six-source public comparison

The fixed manifest contains six distinct detectable public source portraits at native sizes of 224–512 px, three target portraits and two conditions. Each source is protected once at the default 1,000 PhantomSeal + 30 FaceShield steps, then tested against the same targets. This gives **36 paired comparisons and 108 generated swaps**, not 36 independent source identities.

The following are **unfiltered descriptive source-weighted means**, including weak baselines. They are retained for traceability rather than as efficacy evidence:

| Condition | Unprotected swap | PhantomSeal | PhantomSeal + FaceShield |
|---|---:|---:|---:|
| PNG | 0.6372 | 0.4725 | 0.4460 |
| JPEG95 | 0.6149 | 0.4749 | 0.4169 |

The raw chain score was lower in 28/36 comparisons and in 5/6 source means across conditions. For the `james` source the raw means went the other way: 0.5420 for PhantomSeal and 0.6505 for the chain. Neither direction, without a credible baseline, proves a defense effect. All 108 generated faces were detected; that confirms face generation, not correct identity copying.

Distortion budgets differ. Mean protected-image SSIM was 0.6358 for PhantomSeal and 0.6251 for the chain, while maximum measured pixel changes were 19 and 31/255. Any apparent improvement may come with greater distortion. This is a comparison of the application's default settings, **not a matched-budget ablation**.

The `lcw` input was excluded because the application's detector found no face before swap scoring; `gy` replaced it after detection preflight. No protection parameters were adjusted from swap results. The `portrait`/`cage` pairing had been used in earlier exploration, so the suite is not strictly held out.

## 1280 px supplement

An additional application probe used the prior 1280 × 1280 protected portrait on the same three targets and both conditions: six paired comparisons and 18 generated swaps. That portrait had been **upscaled from a 512 px public image before the probe**. It is a reused identity and is not independent evidence from a native high-resolution source. This prepared test image does not change the application's no-upscale rule for uploaded inputs.

Again, these are raw descriptive means:

| Condition | Unprotected swap | PhantomSeal | PhantomSeal + FaceShield |
|---|---:|---:|---:|
| PNG | 0.8227 | 0.8078 | 0.8174 |
| JPEG95 | 0.8242 | 0.8094 | 0.8182 |

PhantomSeal alone had a lower raw score in 4/6 comparisons. Differences were small, and the original swap still requires visual likeness confirmation. **These measurements do not demonstrate reliable 1280 px protection or a benefit from the second stage.**

Supporting 1280 px output means the application preserves a full-resolution delta while evaluating smaller differentiable model views. It does not mean the diffusion UNet runs natively at 1280 px, nor that the papers' reported performance transfers to this adapter.

## Interactive baseline gate

The editor uses the following interpretation policy independently for PNG and JPEG95:

| Baseline state | What the UI shows |
|---|---|
| Failed face detection, null/nonfinite score, or clean cosine below the selected cutoff | **Inconclusive — weak baseline**, actual score/cutoff, raw images and scores; no comparative effect delta |
| Numerical gate passes, but visual confirmation is unchecked | Inconclusive pending likeness confirmation; no comparative effect delta |
| Numerical gate passes and the user confirms the unprotected swap looks like them | Measured comparison delta, subject to successful detection/finite values for the compared rows |

The original photo and demo target are displayed separately, and the clean generated card is called **Unprotected swap**. Visual checkboxes start unchecked and reset for a new photo, result, test job or cutoff change. Cached tests are screened again on reload/open. The server's raw delta cannot bypass these browser interpretation gates.

A negative `final − stage1` cosine delta means the final swap has lower evaluator similarity than the PhantomSeal swap for that case. A positive delta means greater evaluator similarity. An accepted baseline is not a defense “PASS”; the application does not display a Safe badge or claim general protection.

## FaceShield weight caveat

The local copy of the archive linked by the [FaceShield model setup](https://github.com/kuai-lab/iccv25_faceshield#-model-setup) contains byte-identical `arcface50_checkpoint.tar` and `arcface100_checkpoint.tar`. Both have SHA-256:

```text
52ea5ce4902017b77a2bb811dd9a82f57dee3883e06c18a42d288437263c7a20
```

This installation therefore does not establish two distinct FaceShield ArcFace networks merely from those filenames. The caveat limits reproduction claims. It does **not** explain the weak clean SimSwap baseline, because synthesis uses SimSwap's separate paired generator and identity backbone.

## Operational validation versus efficacy

After consolidation under `weights/`, **95 Python tests** plus PowerShell and frontend checks passed. The migration verified the original resource bytes, removed an identical SimSwap alias only after checking its canonical copy, and left all active weights inside the central directory. The three local MTCNN states match the pinned packaged detector's parameters and public-image detections. A fresh 1280 × 1280 run loaded the complete ensemble and FaceShield from the new paths, using 10 PhantomSeal steps and 1 FaceShield step, then generated real PNG/JPEG95 SimSwap outputs. This is a bounded execution check, not an efficacy evaluation. The current inventory and checks are recorded in [weights-layout.json](evidence/weights-layout.json).

The standalone checkout was checked again after assembly: **81 Python tests**, PowerShell management checks and frontend regressions passed. Its own Python 3.11 environment passed `pip check`, CUDA was available, and model readiness found all four branches and the 214-image bank. The new server uses port 7876 independently of the earlier installation.

A fresh run from this checkout completed the default 1,000 + 30 step chain on an explicitly enlarged public 1280 × 1280 input, downloaded and decoded the final PNG, preserved the intermediate PNG, and generated all six real swap outputs for PNG/JPEG95. The UI kept comparison effects hidden pending visual baseline acceptance. See the path-free [operational record](evidence/studio-smoke.json). This reused public identity is a software execution check, not additional independent efficacy evidence.

The integrated runtime has been exercised on a Windows RTX 5070 Ti 16 GB system, including a full-ensemble 1280 px application job, PNG exports and actual SimSwap tests. Strict CPU loads and input-gradient checks covered the four feature branches. Automated tests cover resolution/no upscale, bounds, preprocessing, sequential gradients, API validation, job serialization, benchmark aggregation/resume, export and baseline-gate regressions.

These checks support that the software executes the described path and handles invalid data. They do not validate broad robustness to different swappers, destination attacks, social-platform processing, alternate JPEG qualities, resizing, denoising or deliberate adaptive attacks. The sample is small, one swapper is used, some evaluators overlap the objective, and numerical eligibility represents only two source identities. No p-values, calibrated attack-success rates or universal efficacy claims follow from this record.
