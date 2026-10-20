# NullSwap and additional face-swap tests

Rani Shield supports **NullSwap as a separate protection mode**, and **DiffSwap,
InfoSwap and UniFace as selectable result-test backends**, alongside SimSwap.
Choosing another test backend generates another real face swap; it does not add
another protection layer to the photograph.

## Local setup

All runtime checkpoints and statistics belong under this repository's `weights/`
directory. No adapter reads the former project's cache or downloads a missing
model during processing.

The optional-model importer accepts the old extracted `checkpoints` directory,
or its ZIP archive. For example, from the Rani Shield repository:

```powershell
.\.venv\Scripts\python.exe scripts/import_swap_assets.py --checkpoints "PATH_TO_CHECKPOINTS_DIR_OR_ZIP" --models all --dry-run
.\.venv\Scripts\python.exe scripts/import_swap_assets.py --checkpoints "PATH_TO_CHECKPOINTS_DIR_OR_ZIP" --models all
```

The old directory is an **installation source only**. After import, the project
uses its local copies. `--models nullswap`, `diffswap`, `infoswap` or `uniface`
selects a subset. InfoSwap also needs 11 readout-statistics files published with
its source; the importer downloads the pinned files if absent. `--no-download`
requires them to be available already.

The shared MTCNN detector files and FaceNet evaluation checkpoint stay in
`weights/mtcnn/` and `weights/facenet/`. DiffSwap also requires the pinned
`dlib-bin` package installed by the project's setup. Other new models use the
existing PyTorch, NumPy, Pillow and OpenCV packages.

| Feature | Required model-specific files inside `weights/` | Native model view/output |
| --- | --- | --- |
| NullSwap | `nullswap/epoch_0001_step_012000.pth` | Aligned 256px face; residual mapped to the full image |
| DiffSwap | `diffswap/diffswap.pth`, `diffswap/glint360k_r100.pth`, `diffswap/shape_predictor_68_face_landmarks.dat` | Aligned 256px generated face |
| InfoSwap | `infoswap/model_128_ir_se50.pth`; `infoswap/w_kernel_smooth/ckpt_ks_E.pth`, `ckpt_ks_G.pth`, `ckpt_ks_I.pth`; `infoswap/readout/readout_layer0.pth` through `readout_layer10.pth` | Aligned 512px generated face |
| UniFace | `uniface/500000.pt` | 256px generated face |

Discriminator checkpoints, training optimizers and the unused InfoSwap
no-smoothing variant are unnecessary for these inference paths. Imported asset
hashes and provenance are retained in `weights/manifests/`.

## NullSwap protection

The available checkpoint belongs to **PhantomSeal's reconstruction of NullSwap**.
Its own model source calls it a best-effort reconstruction of the paper's Figure 2;
it has not been established as the original NullSwap authors' released model.

Rani Shield detects the largest clear face, aligns it to 256px, evaluates that
network and maps the actual `protected_face - clean_face` residual back to the
photograph. The output retains its aspect ratio, with a maximum 1280px long edge
and no enlargement of smaller inputs. Pixels outside the transformed face crop
remain unchanged. The **NullSwap seed** control selects the random noise (integer
0–4294967295, default **0**). Reusing it reproduces the noise on the same runtime/device.

**NullSwap strength** scales the residual after mapping it to the photograph.
Its range is **0–4**, default **1**, with UI increments of 0.25. At the default,
the original **8/255** pixel bound and output are preserved. Strength 2 gives
a bound of 16/255; strength 0 preserves the normalized original pixels.
The 8-bit PNG rounds the result; the maximum integer pixel change is bounded
by `ceil(8 × strength)`. This scales the generated residual, without changing
the checkpoint's native epsilon or rerunning optimization.

Both controls are available in the sidebar and Settings, enabled only for
NullSwap, and included in saved presets. Old settings default to strength 1
and seed 0. Increasing strength can increase visible distortion and does not
guarantee stronger protection. PhantomSeal's metric-only gain 2.75 is not
applied automatically. Metadata records the chosen gain/seed, native bound,
scaled bound before rounding, integer bound and actual maximum pixel change.

This is a full-image adaptation of a 256px network. The ability to export a
1280px image does not establish protection efficacy at that resolution.

## Test backends

**DiffSwap** performs diffusion-based face replacement with identity, geometry
and masking conditions. The adapter retains the released EMA denoiser,
200 DDIM steps, `eta=0` and target guidance `0.01`. Dlib supplies 68 face
landmarks for FFHQ-style cropping and mask construction; local MTCNN supplies
five points for the 112px ArcFace identity view. Source and target latents are
encoded separately, following PhantomSeal's correction to the original dataset
script. Generated and target pixels are combined with the original multiband
blending procedure. Missing detection is an explicit failure.

**InfoSwap** uses the official native 512px smooth checkpoint set and
information bottleneck inference loop. Both portraits are independently aligned
using its original square template and 0.85 adjustment. The original demo reads
OpenCV BGR images into the network, so the adapter explicitly converts RGB to BGR
and its generated pixels back to RGB. All 11 readout statistics are local.
The original inference expression that uses source-feature mean for both noise
coefficients is preserved. The result is the generated aligned crop, without
paste-back or Poisson blending.

**UniFace** uses the released swapping `g_ema` and `e_ema` weights. Its official
inference convention resizes each supplied portrait to 256px with PIL bilinear
interpolation and normalizes RGB to `[-1,1]`; the adapter preserves that convention
without adding a face alignment stage. Its custom StyleGAN CUDA operators are
implemented with equivalent PyTorch tensor operations, avoiding a local compiler
requirement. Inference runs in float32. This is the swapping model, not the
separately trained reenactment model.

Each test compares clean and protected sources using the same target, backend,
input transformation and random seed. The independent evaluator measures
FaceNet cosine against the original clean source. Generated images are
quantized to PNG before evaluation. PNG and JPEG95 conditions retain the
original uncompressed identity reference.

A usable unprotected baseline is required for each backend and condition:
the configured numeric gate defaults to **0.80 cosine**, followed by visual
confirmation that the unprotected swap resembles the source. This cutoff is
not a probability or 80% confidence. Weak or undetected baselines remain
**Inconclusive**, and a weak baseline cannot establish stronger protection.

## Provenance and licensing

| Component | Pinned source | Retained notice |
| --- | --- | --- |
| NullSwap reconstruction | [PhantomSeal](https://github.com/LiangqinRen/PhantomSeal/tree/f7705791920abddaa09d0f54d0a2e448b48e07d5), `f7705791920abddaa09d0f54d0a2e448b48e07d5` | [MIT and reconstruction note](../third_party/nullswap/NOTICE.md) |
| DiffSwap | [wl-zhao/DiffSwap](https://github.com/wl-zhao/DiffSwap/tree/c316952f425347a3fd5a7e8bcc624958be34265f), `c316952f425347a3fd5a7e8bcc624958be34265f` | [Source notices and adaptations](../third_party/diffswap/NOTICE.md); upstream has no repository LICENSE at this commit |
| InfoSwap | [GGGHSL/InfoSwap-master](https://github.com/GGGHSL/InfoSwap-master/tree/0484ef5fcb35bf811f2c3ec52ecf77ccade822db), `0484ef5fcb35bf811f2c3ec52ecf77ccade822db` | [CC BY-NC-SA 4.0 and changes](../third_party/infoswap/NOTICE.md), plus the retained MTCNN license |
| UniFace | [xc-csc101/UniFace](https://github.com/xc-csc101/UniFace/tree/05f15e57668e7ce30399233c2456bb0f4cb35055), `05f15e57668e7ce30399233c2456bb0f4cb35055` | [Source notice](../third_party/uniface/NOTICE), including StyleMapGAN CC BY-NC 4.0; no separate upstream repository LICENSE |

The ability to run these local research components does not grant a common
commercial license. The existing FaceShield and iccv25_faceshield credits and
license remain applicable; see [project credits](../README.md#credits-and-licenses).

Runtime weight loading is restricted and begins on CPU. Model state keys are
checked strictly. The UniFace release additionally requires its audited
`argparse.Namespace` metadata type. There is no unrestricted pickle fallback.
The source subsets and source hashes are retained under `third_party/`; model
binaries and installation caches remain excluded from Git.

## Validation scope

Offline verification ran actual NullSwap protection on a public portrait
upscaled to 1280px, with an observed maximum change of 8 pixel levels. This
exercises the full output dimensions, not native high-resolution efficacy.
The generated 512px InfoSwap baseline was detected successfully, but its
FaceNet cosine was 0.649, below the default gate, so that comparison is
inconclusive. Other backends and complete test results are recorded in
[VALIDATION.md](VALIDATION.md). These checks do not require a running server.
