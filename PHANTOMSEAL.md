# PhantomSeal ensemble → FaceShield

Rani Shield supports **PhantomSeal → FaceShield**, **PhantomSeal only**, and
**FaceShield only**. The default chain runs 1,000 PhantomSeal steps and then
30 FaceShield steps. The original normalized photo remains the comparison
baseline. Both the intermediate PhantomSeal PNG and final PNG can be exported.

## Full original mixed cloak bank

The connected bank is the complete **214-photo** `vggface2_cloak/mix_full`
directory supplied in PhantomSeal's `data.tar.gz`, replacing the earlier
24-example SimSwap starter bank. Selection uses MTCNN and FaceNet VGGFace2,
choosing a distance closest to 1.20 among candidates above 1.04.

The first use builds a CPU index of all detectable bank faces. The cache is
invalidated if any bank image, FaceNet weights, or preprocessing version changes.
Only the selected portrait is loaded during subsequent jobs. The metadata
records the bank's total/usable count, cache fingerprint and selected cloak.

## Full identity ensemble

All four frozen feature branches from PhantomSeal's upstream `unify` objective
participate in every optimization step:

| Branch | Pretrained component |
|---|---|
| SimSwap identity | SimSwap ArcFace |
| FaceShifter identity | IR-SE-50 backbone |
| HifiFace identity | ArcFace IResNet-100 |
| HifiFace shape | Deep3D reconstruction network, first 80 coefficients |

HifiFace's two feature networks are loaded from the final full HifiFace
checkpoint, matching upstream's override of its standalone initialization.
The loss weights, displacement caps and 0.003 update scale follow `unify.yaml`.
The previous SimSwap-only context loss is not added to the ensemble objective.

Gradients are evaluated one branch at a time and summed before one sign update,
avoiding simultaneous activation graphs. All models are frozen, in evaluation
mode, and use float32 input gradients. Tensor state dictionaries are loaded
strictly with `weights_only=True`. SimSwap's serialized module uses the existing
audited exact-class allowlist. No unrestricted pickle fallback is used.

The protection objective uses encoders, so the HifiFace/FaceShifter synthesis
decoders, discriminators, Lightning training wrappers and BFM renderer are not
loaded onto the GPU. This does not omit a feature branch from the objective.

## Local configuration

All active resources are kept under `weights/`. Runtime paths must stay inside
the project directory. `phantomseal_config.json` uses these relative defaults:

```json
{
  "checkpoint_dir": "weights",
  "cloak_dir": "weights/phantomseal/cloaks/mix_full",
  "expected_cloaks": 214
}
```

Put the SimSwap identity checkpoint in `weights/simswap/arcface_checkpoint.tar`,
FaceShifter recognition weights in `weights/faceshifter/model_ir_se50.pth`,
and the full final HifiFace checkpoint in
`weights/hififace/hififace_opensource_299999.ckpt`. FaceNet is stored in
`weights/facenet/facenet_vggface2.pt`; the SimSwap test generator is
`weights/simswap/latest_net_G.pth`. Keep all 214 bank photos in the configured
`mix_full` directory. The runtime reports missing assets instead of silently
reverting to the smaller bank or a partial ensemble. See
[weights/README.md](weights/README.md) for the complete resource table,
download/import steps and local migration command.

Weight audit caveat: the local copy of the archive linked in the authors'
[model setup](https://github.com/kuai-lab/iccv25_faceshield#-model-setup) contains
identical `arcface50_checkpoint.tar` and `arcface100_checkpoint.tar` entries.
Both have SHA-256
`52ea5ce4902017b77a2bb811dd9a82f57dee3883e06c18a42d288437263c7a20`.
This installation therefore does not establish two distinct FaceShield ArcFace
models from those filenames. This duplication does not explain a weak clean
SimSwap baseline: that attack uses its paired SimSwap generator and ArcFace.

## Resolution and output

The local adapter optimizes a full-resolution perturbation through a fixed,
differentiable, aligned **256×256 face view** of the largest detected face.
The output keeps the original proportions and a maximum long edge up to 1280;
smaller images are not enlarged. This preserves the full ensemble objective,
but differs from the paper's pre-cropped-image evaluation protocol.

`results/editor/jobs/<id>/original.png` is the clean normalized baseline,
`stage1/phantomseal.png` is stage one, and `protected.png` is the final result.
There is no intermediate change in image dimensions between the two stages.
PhantomSeal's RGB limits are approximately 19.1/7.7/19.1 levels out of 255;
FaceShield's noise limit applies to stage two. Cumulative changes may exceed
either individual limit. Actual total changes, model paths, branch names and
timings are written to `metadata.json`. GPU models are released between stages.

Sequential processing can strengthen or weaken a particular defense. Image
SSIM/PSNR and surrogate embedding scores do not establish face-swap protection.
`benchmark_chain.py` compares original, first-stage and chained images using a
fixed SimSwap target and FaceNet scoring; a single pair is not a general benchmark.

## Local real-SimSwap comparison (20 October 2026)

`make benchmark` / `manage.cmd benchmark` run the reusable paired suite in
`benchmark_suite.py`. The fixed manifest has six public source portraits,
three distinct prepared SimSwap targets and two input conditions: PNG and
JPEG quality 95 with 4:2:0 subsampling. Each source is protected once using
1,000 PhantomSeal steps and 30 FaceShield steps; its exact intermediate PNG
is compared with its chained PNG. The generator actually synthesizes a face
for all three variants, including the clean baseline. Each attack independently
redetects and aligns its source using SimSwap's five-pose alignment.

The clean original PNG remains the FaceNet identity reference in both input
conditions. A lower cosine describes less similarity under this evaluator;
it is not an attack success probability. Sources, targets, model/code hashes,
parameters, input exclusions, generated swaps and all paired measurements are
saved. Protection cannot be inferred when the unprotected swap already fails
to reproduce the source person's identity.

The earlier unqualified efficacy conclusion based on lower scores in **28/36**
comparisons is withdrawn. Many clean swaps were weak, so a further score drop
cannot establish that protection defeated a successful face swap. A subsequent
**0.80 clean-swap FaceNet cosine** screen leaves only **8/36** numerically
qualified comparisons, representing **two source identities**. This is a local
screening heuristic, not a calibrated identity threshold or a validation of
visual likeness. Those eight comparisons still require visual review; they do
not provide a replacement success rate or support a general efficacy claim.

Original source-weighted means on the six detectable 224–512px sources are
retained below as descriptive raw measurements, including weak baselines:

| Condition | Clean swap | PhantomSeal | PhantomSeal + FaceShield |
|---|---:|---:|---:|
| PNG | 0.6372 | 0.4725 | 0.4460 |
| JPEG95 | 0.6149 | 0.4749 | 0.4169 |

The raw chain score was lower in 28/36 comparisons and in 5/6 source means
when both conditions were included. For `james`, the raw means were 0.5420 for
PhantomSeal versus 0.6505 for the chain. These counts and means do not establish
protection effectiveness. All 108 output faces were detected, which confirms
face generation but does not confirm that the intended identity was copied.
The mean protected-image SSIM also fell from 0.6358 to 0.6251; maximum color
changes rose from 19 to 31/255. The methods have different perturbation budgets.

An additional check reused the earlier full-ensemble **1280×1280** application
test result on three targets and both conditions (18 generated swaps). That
portrait was upscaled from 512px and is not another independent source identity.
The raw PNG clean/PhantomSeal/chain means at this size were
0.8227/0.8078/0.8174; JPEG95 means were 0.8242/0.8094/0.8182. PhantomSeal alone
was lower in 4/6 raw comparisons. These measurements also require a visually
credible clean baseline and do not demonstrate reliable 1280px protection.

## Interpreting Test protection

The UI applies the 0.80 numerical screen separately to PNG and JPEG95.
For each condition, inspect the actual **unprotected** swap and confirm
**The unprotected swap looks like me** only if it visibly retains your identity.
Each checkbox starts unchecked. A weak baseline or an unchecked confirmation
is shown as **Inconclusive**, with the paired delta hidden. The original scores
remain descriptive measurements; neither a high cosine nor ticking a checkbox
turns a single SimSwap example into a general protection guarantee.

These tests cover extraction of the protected photo's **source identity** for
insertion into another face. They do not test replacing the protected photo's
face when it is used as the destination. FaceNet took part in cloak selection,
and SimSwap is itself an ensemble branch; neither is a fully unseen evaluator.
The small sample, reused exploratory portrait and single swapper limit general
conclusions. The darker `lcw` candidate was rejected because the application's
detector found no face before swap scoring; `gy` replaced it after preflight.
No parameters were adjusted from the swap results.

Saved local reports: [six-source comparison](results/benchmarks/2026-10-20-full/index.html)
and [1280px supplement](results/benchmarks/2026-10-20-full/resolution1280/index.html).
Each includes raw JSON/CSV and actual generated images. A new run defaults to
`results/benchmarks/latest/`; verified protection PNGs can be resumed.

## Sources and notices

- PhantomSeal: https://github.com/LiangqinRen/PhantomSeal (MIT).
- Encoder architectures and retained notices: `third_party/phantomseal/full/`.
- SimSwap source: `third_party/phantomseal/simswap_networks.py`, retained for
  local evaluation; its original notices remain beside it.

FaceShield's KUAICV Research-Only License and all third-party model/data terms
remain applicable. Removing the optional UI footer does not change these terms.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node tests/frontend-check.cjs
```
