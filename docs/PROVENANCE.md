# Provenance and retained terms

Rani Shield is an adapted local application based directly on **FaceShield: Defending Facial Image against Deepfake Threats**, developed by **Korea University Artificial Intelligence Computer Vision Lab (KUAICV), Department of Artificial Intelligence, Korea University — https://kuaicv.com/**. The original [KUAICV Research-Only License](../LICENSE) remains in the repository. The original README is preserved as [UPSTREAM_README.md](UPSTREAM_README.md).

## FaceShield and iccv25_faceshield credits

Both named repositories are explicitly acknowledged:

- **[FaceShield — Why0912/FaceShield](https://github.com/Why0912/FaceShield)** is *FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models*, built on Bunny. Investigation of that project was the starting point of this development effort. Its RGB/SAV anti-spoofing task differs from image protection, and its trained multimodal model is not integrated in the current application. Its repository carries an MIT notice; that notice does not license the distinct KUAICV protection code.
- **[iccv25_faceshield — kuai-lab/iccv25_faceshield](https://github.com/kuai-lab/iccv25_faceshield)** is the direct implementation foundation of the protection stage. The work by Jaehwan Jeong, Sumin In, Sieun Kim, Hannie Shin, Jongheon Jeong, Sang Ho Yoon, Jaewook Chung and Sangpil Kim appeared at ICCV 2025 as *FaceShield: Defending Facial Image against Deepfake Threats*. Rani Shield adapts that code for local Windows execution, the editor, sequential gradients and output-resolution perturbations.

The identities above were verified against both local Git remotes and the official repository pages. The same name in the two repositories does not imply that they are the same model, authors or license. Neither acknowledgment implies upstream endorsement of Rani Shield.

The KUAICV license permits non-commercial academic, educational and research use subject to its conditions. It expressly requires separate written permission for commercial use, including internal commercial research/product development by a for-profit entity. Third-party code, weights and datasets retain their own terms; an MIT component does not relicense this combined application or its models.

The source snapshot was assembled from upstream FaceShield commit `2eeabde945952c84aaaab3dee7f6821f302ad6a2` plus the local application, integration and validation changes described in [the changelog](../CHANGELOG.md). This is a separate Git repository with its own history; downloaded models and public benchmark image bundles remain local assets.

## Research and implementation sources

| Component | Primary source | Local adaptation and notices |
|---|---|---|
| iccv25_faceshield / KUAICV FaceShield | [Official repository](https://github.com/kuai-lab/iccv25_faceshield), [paper](https://arxiv.org/abs/2412.09921) | Direct protection-code foundation; Windows runtime, lazy/local weights, sequential gradients, full-resolution output delta and smaller differentiable diffusion view; original root license retained |
| PhantomSeal | [Official repository](https://github.com/LiangqinRen/PhantomSeal), [paper](https://arxiv.org/abs/2607.20564) | Full `unify` four-feature objective and 214-photo mixed bank; source MIT notice in `third_party/phantomseal/PHANTOMSEAL_LICENSE` |
| SimSwap | [Official repository](https://github.com/neuralchen/SimSwap) | Minimal generator for real local validation; `simswap_networks.py` retains NVIDIA's CC-BY-NC-SA-4.0 file notice; repository CC-BY-NC-4.0 notice retained in `SIMSWAP_LICENSE` |
| FaceShifter recognition | [PhantomSeal-pinned Research_Project source](https://github.com/richarduuz/Research_Project/blob/d970296cf9d9fe059d729444c757334c9e13a798/ModelC/face_modules/model.py) | Recognition classes only, `view` changed to `reshape`; originating InsightFace_Pytorch MIT notice retained. The containing fork has no root license file; this does not assign a license to its GAN or weights |
| HifiFace interface | [maum-ai/hififace](https://github.com/maum-ai/hififace) | Feature interface from commit `c92db89c7bd0e4d1518f872827fdb69a98f16060`; BSD-3-Clause notice retained |
| IResNet | [InsightFace ArcFace Torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) | Minimal identity backbone; upstream code-license statement and separate pretrained-model restrictions in `LICENSE.insightface-notice.txt` |
| Deep3D | [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) | ResNet50/coefficient heads only; removed unused training, downloads, warping and renderer dependencies; MIT and torchvision BSD notices retained |
| FaceNet | [facenet-pytorch](https://github.com/timesler/facenet-pytorch) | VGGFace2 pretrained encoder for cloak selection and measurement; upstream code/model/data terms apply |
| Stable Diffusion 1.5 | [Model repository](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | Local fp16 UNet, VAE and text encoder; model-card/license terms remain applicable |
| IP-Adapter | [Official code](https://github.com/tencent-ailab/IP-Adapter), [released models](https://huggingface.co/h94/IP-Adapter) | SD 1.5 adapter and CLIP image encoder; upstream code/model terms remain applicable |

Detailed architecture source URLs, imported-source hashes and retained notices are in [`third_party/phantomseal/full/SOURCES.json`](../third_party/phantomseal/full/SOURCES.json) and [`NOTICE.md`](../third_party/phantomseal/full/NOTICE.md). The vendor directory includes separate notices for Deep3D, HifiFace, InsightFace_Pytorch, InsightFace and torchvision. Model weights are not relicensed by those source-code notices.

The browser's canvas interaction model was adapted from the local DeSynth project during development: pan/zoom, comparison modes, exact pixel differences and PNG export. The integration adds model controls, asynchronous local jobs and baseline-gated face-swap testing. This attribution does not assert an independently verified license for that local project's other assets.

## Additional protection and evaluation models

NullSwap uses PhantomSeal's documented best-effort reconstruction, with its released checkpoint and default gain 1, not a claim of original-author reproduction. DiffSwap, InfoSwap and UniFace use their released swap checkpoints and vendored inference code. [NEW_MODELS.md](NEW_MODELS.md) documents preprocessing adaptations, pinned source commits and notices. Separate retained source/license files live under `third_party/nullswap/`, `third_party/diffswap/`, `third_party/infoswap/` and `third_party/uniface/`; observed file hashes live in local `weights/manifests/` records. The optional importer records observed checkpoint hashes, verifies copied bytes, and pins the downloaded InfoSwap statistics by revision, byte count and SHA-256. Runtime models never resolve through the old archive directory.

## Released weights and downloads

The installers use publicly released pretrained weights; no training from scratch is required for these runtime paths. All active resources now share the ignored `weights/` tree, with the complete placement table in [weights/README.md](../weights/README.md). Download/import/migration records are kept under `weights/manifests/`, and optional Hugging Face library downloads use `weights/cache/huggingface/`. Large assets are not part of the source repository.

| Asset | Release source |
|---|---|
| FaceShield ArcFace archive | [Authors' model archive](https://kuaicv.synology.me/weights/iccv2025/FaceShield/models.zip), linked by the official FaceShield README |
| SimSwap generator | [Official checkpoints.zip](https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip); only the 224 px `latest_net_G.pth` is extracted by the small installer |
| SimSwap ArcFace | [Official arcface_checkpoint.tar](https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar) |
| FaceNet VGGFace2 | [Official v2.2.9 release asset](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) |
| PhantomSeal full data | [Official data archive](https://drive.google.com/file/d/1feXBOP0WGemPMpsjeNLKe1j0l1A3Cxjj/view) |
| PhantomSeal full checkpoints | [Official checkpoint archive](https://drive.google.com/file/d/1p-_w1OMEYtph7R3yI188mE-o-xmJ6ya_/view) |

The full PhantomSeal archives contain more material than this feature-only optimization path uses. The runtime needs the full mixed bank, SimSwap identity encoder, FaceShifter recognition backbone and final HifiFace checkpoint. Copying extra GAN weights alone does not add new objective branches; the four integrated branches are explicitly enumerated in [Development](DEVELOPMENT.md#processing-and-memory).

Observed hashes of the smaller released files used by this installation:

The historical runtime inventory, including the full bank and encoder checkpoints, is recorded as relative paths, sizes and observed hashes in [runtime-assets.json](evidence/runtime-assets.json). Its paths describe the layout at the time of that evidence snapshot; the current paths are the unified `weights/` layout. Migration records retain the mapping and verified hashes without changing those historical evidence files. The verified Python package inventory is in [environment-windows.txt](evidence/environment-windows.txt). These inventories contain no private editor inputs or absolute workstation paths.

| File | Bytes | SHA-256 |
|---|---:|---|
| SimSwap `latest_net_G.pth` | 220243703 | `24caf144e9aabd5acd1127b06f13ed3528240adb9e747d77a94ac3f33e672330` |
| SimSwap `arcface_checkpoint.tar` | 209280521 | `52ea5ce4902017b77a2bb811dd9a82f57dee3883e06c18a42d288437263c7a20` |
| FaceNet VGGFace2 | 111898327 | `281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b` |

Download manifests record observed hashes and source URLs. These provide local integrity/reproduction records, not publisher-signed authenticity guarantees. The small `download_phantomseal.py` installer uses TLS verification, temporary files and bounded retries; its generator archive extraction selects the expected member instead of extracting arbitrary archive paths. The separate FaceShield base downloader uses temporary download files but does not implement the same retry policy.

The FaceShield archive's `arcface50_checkpoint.tar` and `arcface100_checkpoint.tar` were identical in the local download, with the ArcFace hash above. The application does not infer model depth or independence from filenames. See the [weight caveat](VALIDATION.md#faceshield-weight-caveat).

## Data and evidence

The active PhantomSeal bank is the **214-photo VGGFace2 `mix_full` directory from its official data archive**, stored at `weights/phantomseal/cloaks/mix_full/`. The earlier 24-image SimSwap example collection remains separate as public demo/benchmark material; it is not the runtime's full cloak bank. Its source commit is `bd7b7686a17f41dd11cfcd5d82f7e4c5eb94b780`, with per-file URLs/hashes and an adaptation notice in [`data/phantomseal_cloaks/PROVENANCE.json`](../data/phantomseal_cloaks/PROVENANCE.json). Image rights and dataset terms remain applicable independently of code licenses.

[`benchmark_manifest.json`](../benchmark_manifest.json) identifies the six public source portraits, three fixed targets, conditions, parameters and the detection-based input exclusion. The main `portrait` source comes from FaceShield's supplied `data/test/image.png`; other source/target crops come from SimSwap's public examples. The 1280 px supplement reuses an enlarged version of that portrait rather than introducing a new source identity.

Sanitized numerical evidence is committed under [`docs/evidence/`](evidence/), alongside an [environment package snapshot](evidence/environment-windows.txt). The numerical snapshots omit private user photos and machine-specific filesystem paths. Full public generated-image reports can be retained locally under ignored `results/benchmarks/`. The corrected [validation record](VALIDATION.md) governs interpretation; archived raw scores do not establish protection efficacy.

## Checkpoint loading safeguards

Custom `torch.load` paths load on CPU with `weights_only=True`; the explicit encoder/generator state loaders use strict matching. Hugging Face components use their own `from_pretrained` loaders. The legacy serialized SimSwap ArcFace module uses a static unsafe-global check and an exact allowlist of audited local ArcFace classes and necessary PyTorch layers. The legacy FaceShifter format has bounded metadata inspection before the restricted tensor loader. There is no unrestricted pickle fallback.

The full HifiFace checkpoint is read as tensor state, and only the exact identity/shape prefixes are transferred into the minimal encoders. The runtime does not instantiate the downloaded Lightning training class. These choices reduce the code-execution surface of loading released weights; they do not certify all external files or their provenance.

## Local runtime notices

The local Python installation under `.runtime/python/` is an independent copy of CPython 3.11, with its `LICENSE.txt` retained. The optional Node.js executable under `.runtime/node/` retains the license from the same distribution. Bootstrap manifests record the copied files and observed hashes; their original installation paths are historical records, not locations used for normal execution.

Python package licenses remain with their installed distributions in `.venv/`. These runtime components do not change the KUAICV license on the protection derivative or the individual model/data terms. The older unused research launch scripts and face-parser modules were removed from the active application during refactoring; the original research documentation and source attributions remain available.
