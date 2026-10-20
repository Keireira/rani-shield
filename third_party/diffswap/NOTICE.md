# DiffSwap inference components

Authors: Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie Zhou and Jiwen Lu.
Paper: **DiffSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware
Masked Diffusion**, CVPR 2023.

Source: <https://github.com/wl-zhao/DiffSwap>, pinned in PROVENANCE.json.
The upstream checkout did not include a repository LICENSE file. The presence
of these research sources or checkpoints does not establish a commercial license.
No blanket license is applied here to the upstream material.

DiffSwap's denoiser, autoencoder and attention modules derive from CompVis latent
diffusion/stable diffusion, OpenAI improved/guided diffusion and taming-transformers;
the ArcFace IResNet derives from InsightFace. Original inline source notices remain.
The unmodified upstream README and model configuration are retained.

Adaptations for Rani Shield:

- Namespaced imports; unused IPython import removed.
- ArcFace autocast uses the current PyTorch API, with the author's CUDA fp16
  setting retained. The similarity fit keeps NumPy's old rcond=-1 behavior.
- Inference-only model assembly with torch.nn.Module replaces Lightning and
  training dependencies. Tensor-only CPU checkpoint loading is strict; no
  unrestricted pickle fallback, downloading, symlinks or working-directory changes.
- The released EMA denoiser is used, matching the author's inference script.
- VQ nearest-code lookup is evaluated in spatial chunks with the same distance
  expression. Gradient checkpointing is bypassed when gradients are disabled.
- Source and target latent images are encoded separately. This follows the
  corrected PhantomSeal runtime wrapper, which fixes the author's original
  dataset script's rolling of target latents into source conditioning.
- Dlib 68-point landmarks, FFHQ geometry, convex hull masks, 112px ArcFace affine
  conversion and multiband output blending follow the author pipeline. Central
  facenet-pytorch MTCNN supplies the five landmarks instead of TensorFlow MTCNN.
  Detection failures are explicit errors, never silently substituted target faces.
- Native output is an aligned 256px crop; there is no claim of native 1280px
  DiffSwap generation. Sampling uses 200 DDIM steps, eta=0 and target guidance
  0.01, matching official inference; PhantomSeal's alternative guidance=0 and
  mask dilation=15 are not the default here.

The multiband blending implementation in the author pipeline credits Greg
Marcil (CS280, UC Berkeley), based on Burt and Adelson's multiresolution spline.
Rani Shield retains its Gaussian/Laplacian kernels and seven-layer construction.

Preprocessing and source-latent correction reference:
<https://github.com/LiangqinRen/PhantomSeal>, pinned in PROVENANCE.json.
The PhantomSeal copyright/license notice is retained in PHANTOMSEAL_LICENSE.
