# Minimal PhantomSeal ensemble architectures

- `faceshifter.py`: recognition backbone selected from PhantomSeal-pinned Research_Project/ModelC (commit d970296cf9d9fe059d729444c757334c9e13a798). Its InsightFace_Pytorch architecture originates from TreB1eN; the MIT text is retained. Only recognition classes remain. Flatten uses reshape for noncontiguous tensors. The containing FaceShifter fork has no root license file; this notice does not assign a new license to its GAN or pretrained weights.
- `iresnet.py`: official InsightFace ArcFace Torch backbone. The upstream README declares its code MIT; that license statement and its separate pretrained-model restrictions are retained verbatim in LICENSE.insightface-notice.txt.
- `deep3d.py`: ResNet50 and coefficient-head architecture reduced from Deep3DFaceRecon_pytorch networks.py. Unused recognition, warping, training, network-download and pickle-loading code was removed. This does not require kornia or BFM resources. Deep3D MIT and torchvision BSD notices are retained.
- `LICENSE.hififace.txt` records the BSD3 notice of the HifiFace implementation whose feature extraction interface is reproduced by the adapter.

The exact four-branch preprocessing and loss configuration comes from PhantomSeal src/unify/base.py and src/unify/defense.py. All model weights retain their upstream terms. No license in this folder changes the repository license.
