# InfoSwap provenance and modifications

The model definitions and inference algorithm are from [GGGHSL/InfoSwap-master](https://github.com/GGGHSL/InfoSwap-master), commit `0484ef5fcb35bf811f2c3ec52ecf77ccade822db` (Gao and contributors). The upstream **CC BY-NC-SA 4.0** license is retained in `LICENSE.md`; these adapted files retain that license.

Copied definitions: `modules/encoder128.py`, `modules/aii_generator.py`, `modules/decoder512.py`, and `modules/iib.py`. `convex_upsample.py` contains only the necessary `ConvexUpsample` class from `utils.py`. Alignment files are copied from `preprocess/mtcnn_pytorch/src/align_trans.py` and `matlab_cp2tform.py`; their authorship headers are retained. Changes are package-relative imports and equality comparisons for string values.

`swap_adapters/infoswap.py` adapts the official `inference_demo.py` inference loop: native 512px smooth G/E/I weights, the identity encoder, and all 11 released readout-statistics files are loaded locally with restricted CPU checkpoint loading and strict state matching. Runtime uses the project's local MTCNN detector and the original 0.85-scaled square alignment template. Source and target are aligned independently. Missing detection raises an error; it never substitutes an unchanged image for a failed swap.

The original demo reads OpenCV BGR arrays directly into the network. This adapter explicitly converts RGB to BGR and generated BGR back to RGB. Stochastic bottleneck noise is seeded for repeatable comparisons. The author's source-mean-as-standard-deviation expression is preserved. Output is the actual aligned generated face (512px), without paste-back or Poisson blending.

The discriminator and no-smooth checkpoints are unnecessary for this selected inference variant. Asset hashes and acquisition provenance are recorded locally in `weights/manifests/nullswap-infoswap.json`.
