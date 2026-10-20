# NullSwap reconstruction provenance

`model.py` is from [LiangqinRen/PhantomSeal](https://github.com/LiangqinRen/PhantomSeal), commit `f7705791920abddaa09d0f54d0a2e448b48e07d5`, path `src/nullswap/model.py`. The MIT license is retained in `LICENSE`.

The upstream model describes itself as a **best-effort reconstruction of NullSwap Figure 2**. This is not verified as the original NullSwap authors' architecture or checkpoint. The local `epoch_0001_step_012000.pth` comes from PhantomSeal's released checkpoint archive.

The architecture is unmodified. Rani Shield adds restricted checkpoint loading, a largest-face aligned 256px view, and inverse mapping of its clipped residual onto the full photograph. Residual gain is **1**, with the released configuration's **8/255** bound. Upstream `Defense.metric` amplifies the residual by 2.75, while `sample` does not; that amplification is deliberately absent from this adapter. A deterministic seed 0 makes the architecture's random noise reproducible on the same device.

This inference adaptation preserves image dimensions, not a guarantee of protection at those dimensions. Generated results require independent face-swap evaluation.
