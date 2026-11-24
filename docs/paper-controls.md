# Trained decoder-feature controls

One, RGB-only and Five are three separately trained predictors. Each has an 8,192-update bridge and a 2,048-update highlight head. Table IV reports both the bridge output (H0) and the final output (H). The 14.30% hard-clip lost-channel reduction uses final H for Five and RGB-only.

| Architecture | Bridge input | Bridge + head parameters | Recovery VAE |
|---|---|---:|---|
| One | Last decoder upsampling block, 128 channels | 42,917 + 2,769 | Frozen Lumina VAE |
| RGB-only | Three linear SDR channels | 42,852 + 2,769 | None |
| Five | Five frozen decoder feature maps | 42,788 + 2,769 | Frozen Lumina VAE |

With the package installed from this release, the following commands fetch the required public weights and recover the same RGB8 image. Choose an unused output directory for each run.

```bash
python -m gainweave.paper_controls --architecture one \
  --input examples/input.png --output outputs/one
python -m gainweave.paper_controls --architecture rgb \
  --input examples/input.png --output outputs/rgb
gainweave recover --input examples/input.png --output outputs/five
```

The controls write **anchor.exr** (H0) and **native.exr** (H), both uncalibrated relative-linear FP32 RGB. The One/RGB loader verifies the pinned manifest and component digests, then loads each arm's own pair of weights. One also verifies the same upstream recovery VAE used by Five; RGB-only downloads no VAE. The pretrained [control files](https://huggingface.co/Noel2211/GainWeave/tree/b972bb16640226b2fd910d2e6556b0544ba95849/controls) contain model tensors only. The [manifest](https://huggingface.co/Noel2211/GainWeave/blob/b972bb16640226b2fd910d2e6556b0544ba95849/controls/manifest.json) gives checkpoint identity, layout, parameter counts and SHA-256. Existing Five weights are available at [v0.1.0](https://huggingface.co/Noel2211/GainWeave/tree/v0.1.0).

For offline recovery, set `--model /path/to/hub-snapshot --local-files-only`. The local snapshot root must contain `controls/manifest.json` and the relevant `controls/one` or `controls/rgb` subdirectory. For One you can additionally set `--vae-source /path/to/Lumina-Image-2.0/vae`. Both controls support `--device cpu` for inference; the paper used FP32 CUDA on an RTX 4090. The ordinary image interface pads nonmultiples of eight by reflection and crops the returned predictions.

## Continuous inputs for Laval

The RGB8 command is appropriate for VDS. Laval evaluation supplies the prequantization tensors directly. For example, given the paper's linear hard-clipped or soft-compressed view as an HWC float32 NumPy array `s` in [0,1] with height and width divisible by eight:

```python
import numpy as np
import torch
from gainweave.io import linear_to_srgb
from gainweave.paper_controls import PaperControl

model = PaperControl.from_pretrained("one", device="cuda")
def nchw(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).cuda()
output = model.forward_tensors(nchw(linear_to_srgb(s)), nchw(s))
h0, h = output["H0"], output["H_new"]
```

The main `GainWeave.from_pretrained()` exposes the same two tensor-output keys for Five. Run the paper scoring protocol separately; VDS Table III fits its exposure multiplier using CAL scenes, and Table IV uses native relative-linear predictions.
