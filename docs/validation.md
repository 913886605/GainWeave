# v0.1.0 release validation

The release was exercised on Linux, Python 3.12.3, PyTorch 2.8.0+cu128,
Diffusers 0.35.2, Transformers 4.53.3, NumPy 1.26.4, and an NVIDIA RTX 4090
with 24 GiB VRAM. These are software and numerical acceptance checks, separate
from the development-set results in [results.md](results.md).

| Check | Result |
|---|---|
| CPU suite in an independent environment | 28 tests passed |
| GitHub CPU-only CI, clean checkout and installed package | Passed |
| Wheel and source distribution | Built successfully |
| Installed dependencies | `pip check` passed |
| Original bridge/head to safetensors | All 46 tensors exactly preserved |
| Saved generated 1024×1024 case, migrated anchor and final HDR | Zero differing values; maximum absolute error 0 |
| Actual private HF download, official VAE download, installed recovery | Exact agreement with the saved HDR output |
| 513×769 input | Original size restored, finite output, residual bound and low-luminance exact copy passed |
| New text-to-HDR generation | 1024×1024, 50 steps, BF16 Lumina with model CPU offload; native EXR, preview and presentation output completed |
| Presentation EXR | Exactly twice native float32 values at +1 EV |
| Synthetic training command | One update, eight microbatches; frozen foundation unchanged; final package exported |
| Synthetic paired evaluation command | Exact-zero fixture scored correctly |

The complete migrated code was cloned back from GitHub, installed as a normal
package, and tested outside the original research source tree. The recovery
check used the weights downloaded from Hugging Face rather than the original
training checkpoint. The text-generation test reused the already available
official Lumina base files and the independently downloaded GainWeave/VAE cache.
It also succeeded without torchvision installed.

The two-case recovery acceptance process observed about **3.48 GiB maximum
allocated** and **8.53 GiB maximum reserved** Torch memory. The fresh installed
text run recorded approximately **19.50 seconds of sampling** and **0.52 seconds
of recovery**, excluding model loading and file export. These measurements are
specific to this environment and do not set minimum hardware requirements.

The synthetic update validates the exported training interface; it did not
update the released weights or start another research experiment. No sealed
holdout, SI-HDR, JourneyDB or Final Test was used for this delivery.

## Fixed model artifacts

Model repository: `Noel2211/GainWeave`, tag `v0.1.0`, commit
`1b7a785d4ea7ad40e7170981e602630c077317ea`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `bridge.safetensors` | 174760 | `fda31000786aa2543ffa084e65402ea576e2d00d0657f44b6433eaa82692b188` |
| `highlight.safetensors` | 11436 | `fba0dc1596ae2f28d9a2a948fd79618894acbdc12b8abf33ba3decd66d3f4041` |

Both repositories remain private during submission preparation. Visibility is
changed manually by the author after submission.
