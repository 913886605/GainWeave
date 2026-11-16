# Method and input/output contract

GainWeave v0.1.0 combines a frozen VAE, a learned HDR bridge, and a bounded residual highlight head. The two released parameter files contain 45,557 parameters: 42,788 in the bridge and 2,769 in the highlight head.

## Frozen decoder features and the anchor

The recovery API accepts an image path, a PIL image, or a uint8 HWC RGB array, interpreted as sRGB. Exact inverse sRGB converts it to linear RGB `S` in `[0,1]`; the VAE receives sRGB mapped to `[-1,1]`. Encoding uses the posterior mode. The implementation preserves the VAE's raw latent domain through its scaling/shift round trip.

Five decoder features have 512, 512, 512, 256, and 128 channels. Each passes through a 16-channel adapter; four fusion blocks combine the scales. A 16-channel full-resolution head predicts a shared gate and three gain magnitudes. With output logits `r`, the anchor is:

```text
gate = sigmoid(r[0])
gain_rgb = softplus(r[1:4])
H0 = S × 2^(gate × gain_rgb)
```

The VAE is frozen. The bridge contains the adapters, fusion blocks, head stem, and output layer; upstream VAE weights are stored separately. Recovery uses FP32 throughout.

## Bounded residual gains

Capture the bridge's 16-channel head-stem feature `F` once at native resolution. Normalize it across channels at each pixel, then concatenate linear RGB `S`:

```text
F_norm = F / sqrt(mean_channel(F²) + 1e-6)
z = Conv1×1(SiLU(Conv3×3(concat(F_norm, S))))
```

The convolutions map 19 → 16 → 1 channels, with biases. The final convolution starts at zero during training. A continuous input-luminance weight limits where the residual acts:

```text
Y = 0.2126 S_R + 0.7152 S_G + 0.0722 S_B
u = clamp((Y − 0.5) / 0.4, 0, 1)
w = u² (3 − 2u)
delta = 0.5 w tanh(z)
H = H0 × 2^delta
```

`delta` is shared across RGB channels and lies within `[-0.5w, +0.5w]` EV. Pixels with `Y ≤ 0.5` copy `H0` exactly. The shared scalar preserves the anchor's native RGB ratios. The residual changes radiometric values; it does not blur the final image or resynthesize texture.

## Variants

| CLI variant | Native HDR |
|---|---|
| `anchor` | `H0` |
| `fixed` | `H0 × 2^(0.25w)` |
| `gainweave` | `H0 × 2^(0.5w tanh(z))` |

The historical `T_MIX` bridge in the development tables is a separately retrained comparison. It is not an alias for the released `gainweave` variant.

## Image size

The reported recovery and generation experiments use 1024 × 1024 images. The recovery API accepts compatible RGB inputs and pads spatial dimensions to a multiple of eight for the VAE, then crops the output to the original size. Compatibility at other sizes is separate from the 1024 × 1024 validation. A supplied image must have at least eight pixels in each dimension.

## Output and display

The canonical Python output is `result["hdr_linear_rgb"]`, a nonnegative float32 HWC RGB array. Native EXR output uses this array directly, with ZIP lossless compression and an exposure multiplier of one.

For a selected presentation exposure `e`, the preview is computed from the native array:

```text
v = 2^e × H
mapped = v / (1 + v)            # per RGB channel
display = exact_sRGB_OETF(mapped)
PNG = nearest_even_RGB8(clip(display, 0, 1))
```

The default `e = +1` is an author-selected presentation setting. `presentation_plus1ev.exr` stores `2 × H`; the default preview applies +1 EV to `H` once. A viewer should avoid applying another +1 EV to that presentation EXR when reproducing the same exposure.

Native values are relative HDR. The model does not provide calibrated physical units, a display luminance in nits, or a common white point for unrelated upstream methods. Photoshop and other EXR viewers can apply different exposure, tone-mapping, and color-management settings.

## Text generation

The optional text path runs the pinned Lumina-Image 2.0 pipeline to obtain an SDR RGB image, then applies the same recovery module. Defaults are 1024 × 1024, 50 steps, guidance 4.0, CFG truncation ratio 0.25, CFG normalization enabled, and seed 0. Generation uses BF16; the VAE used by the recovery stage and both GainWeave modules use FP32. Offloading controls memory placement without changing the release's parameter files. Resource use depends on the selected generation/offload path and hardware.
