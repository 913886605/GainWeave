# Data preparation and evaluation schema

The release operates on images supplied by the user. It contains no Laval, Hypersim, sealed evaluation data, private annotations, or recoverable research-data cache.

## Linear HDR input

Place a `dataset.json` file under the input directory:

```json
{
  "schema": "gainweave-input-dataset-v1",
  "color_space": "linear_rgb",
  "rows": [
    {"source_id": "scene_001", "path": "hdr/scene_001.npy", "split": "train"},
    {"source_id": "scene_002", "path": "hdr/scene_002.npz", "split": "dev", "valid_mask": "masks/scene_002.npy"}
  ]
}
```

Paths are relative to the input directory. Each source must stay in one split. Distinct source names cannot be used to place identical RGB content in different roles. If split assignment is omitted, preparation assigns it at source level using the preparation seed.

Supported image containers:

| Container | Required content |
|---|---|
| `.npy` | float32 HWC array with three linear RGB channels; the manifest declares `linear_rgb` |
| `.npz` | RGB array named `hdr_linear_rgb` |
| `.exr`, `.hdr` | Floating-point RGB image |

Values must be finite and nonnegative. Training images have height and width divisible by eight; preparation does not silently resize them. An optional validity mask supplies the valid support for losses and metrics.

```bash
gainweave prepare-data \
  --input-root /path/to/licensed_hdr \
  --output-root /path/to/prepared
```

Preparation creates `manifest.json` and normalized-container samples. Each prepared row records `source_id`, `view_id`, `path`, `split`, `sha256`, `a0`, `positive_log_average`, `positive_luma_count`, and `shape`. Prepared NPZ samples contain `hdr_linear_rgb` and `valid_mask`.

## Exposure and synthetic camera mappings

For each HDR view `R`, compute float32 luminance `Y = 0.2126R_R + 0.7152R_G + 0.0722R_B`. Let `L` be the geometric mean of strictly positive luminances, using float64 for the log accumulation:

```text
L = exp(mean(log(Y[Y > 0])))
a0 = float32(0.18 / L)
T = R × a0 × 2^EV
```

All-black images are rejected. `a0` is a view's fixed exposure anchor; training samples use EV from `[-2, -1, 0, +1, +2]`. There is no per-image maximum normalization.

The two input mappings share the reference `T`:

```text
hard_clip:        S = clip(T, 0, 1)
soft_compression: S = T / (1 + T)
```

Exact sRGB OETF converts `S` for the VAE input. Native losses and metrics compare the prediction with `T`. The default +1 EV presentation is applied separately after inference.

## Paired evaluation

Evaluation reads explicit prediction/reference pairs. An example manifest is:

```json
{
  "schema": "gainweave-paired-evaluation-v1",
  "color_space": "linear_rgb",
  "rows": [
    {
      "source_id": "scene_002",
      "view_id": "view_001",
      "prediction": "predictions/scene_002.exr",
      "reference": "references/scene_002.npz",
      "EV": 0,
      "valid_mask": "masks/scene_002_valid.npy",
      "region_mask": "masks/scene_002_region.npy"
    }
  ]
}
```

Array paths are relative to the paired manifest. Floating-point RGB data follow the same `hdr_linear_rgb` convention; a standalone NPZ mask uses the key `mask`. Optional masks use `0`, `1`, and `255`: validity masks mark invalid, valid, and ignored pixels; region masks mark outside, inside, and ignored pixels. Regions remain undefined when no region mask is supplied.

`EV` is an explicit integer from `[-2, -1, 0, +1, +2]`. Use `view_id` to distinguish multiple views of the same source. The default method label is `GainWeave`; duplicate source/view/EV/method entries are rejected.

The evaluation configuration is:

```yaml
schema: gainweave-eval-v1
manifest: ../examples/paired_manifest.json
output_dir: ../outputs/evaluation
output_name: metrics.json
```

Configuration paths are relative to the configuration file. Edit the paired manifest to point at your own predictions and references, then run:

```bash
gainweave evaluate --config configs/evaluate.yaml
```

Reference and prediction arrays must use the same radiometric convention. Presentation EXRs with an added +1 EV multiplier belong to presentation comparisons and should be identified explicitly when used as numeric predictions.
