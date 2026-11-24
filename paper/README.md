# Reproducing the GainWeave paper tables

Run commands from a **v0.2.0 checkout** of the code repository. The [model control guide](../docs/paper-controls.md) gives the One and RGB-only inference commands. The CPU table command uses only the Python standard library. The `paper/` directory contains the registered case lists and case-level MAEs with their original RGB-component counts. The compressed CSVs contain numerical measurements only. Recomputing the reported tables does not require a GPU, model download, or access to licensed photos.

```sh
python paper/reproduce_tables.py --output results/archived
```

This writes full-precision `table_II.csv` (Laval fixed gain and common copying), `table_III.csv` (VDS), `table_IV.csv` (Laval front ends), `gain_scan.csv` and `summary.json`. Checksums are verified before reading the bundled scores. `summary.json` computes the hard-clipped lost-channel reduction as 100 × (RGB MAE − Five MAE) / RGB MAE, separately for the anchor and the learned output. It gives **13.47% at H0** and **14.30% at H**; both ratios use unrounded MAEs. Source labels and case IDs in the score ledgers join directly to [the manifests](manifests/).

## Recompute Laval scores from images and weights

Install the package as described in the root README. Supply your own authorized Laval Indoor HDR panoramas. The view manifest records retained and excluded projections. The paper's 867 retained development views from 218 source panoramas produce 4,335 cases per camera; Q80 is a nested set of 16 views at five exposures.

```sh
python paper/manifests/match_laval.py --root /data/laval --scope dev --output local/sources.csv
python paper/project_laval.py --root /data/laval --source-map local/sources.csv --output local/views
python paper/run_laval.py --views local/views --output results/laval-scores
python paper/reproduce_tables.py --scores results/laval-scores --only II IV --output results/laval-tables
```

The GPU runner loads Five, One and RGB in turn, using each architecture's own bridge and trained head. It reads `<view_id>.npz` with a float32 `hdr_linear_rgb` array, constructs hard-clipped and soft-compressed inputs, and supplies **continuous floating-point** sRGB to the model. It scores native outputs before applying exposure for display. For a quick wiring check, use `--limit-views 1` and a new output directory. For Q80 alone use `--q80-only --arms five`. These subsets emit case rows; population checks require the full set before generating the complete paper tables. The model runners accept `--device`, `--model`, `--vae-source`, and `--local-files-only`.

**Evaluation domains.** For a projected Laval view, expose its linear HDR using 0.18 divided by the geometric mean of its positive luminances, then multiply by the specified power of two. For hard clipping the shared input is S = min(T, 1); for soft compression it is S = T / (1 + T). The hard lost mask uses **T > 1 per channel**. The bright mask uses all three RGB channels at pixels at or above the within-image 99th reference-luminance percentile (linear interpolation, ties included). A mask with no support contributes no value. MAE is the mean absolute difference after applying natural log1p to nonnegative linear RGB predictions and references.

**Aggregation and controls.** Give supported views equal weight at a given exposure within a source panorama, supported exposures equal weight within that panorama, and panoramas equal weight overall. Table II replaces each channel with its observed linear value wherever S < 1 for both copy arms. Over the 21 fixed peak gains, minimize development lost-channel MAE; the registered tie rule uses smallest absolute gain, then smallest signed gain. This gives +0.125 EV. Native Q80 Fixed uses the original +0.25 EV operation. Table IV measures native outputs without channel copying, separately for the anchor (H0) and trained head (H).

## Recompute VDS scores with the public models

Start from the unpacked [author VDS dataset](https://siyeong-lee.github.io/hdr_vds_dataset/). The preparation command matches original input/reference hashes, performs the registered center crop and area resize, and verifies the prepared RGB8 input and linear reference. All methods use the same 512 × 512 crop.

```sh
python paper/manifests/prepare_vds.py --root /data/VDS --output local/vds
python paper/run_vds.py --prepared local/vds --output results/vds-scores
python paper/reproduce_tables.py --scores results/vds-scores --only III --output results/vds-table
```

This regenerates the linearized SDR, Anchor, One, RGB, Five, and both shared-anchor fixed/learned rows. By default it takes **only the four external-method rows** (GMNet, Vid2HDRImg, X2HDR-LDR and LEDiff) from the released per-scene scores. To rescore their predictions too, supply `--external-predictions /path/to/native` with one finite nonnegative linear float32 HWC array per official CAL and TEST input in `<method>/<scene>.npy`, e.g. `gmnet/VDS_t3.npy`. The other directory names are `vid2hdrimg`, `x2hdr_ldr`, and `lediff`. External checkpoints and predictions are separate upstream artifacts. `vds_provenance.json` distinguishes fresh pixel scoring from archived external rows and lists the new calibration scales.

For each method, fit a single nonnegative multiplier using the 20 CAL scenes, pooling all linear-RGB components for least squares. Use these fixed scales for the 48 TEST scenes. The two copied arms share the anchor scale, and copy the linearized input wherever the **resized RGB8 code is below 255** before scaling. Code-255 error averages the 43 TEST scenes with digital-ceiling support. These are counts at the encoded ceiling, not a sensor white-level measurement. The VDS bright mask uses float64 reference luminance in the registered scene-relative scale.

## Score predictions prepared by a different implementation

`paper/score_predictions.py` accepts a local JSONL manifest with one object per case. It checks the exact set of sample IDs, sources, exposures or splits against the published manifests. File paths are relative to the JSONL's directory (absolute local paths also work). Linear RGB arrays are `float32` HWC `.npy` or three-channel `.exr`; VDS inputs are RGB8 PNGs. Use the same registered projections and units. Every Laval record identifies `case`, `source`, integer `ev` and `raw_hdr` (unexposed projected view in linear RGB). Add fields according to the group:

| Group | Additional local inputs |
|---|---|
| `features_hard_clip`, `features_soft_compression` | A `predictions` object mapping `five_H0`, `five_H`, `one_H0`, `one_H`, `rgb_H0`, `rgb_H` to native files. |
| `laval_dev` | `anchor`, `learned`, `weight`; the last is the FP32 H×W weight plane (.npy or the published CLI's gain.npz). |
| `laval_q80` | `fixed_native`, `learned`; the scorer constructs the shared-copy rows from these native predictions. |
| `vds` | `scene`, `split` (CAL or TEST), `input`, `reference`, and `predictions` keyed by `anchor`, `five`, `one`, `rgb`, `gmnet`, `vid2hdrimg`, `x2hdr_ldr`, `lediff`. TEST also supplies the Five `weight` file. |

For example, score the two complete camera manifests and regenerate Table IV:

```sh
python paper/score_predictions.py features_hard_clip local/hard.jsonl --output results/custom
python paper/score_predictions.py features_soft_compression local/soft.jsonl --output results/custom
python paper/reproduce_tables.py --scores results/custom --only IV --output results/custom-table
```

The package's `gainweave evaluate` command remains available for its continuous-weight development metric; use this paper-specific path for Tables II–IV. Display-adjusted EXRs and tone-mapped previews have different units from the native prediction.

## Release verification

The score-to-table command reproduced all 81 printed MAEs (18 in Table II, 33 in III, 30 in IV). For an independent pixel check, we reran the seven VDS model and control rows on the 20 CAL and 48 TEST scenes. Their supported per-scene scores agreed with the archive to within 2.3 × 10⁻¹⁶. On Laval, the raw runner was checked for all three architectures on a retained development view at five exposures under both camera models. The full Laval tables above aggregate the released case scores.
