# Recorded development results

The reported fixed endpoint is the 2,048-update bounded highlight head, called `C_SMOOTH` in the research record and `gainweave` in this release. The evaluation uses 218 development sources, 867 views, and five exposures per view: 4,335 cases per method for each input mapping. These are equal-source macro averages in the original relative HDR convention.

`anchor` is R3-B/update8192. `fixed` is the zero-training mild enhancement `H0 × 2^(0.25w)`. `T_MIX` is an earlier 42,788-parameter bridge retrained with mixed hard-clip and soft-compressed inputs. `T_MIX` is retained here as a comparison; the v0.1.0 model files contain the anchor bridge and bounded residual head.

## Full development: hard clipping

Full-image and bright-region MAE use native RGB log1p values. Over/under terms partition the reference-bright error by signed reconstruction direction. The q99.9 column is the absolute log1p error of the luminance quantile; the amplitude column compares bright/midtone luminance ratios in log2. Lower is better for every column.

| Method | Full-image MAE | Bright-region MAE | Over term | Under term | q99.9 log error | Bright/midtone log2-amplitude error |
|---|---:|---:|---:|---:|---:|---:|
| anchor | 0.0281395 | 0.6873954 | 0.0483264 | 0.6390690 | 0.9036925 | 1.4172006 |
| T_MIX | 0.0440270 | 0.6614608 | 0.1793254 | 0.4821354 | 0.8292833 | 1.2568767 |
| fixed | 0.0336647 | 0.6540130 | 0.0814099 | 0.5726030 | 0.8502512 | 1.3262161 |
| gainweave | 0.0293308 | 0.6616750 | 0.0663321 | 0.5953429 | 0.8604528 | 1.3387546 |

Relative to `fixed`, GainWeave reduces full-image MAE by 0.0043339 and the bright over term by 0.0150778. Bright-region MAE rises by 0.0076620 and the under term by 0.0227398. The learned head therefore trades some reference brightness recovery for restraint. Relative to the anchor, it lowers bright-region MAE by 0.0257204, with a 0.0011913 increase in full-image MAE.

## Full development: soft compression

| Method | Full-image MAE | Bright-region MAE | Over term | Under term | q99.9 log error | Bright/midtone log2-amplitude error |
|---|---:|---:|---:|---:|---:|---:|
| anchor | 0.1064014 | 1.1989442 | 0.0000003 | 1.1989439 | 1.7028087 | 2.6064789 |
| T_MIX | 0.0312313 | 0.2749227 | 0.0414543 | 0.2334684 | 0.3383990 | 0.5630752 |
| fixed | 0.1038602 | 1.1482377 | 0.0000031 | 1.1482345 | 1.6350810 | 2.4580484 |
| gainweave | 0.1040509 | 1.1527692 | 0.0000005 | 1.1527686 | 1.6395693 | 2.4722399 |

On this mapping, `T_MIX` recovers reference HDR amplitude more strongly. GainWeave improves bright-region MAE over the anchor by 0.0461750, while its full-image and bright-region MAEs are 0.0001907 and 0.0045315 above `fixed`. The learned residual does not establish a consistent additional advantage over the fixed mild enhancement across both input mappings.

## Generated examples and +1 EV presentation

The generated-image review covers 24 fixed images from 12 prompts and two seeds, with four native versions per image. Shared viewing applied the same exposure and RGB Reinhard → exact sRGB transform to each version. The original inspection covered all overview groups at EV 0 and +1, plus six preselected native-resolution details.

The inspected white surfaces, lamp cores, and reflections were more restrained with GainWeave than with `T_MIX`. GainWeave, anchor, and `fixed` were visually close; existing brightness variation in walls, tiles, and backgrounds remained visible. The inspection did not identify a clear, stable additional visual advantage over `fixed`.

The author accepted +1 EV for the overall presentation. This is an overall author assessment. Assistant-assisted inspection and author review are recorded separately from independent human evaluation; they have not been converted into per-image human scores. Generated images have no paired HDR reference.

Native EXRs have multiplier 1. The explicitly marked +1 EV presentation EXRs have multiplier 2. The presentation setting is not included in the native reconstruction tables, and `2H0` is an exposure control rather than an additional learned method.

## Scope and reproducibility

These results describe the fixed development evaluation and generated examples. The repeated development sources retain their development role. A source-weighted result, per-EV result, or leave-one-source-out result can change the direction of a comparison; the full research record preserves those breakdowns. Region annotations cover their original views only and are not propagated to unannotated images.

The published numeric values are transcribed from the sealed 2026-11-16 final report and its research conclusions. This release preparation does not rerun training or evaluation. The software's evaluation command computes metrics for explicit pairs supplied by the user; it does not include the private research datasets.
