# Paper sample manifests

These manifests identify the exact views and scenes in Tables II–IV. They contain study IDs, crop directions, file digests, split assignments and official VDS member names. Obtain the images from their respective providers under their own terms. Run the commands below from the code repository root.

## Laval Indoor HDR

- [laval_views.csv](laval_views.csv) lists all 8,060 materialized views from 2,015 panoramas, before exclusions. Each of the three hash-selected directions and the hotspot has its actual yaw and pitch in degrees. Each panorama is bound to its original EXR **file** SHA-256 and byte count. The `path_sha256` field hashes its normalized relative archive-member path; bucket zero of this integer modulo ten is development.
- [laval_excluded_views.csv](laval_excluded_views.csv) lists all 60 hand-reviewed **views** marked ineligible for border-connected camera, tripod or nadir footprint. They span 58 panoramas: 55 training views and five development views. Other orientations of these sources stay eligible. The resulting partitions contain 7,133 training views from 1,797 sources and 867 development views from 218 sources.
- [laval_dev_cases.csv](laval_dev_cases.csv) lists 4,335 hard-clipped and 4,335 soft-compressed cases. Join by `case_id` or by `view_id` with the signed `ev` and `camera` columns. In `V0073_EV0_hard_clip`, the exposure index zero denotes **−2 EV**. The `q80_id` column marks the 80 hard-clipped cases in the top portion of Table II. These use the same 16 development views (five EVs each).

A local copy of the original matching panoramas can be indexed even when filenames or directories have changed. Python 3.11+:

```bash
python paper/manifests/match_laval.py --root /path/to/licensed/panoramas \
  --scope dev --output local/laval_sources.csv
```

Use `--scope q80` for the 16-source subset or `--scope all` for all 2,015 panoramas. The output maps `source_id` to `local_path` relative to the directory supplied. The input directory must contain the unpacked provider EXR files. Re-encoding changes the file digests. `crop_npz_sha256` in the view manifest identifies the original materialized NPZ **container**. Codec and header changes can change that hash even for identical decoded pixels. The `allowlist_index` is the canonical view index used by the paper; excluded views have no such index. Source data: Gardner et al., *Learning to Predict Indoor Illumination from a Single Image*, SIGGRAPH Asia 2017.

## Real-camera VDS

[vds_scenes.csv](vds_scenes.csv) identifies each middle-exposure input and its matched released scene-relative RGBE reference. The 20 CAL scenes are official training scenes, selected in frozen order by SHA-256 of `gainweave-supp2-cal-v1\nVDS_<id>`, with a literal newline and the author scene ID (e.g. `t67`). TEST is all 48 scenes of the official test split. Fit scale on CAL and evaluate on TEST.

Obtain the [VDS author release](https://siyeong-lee.github.io/hdr_vds_dataset/). Unpack the archive so `train_set`, `train_hdr_set`, `test_set` and `test_hdr_set` sit immediately below the root supplied here. Preparation verifies the member-file hashes, applies the common crop `x=[152,760), y=[0,608)` on the 912×608 source, and area-resizes to 512×512. The PNG is filtered as RGB8 code values; the HDR reference is filtered in linear RGB. The code checks the decoded RGB8 sample bytes and the saved little-endian float32 HDR NPY against the study's recorded identities.

```bash
python paper/manifests/prepare_vds.py --root /path/to/VDS --output local/vds512
```

This writes `local/vds512/<scene>/sdr.png` and `linear_hdr.npy`; the scene ID is the join key in the scores. Use `--verify-only` instead of `--output` to check the hashes without writing images. The reference environment used for validation has NumPy 1.26.4, OpenCV 4.9.0 and Pillow 11.3.0. The manifest stores the camera acquisition asset hashes separately from the prepared input and reference hashes. For the digital-ceiling column compute `code==255` on the prepared input; 43 TEST scenes have qualifying channels. Bright and full-image scores cover all 48.

## Verify the manifests without images

```bash
python paper/manifests/verify.py
```

This checks the manifest checksums, excluded-view identities, source partitions, view × EV × mapping keys, Q80 membership, and CAL/TEST split.
