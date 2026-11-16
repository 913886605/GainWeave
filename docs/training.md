# Training recipes

Version 0.1.0 exports the accepted R3-B bridge at update 8,192 and the bounded highlight head at update 2,048. The bridge has 42,788 parameters; the head has 2,769. Training checkpoints and exported inference weights serve different purposes: the release model repository stores the two parameter tensors without the research training sequence, optimizer, RNG, or private activity ledger.

## Original R3-B bridge

R3-B was trained from a fresh initialization on the established Laval training partition. It uses five frozen VAE decoder features and learns 16-channel adapters, multiscale fusion blocks, the head stem, and a four-channel output. Its gated RGB gain reconstruction is given in [the method](method.md).

| Setting | Recorded recipe |
|---|---|
| Initialization seed | 20260911 |
| Precision | FP32 |
| Optimizer updates | 8192 |
| Effective batch | 8, using microbatch 2 and accumulation 4 |
| Optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8` |
| Learning rate | `1e-4`; 512-update linear warmup; cosine to `1e-6` |
| Weight decay | 0.01 for convolution kernels; 0 for biases and GroupNorm |
| Gradient clipping | Global norm 1.0 |
| Training objective | Gate focal BCE + positive-support RGB gain L1 + full-image RGB log1p HDR L1 |

The gate focal BCE uses positive alpha 0.25 and gamma 2.0. The original data roles, source/view sequence, exposure anchors, and checkpoint lineage are retained in the private research record. The accepted bridge is initialized independently of the frozen VAE parameters.

## Released bounded highlight head

The released training command implements the final residual-head stage. It loads the accepted bridge and VAE, freezes both, and trains the 19 → 16 → 1 convolutional head. The output layer starts at zero, so the initial prediction exactly matches the anchor. Decoder and head-stem features are detached ordinary tensors; only the 2,769 head parameters receive gradients.

| Setting | Fixed recipe |
|---|---|
| Seed | 0 |
| Updates | 2048 |
| Effective batch | 8, using microbatch 1 and accumulation 8 |
| Input slots | Four hard-clip and four soft-compressed inputs per update, alternating by slot |
| Exposure values | `[-2, -1, 0, +1, +2]` |
| Precision | FP32 |
| Optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8` |
| Learning rate | `1e-4`; 128-update linear warmup; cosine to `1e-6` |
| Weight decay | 0.01 for convolution kernels; 0 for biases |
| Gradient clipping | Global norm 1.0 after accumulation |
| Residual budget | `delta = 0.5w tanh(z)` EV, shared across RGB |

The objective is:

```text
L = L_HDR + 0.25 L_bright + 0.10 L_grad + 0.05 L_smooth + L_over
```

`L_HDR` is RGB log1p L1 error over valid pixels. `L_bright` weights that error by `(Y_T / (1 + Y_T))²`. `L_grad` compares horizontal and vertical log1p RGB gradients with the reference. `L_smooth` penalizes neighboring differences of the local EV field using continuous edge-aware weights. `L_over` penalizes positive log1p reconstruction error. This objective favors restrained brightening while retaining a bounded ability to reduce brightness.

The original final stage used 16,384 sample presentations, with the fixed 2,048-update endpoint. Updates 512 and 1,024 described the training trajectory; the endpoint was not chosen by comparing their development scores. Full-development results and the fixed-control tradeoffs are in [results](results.md).

## Run on your data

Prepare the linear HDR inputs using [the data schema](data.md). Then use:

```bash
gainweave train \
  --config configs/train_smooth.yaml \
  --data-root /path/to/prepared
```

The configuration follows:

```yaml
schema: gainweave-train-v1
dataset: manifest.json
model:
  pretrained: Noel2211/GainWeave
  revision: v0.1.0
  vae_source: null
  local_files_only: false
  cache_dir: null
output_dir: ../outputs/train
seed: 0
updates: 2048
checkpoint_every: 128
resume: null
```

The dataset manifest is resolved under `--data-root`; configuration paths follow the implementation's configuration directory convention. Existing output directories are protected from overwrite. Resume checkpoints retain optimizer, scheduler, RNG, sequence, and update position so a resumed run continues its selected recipe.

Training on a newly prepared dataset produces a new model. Matching the released research run additionally requires the same licensed data, split, exposure anchors, geometry, and sample sequence. Those private research data are outside the software and model repositories.

The final inference package is written to `<output>/final` and can be loaded with
`gainweave recover --model /path/to/output/final --input ... --output ...`.
To resume, set `resume` to an explicit checkpoint such as
`../outputs/train/checkpoints/update_000128_attempt_0001.pt`; its `.pt.json` SHA sidecar
must accompany it. Resume paths are resolved relative to the configuration file.
