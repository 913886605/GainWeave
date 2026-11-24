# Components, attribution, and release scope

Apache-2.0 in the repository and model card covers the original GainWeave package and its exported bridge/highlight tensors. Each upstream component retains its own license and access terms. The model package contains `bridge.safetensors`, `highlight.safetensors`, configuration, and release metadata; upstream foundation weights are obtained from their official repositories.

## Runtime components

| Component | Source and recorded revision | Role and conditions |
|---|---|---|
| Lumina-Image 2.0 | [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/tree/53504abd8178b30685b6c4c7a4cd181ff78b73e9) | Its model card declares Apache-2.0; optional text-to-image generation. The bundled components below retain their own terms. |
| VAE | Lumina `vae/` at the same revision; configuration identifies `black-forest-labs/FLUX.1-dev` | Frozen FP32 feature encoder/decoder for recovery. The local audit records the FLUX-derived provenance; the download command obtains the exact Lumina VAE subfolder. |
| Gemma 2 2B | [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b), bundled as Lumina's text encoder | Required by text generation. [Gemma terms](https://ai.google.dev/gemma/terms) include use and redistribution conditions. |
| FLUX.1-dev provenance | [Official license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/LICENSE.md) | Recorded noncommercial/nonproduction terms apply to the FLUX-dev source scope; separate licenses for particular elements must be established from the source. |
| Diffusers | [huggingface/diffusers](https://github.com/huggingface/diffusers), 0.35.2 | Apache-2.0; VAE and generation interfaces. |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers), 4.53.3 | Apache-2.0; text-encoder interface. |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch), 2.8.0 | BSD-style license; tensor computation and model execution. |

The Lumina top-level model-card tag does not establish that Gemma or the FLUX-derived VAE has been relicensed. Recovery downloads the VAE only; it does not need the complete FLUX pipeline or Gemma text encoder. Text generation obtains the larger Lumina dependency set.

## Research data and comparisons

The original bridge and residual head use the established Laval Indoor HDR train/development roles. Cite Gardner et al., *Learning to Predict Indoor Illumination from a Single Image*, SIGGRAPH Asia 2017, when using the corresponding dataset under its EULA. Dataset files, extracts, derived image collections, private masks, signed forms, and provider correspondence are outside this release.

Historical comparison records reference [X2HDR](https://github.com/kingnobro/X2HDR) and [Vid2HDRImg](https://github.com/chinmay0301ucsd/Vid2HDRImg). They are not runtime dependencies of GainWeave recovery. The pinned X2HDR source and formal LoRA records lacked a repository/model-wide license; their rights status remains separate from their availability for download. No external comparison weights are copied into this model package.

## Release record

The unified internal rights ledger retains open questions about training-data weight-release scope and applicable component conditions.

The public release files contain no signed EULA, names/addresses/signatures from private receipts, private access-account details, raw personal correspondence, anonymous-review mappings, research-data caches, or training recovery state. Model hashes and generic provenance can be retained without publishing those underlying private materials.
