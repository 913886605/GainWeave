# Third-party components

GainWeave's original code is licensed under Apache-2.0. Its exported bridge and
highlight parameter files are separate from the frozen upstream models.

- [Lumina-Image 2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) supplies the
  optional text generator and the separately downloaded recovery VAE. The
  fixed source is documented in `docs/third_party.md` and the model config.
- [Diffusers](https://github.com/huggingface/diffusers) and
  [PyTorch](https://github.com/pytorch/pytorch) supply the runtime implementations.
- Upstream Gemma and FLUX-derived component conditions remain separate; a
  top-level model card does not replace component-specific terms.
- Laval training data, private annotations, original correspondence and
  third-party benchmark models are not redistributed in this repository.

See [the complete component and data notes](docs/third_party.md). The private
research rights ledger retains the outstanding publication/weight-release
questions concerning training data.
