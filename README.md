# ProteinGS

Research code for **Physically Valid Biomolecular Interaction Modeling with
Gauss-Seidel Projection**.

ProteinGS studies fast, physically valid biomolecular structure generation by
combining diffusion-based structure prediction with a Gauss-Seidel projection
step. The projection is used to reduce local physical violations while keeping
the predicted structure close to the model output.

This repository is a public code snapshot prepared for the ProteinGS project
page. It contains the model, inference utilities, evaluation helpers, and
experiment configuration templates used during development.

## Status

This is an initial public release. Some experiment-specific assets are not
included:

- trained checkpoints
- private benchmark data
- generated prediction outputs
- local cache directories

Configuration files under `config/` are examples. Before running them, update
dataset paths, output paths, GPU IDs, cache locations, and checkpoint paths for
your machine.

## Installation

Create a fresh Python environment, then install the package in editable mode:

```bash
git clone https://github.com/chensiyuan030105/ProteinGS-code.git
cd ProteinGS-code
pip install -e ".[cuda]"
```

For CPU-only environments, omit the CUDA extra:

```bash
pip install -e .
```

## Inference

The repository includes a small batch launcher:

```bash
python inference.py --config config/boltz-gs-AAV.yaml
```

Each config controls the model variant, input directory, output directory,
random seeds, GPU assignment, diffusion sampling parameters, cache path, and
optional checkpoint path.

Most configs were written for internal experiments and should be treated as
templates rather than plug-and-play examples.

## Repository Layout

```text
config/                 Experiment configuration templates
src/boltz/              Model, data, diffusion, and projection modules
scripts/train/          Training entry points and Hydra configs
scripts/eval/           Evaluation and aggregation utilities
extern/PhysProtein/     Physical validity checks
helper/                 Dataset and result conversion helpers
examples/               Small input-format examples
tests/                  Utility and output checks
```

## Notes

- The package name is still `boltz` for compatibility with the underlying
  model code.
- Checkpoints are expected to be provided separately.
- Large datasets and generated outputs are intentionally excluded from git.
- Some scripts contain local experiment paths and may need cleanup before use
  outside the original environment.

## Acknowledgements

This code builds on the open-source Boltz codebase and extends it with
ProteinGS-specific projection, inference, training, and evaluation utilities.
Please also follow the licenses and citation guidance of the upstream projects
that this work depends on.
