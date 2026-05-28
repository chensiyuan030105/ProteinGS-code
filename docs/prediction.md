# Prediction

ProteinGS predictions are launched through the batch wrapper `inference.py`,
which calls the installed `boltz predict` command with ProteinGS experiment
settings.

## Install

```bash
pip install -e ".[cuda]"
```

For CPU-only environments:

```bash
pip install -e .
```

## Input

Inputs are YAML files describing biomolecular targets. Small examples are
available under `examples/`.

```text
examples/prot.yaml
examples/multimer.yaml
examples/ligand.yaml
examples/affinity.yaml
```

## Batch Inference

Run:

```bash
python inference.py --config config/boltz-gs-AAV.yaml
```

Each config controls:

- `model`: underlying model family
- `input_dir`: directory of YAML inputs
- `output_dir`: output root
- `diffusion_samples`: number of sampled structures
- `seeds`: random seeds to evaluate
- `gpus`: GPU IDs used by the scheduler
- `use_potentials`: whether to enable potential-based guidance
- `cache`: local model/cache directory
- `checkpoint`: optional checkpoint path
- `gamma_0`, `noise_scale`, `step_scale`, `sampling_steps`: sampling settings
- `pdb_id_list`: optional target subset file

Most config files were created for internal experiments. Update absolute paths,
GPU IDs, cache directories, and checkpoint paths before running them elsewhere.

## Output

For each target and seed, the wrapper writes results under:

```text
<output_dir>/<target>/seed_<seed>/boltz_results_<target>/
```

The wrapper skips a target/seed pair when a non-empty prediction directory is
already present.

## Direct Command

You can also call the underlying CLI directly:

```bash
boltz predict examples/prot.yaml \
  --model boltz1 \
  --diffusion_samples 5 \
  --sampling_steps 2 \
  --out_dir output/example \
  --use_msa_server
```

Add `--checkpoint ./checkpoints/checkpoint.ckpt` when using a local ProteinGS
checkpoint.
