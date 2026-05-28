# Evaluation

This repository includes helper scripts for checking ProteinGS predictions and
aggregating physical-validity metrics.

## Available Utilities

- `tests/test_output.py` checks whether expected prediction folders exist for a
  dataset and model.
- `tests/test_evaluation.py` summarizes per-seed prediction/evaluation files.
- `tests/test_supported_data.py` checks whether YAML inputs have matching
  reference structures.
- `scripts/eval/aggregate_evals.py` contains aggregation utilities inherited
  from the underlying structure-prediction codebase.
- `extern/PhysProtein/physcialsim_metrics.py` computes physical-validity checks
  for generated structures.

## Data Layout

The evaluation scripts expect local experiment outputs and benchmark metadata.
These files are not part of the public repository. A typical local layout is:

```text
dataset/
  input/
  supported_data/
output/
  <model>/<dataset>/<target>/seed_<seed>/
evaluation/
  aggregate_input/
  aggregate_results/
```

Many files under `evaluation/aggregate_input/` are small templates used for
internal experiments. Update paths and dataset names before running them on a
new machine.

## Example Checks

```bash
python tests/test_output.py CASP15 boltz boltz-gs
python tests/test_supported_data.py CASP15 boltz
```

For physical-validity checks:

```bash
python extern/PhysProtein/physcialsim_metrics.py \
  --tool boltz-gs \
  --num_samples 20 \
  --model_dir ./output/boltz-gs/CASP15 \
  --output_csv ./evaluation/eval_results_PhysProtein/boltz-gs/CASP15/physical_checks.csv \
  --output_txt ./evaluation/eval_results_PhysProtein/boltz-gs/CASP15/successful_pdb_ids.txt
```

The exact command depends on how your local outputs are organized.
