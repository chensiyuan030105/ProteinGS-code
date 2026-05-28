# Training

ProteinGS training uses the Hydra entry point in `scripts/train/train.py` and
configuration templates under `scripts/train/configs/`.

## Configuration

The most relevant templates are:

- `scripts/train/configs/structure_boltz-gs.yaml`
- `scripts/train/configs/structure_boltz-1.yaml`
- `scripts/train/configs/confidence.yaml`

Before launching a run, update:

- dataset paths
- MSA paths
- checkpoint paths
- output directory
- GPU count and precision
- crop sizes, batch size, and worker count

Several defaults in these files reflect internal experiments and should be
treated as examples rather than portable settings.

## Launch

For a quick local check:

```bash
python scripts/train/train.py scripts/train/configs/structure_boltz-gs.yaml debug=1
```

For a full run:

```bash
python scripts/train/train.py scripts/train/configs/structure_boltz-gs.yaml
```

## Checkpoints

Training checkpoints are not included in this repository. If you want to resume
or initialize from a checkpoint, set the relevant `pretrained`, `resume`, and
`pretrained_path` fields in the config to files available on your machine.

## Preprocessing

The preprocessing scripts live in `scripts/process/`. They are useful when
building local datasets from raw structure and MSA files:

```bash
cd scripts/process
pip install -r requirements.txt
```

You will also need the external tools expected by those scripts, such as
`mmseqs` for sequence clustering/search.
