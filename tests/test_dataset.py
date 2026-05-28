#!/usr/bin/env python3
import sys
from pathlib import Path
import yaml  # Requires PyYAML: pip install pyyaml


def check_dataset(dataset: str):
    """
    For a given dataset name, this script:

    1. Looks for YAML files in:
         ./dataset/input/boltz/{dataset}
    2. For each YAML file, parses its content and finds all `msa` fields
       under `sequences[*].protein.msa`.
       These fields point to CSV paths (e.g.
         ./dataset/msa/boltz/PoseBusters/5sak/5sak_1.csv)
    3. Uses that information to build a table:
         <pdb_id>.yaml <yaml_count> , csv <csv_count> [csv paths...]
       Here, pdb_id is controlled by the YAML filename (first path).
    4. For each msa path, we check whether the CSV file actually exists.
    """

    # Directory that contains YAML files for this dataset
    input_dir = Path("./dataset/input/boltz") / dataset

    # Basic existence check
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    # Collect all YAML files once
    yaml_files = sorted([p for p in input_dir.glob("*.yaml") if p.is_file()])

    print("Summary (per YAML file):")
    print("format: <pdb_id>.yaml <#yaml> , csv <#csv_from_yaml> [msa_paths...]")
    print("-" * 80)

    for yaml_path in yaml_files:
        pdb_id = yaml_path.stem  # pdb_id is controlled by the YAML file name
        yaml_count = 1  # each row is for this single yaml

        # Parse the YAML file
        try:
            with yaml_path.open("r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"{pdb_id}.yaml ERROR parsing yaml: {e}")
            continue

        # Collect msa paths from sequences[*].protein.msa
        msa_paths: list[str] = []
        for seq_entry in data.get("sequences", []):
            if not isinstance(seq_entry, dict):
                continue
            protein_block = seq_entry.get("protein", {})
            if not isinstance(protein_block, dict):
                continue
            msa_path = protein_block.get("msa")
            if msa_path:
                msa_paths.append(msa_path)

        # Unique paths, sorted for stable output
        unique_msa_paths = sorted(set(msa_paths))

        # Check existence of each CSV path (relative to current working directory)
        existing = []
        missing = []
        for p in unique_msa_paths:
            p_obj = Path(p)
            if p_obj.is_file():
                existing.append(p)
            else:
                missing.append(p)

        # Build the output line
        line = f"{pdb_id}.yaml {yaml_count} , csv {len(unique_msa_paths)}"

        if unique_msa_paths:
            line += " ["
            line += ", ".join(unique_msa_paths)
            line += "]"

        print(line)

        if missing:
            print(f"  -> WARNING: missing CSV files:")
            for m in missing:
                print(f"     - {m}")

    print()
    print(f"Input dir (YAML): {input_dir}  -> {len(yaml_files)} yaml files")


if __name__ == "__main__":
    # Expect exactly one argument: dataset name
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} DATASET_NAME")
        sys.exit(1)

    dataset_name = sys.argv[1]
    check_dataset(dataset_name)


