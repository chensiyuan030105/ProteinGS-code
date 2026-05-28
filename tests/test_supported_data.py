#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple, List


def test_supported_data(dataset: str, model_type: str) -> Tuple[int, int]:
    """
    Check consistency between YAML input files and supported mmCIF files.

    For a given dataset and model_type, this function:

    1. Looks for YAML files under:
         ./dataset/input/{model_type}/{dataset}/*.yaml
       Each YAML filename stem is treated as a pdb_id: {pdb_id}.yaml

    2. Checks for corresponding mmCIF files under:
         ./dataset/supported_data/{dataset}_mmcif/{pdb_id}.cif

    3. Counts:
         - total number of YAML files
         - how many have a matching .cif file

    4. Prints all pdb_ids that are missing a matching .cif,
       and returns (success_count, total_count).
    """
    yaml_dir = Path("./dataset/input") / model_type / dataset
    cif_dir = Path("./dataset/supported_data") / f"{dataset}_mmcif"

    if not yaml_dir.is_dir():
        raise FileNotFoundError(f"YAML directory not found: {yaml_dir}")

    if not cif_dir.is_dir():
        raise FileNotFoundError(f"mmCIF directory not found: {cif_dir}")

    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    total = len(yaml_files)

    if total == 0:
        print(f"[WARN] No YAML files found in {yaml_dir}")
        return 0, 0

    success = 0
    mismatches: List[str] = []

    print(f"[INFO] Checking supported data for dataset='{dataset}', model_type='{model_type}'")
    print(f"[INFO] YAML dir : {yaml_dir}")
    print(f"[INFO] mmCIF dir: {cif_dir}")
    print(f"[INFO] Total YAML files: {total}")

    for yaml_path in yaml_files:
        pdb_id = yaml_path.stem  # e.g. 1abc, 5sak

        cif_path = cif_dir / f"{pdb_id}.cif"
        if cif_path.is_file():
            success += 1
        else:
            mismatches.append(pdb_id)

    # Print mismatches
    if mismatches:
        print("\n[MISMATCH] YAML files without matching .cif:")
        for pdb_id in mismatches:
            print(f"  - {pdb_id} (missing {cif_dir / (pdb_id + '.cif')})")
    else:
        print("\n[MISMATCH] None. All YAML files have matching .cif files.")

    print("\n[SUMMARY]")
    print(f"  Success: {success} / {total}")

    return success, total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test whether supported mmCIF files match YAML inputs."
    )
    parser.add_argument("dataset", type=str, help="Dataset name (e.g. dsDNA_Protein)")
    parser.add_argument(
        "model_type",
        type=str,
        help="Model type (used in ./dataset/input/{model_type}/{dataset})",
    )

    args = parser.parse_args()

    success_count, total_count = test_supported_data(args.dataset, args.model_type)
    # If you want an easy-to-parse last line, you can uncomment this:
    # print(f"{success_count}/{total_count}")
