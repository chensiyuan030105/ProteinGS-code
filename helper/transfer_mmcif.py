#!/usr/bin/env python3
import shutil
from pathlib import Path


# Source mmCIF directory containing all {pdb_id}.cif files
SRC_MMCIF_DIR = Path("./dataset/supported_data/mmcif")

# Datasets to process
DATASETS = [
    "dsDNA_Protein",
    "RNA_Protein",
    "RecentPDB",
]


def copy_for_dataset(dataset: str) -> None:
    """
    For a given dataset, copy mmCIF files according to the list of YAML files.

    Expected layout:
        YAML files:
            ./dataset/input/{dataset}/{pdb_id}.yaml

        Source mmCIF files:
            ./supported_data/mmcif/{pdb_id}.cif

        Target mmCIF files:
            ./{dataset}_mmcif/{pdb_id}.cif
    """
    yaml_dir = Path("./dataset/input/boltz") / dataset
    dest_dir = Path(f"./dataset/supported_data/{dataset}_mmcif")

    if not yaml_dir.is_dir():
        print(f"[WARN] YAML dir not found for dataset {dataset}: {yaml_dir}")
        return

    # Create target directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Collect all YAML files to determine which PDB IDs to copy
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"[WARN] No YAML files found in {yaml_dir} for dataset {dataset}.")
        return

    copied = 0
    missing = 0

    print(f"\n[INFO] Processing dataset: {dataset}")
    print(f"[INFO] YAML dir : {yaml_dir}")
    print(f"[INFO] Source   : {SRC_MMCIF_DIR}")
    print(f"[INFO] Target   : {dest_dir}")

    for yaml_path in yaml_files:
        # PDB ID is the YAML file stem (filename without extension)
        pdb_id = yaml_path.stem  # e.g. 1abc, 5sak

        src_cif = SRC_MMCIF_DIR / f"{pdb_id}.cif"
        dst_cif = dest_dir / f"{pdb_id}.cif"

        if not src_cif.is_file():
            # Source CIF does not exist; report and skip
            print(f"[WARN] Missing source cif for {pdb_id}: {src_cif}")
            missing += 1
            continue

        # Copy CIF file, preserving metadata (timestamps, etc.)
        shutil.copy2(src_cif, dst_cif)
        copied += 1

    total = len(yaml_files)
    print(f"[SUMMARY] {dataset}: YAML={total}, copied={copied}, missing_cif={missing}")


def main():
    """
    Run the copy routine for all datasets listed in DATASETS.
    """
    for dataset in DATASETS:
        copy_for_dataset(dataset)


if __name__ == "__main__":
    main()
