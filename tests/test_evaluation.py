#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def collect_pdb_ids(dataset: str, model_type: str) -> List[str]:
    """
    Collect all pdb_ids for a given dataset and model_type.

    For model_type == "boltz", this assumes:
        ./dataset/input/boltz/{dataset}/*.yaml
    and each YAML file name (without extension) is a pdb_id.
    """
    if model_type != "boltz":
        raise ValueError(f"collect_pdb_ids: model_type '{model_type}' is not supported yet.")

    input_dir = Path(f"./dataset/input/{model_type}") / dataset
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    yaml_files = sorted(p for p in input_dir.glob("*.yaml") if p.is_file())
    pdb_ids = [p.stem for p in yaml_files]

    if not pdb_ids:
        print(f"[WARN] No YAML files found in {input_dir}, dataset may be empty.")

    return pdb_ids


def _parse_model_idx_from_name(name: str, suffix: str) -> Optional[int]:
    """
    Parse model_idx from a file name of the form 'sample_{idx}{suffix}'.
    Returns None if the pattern doesn't match or idx is not an int.
    """
    prefix = "sample_"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None

    middle = name[len(prefix):-len(suffix)]
    try:
        return int(middle)
    except ValueError:
        return None


def get_seed_model_stats_for_pdb(
    model_name: str,
    dataset_name: str,
    pdb_id: str,
    eval_root: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, Dict[str, List[int]]]]:
    """
    For a given pdb_id, return per-seed model_idx success/fail statistics.

    We look under:
        ./evaluation/eval_results/{model_name}/{dataset_name}/{pdb_id}/{seed_id}/

    Each seed directory may contain files like:
        sample_{model_idx}_metrics.json
        sample_{model_idx}_confidences.json

    For each seed:
        - A model_idx is considered SUCCESS if BOTH files exist.
        - If only one exists (or none), it's considered FAIL.

    Returns
    -------
    seeds : list[str]
        List of seed directory names (e.g. ["seed_0", "seed_1", ...]).
    seed_stats : dict[str, dict]
        {
          "seed_0": {
            "success_idxs": [...],
            "fail_idxs":    [...],
          },
          ...
        }
    """
    if eval_root is None:
        eval_root = Path("./evaluation/eval_results")

    pdb_root = eval_root / model_name / dataset_name / pdb_id
    if not pdb_root.is_dir():
        # No evaluation directory at all
        return [], {}

    seed_dirs = [
        d for d in pdb_root.iterdir()
        if d.is_dir()
    ]

    if not seed_dirs:
        # No seed directories
        return [], {}

    seeds = [d.name for d in seed_dirs]
    seed_stats: Dict[str, Dict[str, List[int]]] = {}

    metrics_suffix = "_metrics.json"
    confid_suffix = "_confidences.json"

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name

        metrics_files = list(seed_dir.glob(f"sample_*{metrics_suffix}"))
        confid_files = list(seed_dir.glob(f"sample_*{confid_suffix}"))

        metrics_idxs = set()
        for f in metrics_files:
            idx = _parse_model_idx_from_name(f.name, metrics_suffix)
            if idx is not None:
                metrics_idxs.add(idx)

        confid_idxs = set()
        for f in confid_files:
            idx = _parse_model_idx_from_name(f.name, confid_suffix)
            if idx is not None:
                confid_idxs.add(idx)

        # All model_idx that appear in either metrics or confidences
        all_idxs = sorted(metrics_idxs | confid_idxs)

        success_idxs: List[int] = []
        fail_idxs: List[int] = []

        for idx in all_idxs:
            has_metrics = idx in metrics_idxs
            has_confid = idx in confid_idxs
            if has_metrics and has_confid:
                success_idxs.append(idx)
            else:
                fail_idxs.append(idx)

        seed_stats[seed_name] = {
            "success_idxs": success_idxs,
            "fail_idxs": fail_idxs,
        }

    return seeds, seed_stats


def test_evaluation(dataset: str, model_type: str, model_name: str) -> List[str]:
    """
    Main logic:

    1. Collect pdb_ids for the given dataset/model_type.
    2. For each pdb_id, inspect:
         ./evaluation/eval_results/{model_name}/{dataset}/{pdb_id}/
       and report, per seed:
         - how many model_idx succeeded
         - how many failed
         - which model_idx succeeded/failed
    3. Returns the list of pdb_ids processed (for possible further use).
    """
    if model_type != "boltz":
        raise ValueError(f"test_evaluation: model_type '{model_type}' is not supported yet.")

    pdb_ids = collect_pdb_ids(dataset, model_type)
    return pdb_ids


if __name__ == "__main__":
    # Expect: test_evaluation.py DATASET_NAME MODEL_TYPE_NAME MODEL_NAME
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} DATASET_NAME MODEL_TYPE_NAME MODEL_NAME")
        print(f"Example: {sys.argv[0]} PoseBusters boltz boltz-1")
        sys.exit(1)

    dataset_name = sys.argv[1]
    model_type_name = sys.argv[2]   # e.g. "boltz"
    model_name = sys.argv[3]        # e.g. "boltz-1"

    print(
        f"[INFO] Testing evaluation results for dataset '{dataset_name}' "
        f"with model type '{model_type_name}' and model '{model_name}'..."
    )

    try:
        pdb_ids = test_evaluation(dataset_name, model_type_name, model_name)

        if not pdb_ids:
            print(f"[INFO] Dataset '{dataset_name}' has 0 pdb_ids (no YAML files).")
            sys.exit(0)

        no_seed_pdbs: List[str] = []
        total_pdb_count = len(pdb_ids)

        for pdb_id in sorted(pdb_ids):
            seeds, seed_stats = get_seed_model_stats_for_pdb(
                model_name=model_name,
                dataset_name=dataset_name,
                pdb_id=pdb_id,
            )

            print("=" * 80)
            print(f"PDB ID: {pdb_id}")

            if not seeds:
                print("  total seeds: 0")
                print("  NOTE: no seed directories found for this pdb_id.")
                no_seed_pdbs.append(pdb_id)
                continue

            print(f"  total seeds: {len(seeds)}")

            for seed_name in sorted(seeds):
                stats = seed_stats.get(seed_name, {"success_idxs": [], "fail_idxs": []})
                success_idxs = sorted(stats["success_idxs"])
                fail_idxs = sorted(stats["fail_idxs"])

                total_models = len(success_idxs) + len(fail_idxs)

                print(f"  - {seed_name}:")
                print(f"      total model_idx      = {total_models}")
                print(f"      success models ({len(success_idxs)}): {success_idxs if success_idxs else '[]'}")
                print(f"      failed  models ({len(fail_idxs)}): {fail_idxs if fail_idxs else '[]'}")

                if total_models == 0:
                    print("      NOTE: no sample_*_metrics.json or sample_*_confidences.json files found for this seed.")

        # Global summary
        print("=" * 80)
        print("Global summary:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Model:   {model_name}")
        print(f"  Total pdb_ids checked: {total_pdb_count}")
        print(f"  pdb_ids with NO seeds: {len(no_seed_pdbs)}")
        if no_seed_pdbs:
            print("  These pdb_ids have no seed directories:")
            for pid in no_seed_pdbs:
                print(f"    {pid}")
        print("=" * 80)

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
