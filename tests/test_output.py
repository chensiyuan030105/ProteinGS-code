#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def collect_pdb_ids(dataset: str, model_type: str) -> List[str]:
    """
    Collect all pdb_ids for a given dataset and model_type.

    For model_type == "boltz", this assumes:
        ./dataset/input/boltz/{dataset}/*.yaml
    and each YAML file name (without extension) is a pdb_id.
    """
    if model_type == "boltz":

        input_dir = Path(f"./dataset/input/{model_type}") / dataset
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")

        yaml_files = sorted(p for p in input_dir.glob("*.yaml") if p.is_file())
        pdb_ids = [p.stem for p in yaml_files]

        if not pdb_ids:
            print(f"[WARN] No YAML files found in {input_dir}, dataset may be empty.")

        return pdb_ids

    elif model_type == "protenix":

        input_dir = Path(f"./dataset/input/{model_type}") / dataset
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")

        yaml_files = sorted(p for p in input_dir.glob("*.json") if p.is_file())
        pdb_ids = [p.stem for p in yaml_files]

        if not pdb_ids:
            print(f"[WARN] No JSON files found in {input_dir}, dataset may be empty.")

        return pdb_ids       

    else:
        raise ValueError(f"collect_pdb_ids: model_type '{model_type}' is not supported yet.")


def get_boltz_seed_info_for_pdb(
    model: str,
    dataset: str,
    pdb_id: str,
    output_root: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """
    For a given pdb_id, return information about its seeds.

    We look under:
        ./output/{model}/{dataset}/{pdb_id}/seed_{seed}/
            boltz_results_{pdb_id}/predictions/{pdb_id}

    Returns
    -------
    all_seeds : list[str]
        Names of all seed directories found (e.g. ["seed_0", "seed_1", ...]).
    successful_seeds : list[str]
        Subset of all_seeds for which 'predictions/{pdb_id}' exists.
    """
    if output_root is None:
        output_root = Path(f"./output/{model}")

    pdb_root = output_root / dataset / pdb_id

    if not pdb_root.is_dir():
        # No output directory at all
        return [], []

    # Find all seed_* directories
    seed_dirs = [
        d for d in pdb_root.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    ]

    if not seed_dirs:
        # No seed_* directories
        return [], []

    all_seeds: List[str] = [d.name for d in seed_dirs]
    successful_seeds: List[str] = []

    for seed_dir in seed_dirs:
        result_dir = seed_dir / f"boltz_results_{pdb_id}" / "predictions" / pdb_id
        if result_dir.is_dir():
            successful_seeds.append(seed_dir.name)

    return all_seeds, successful_seeds

def get_protenix_seed_info_for_pdb(
    model: str,
    dataset: str,
    pdb_id: str,
    output_root: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """
    For a given pdb_id, return information about its seeds.

    We look under:
        ./output/{model}/{dataset}/{pdb_id}/seed_{seed}/
            boltz_results_{pdb_id}/predictions/{pdb_id}

    Returns
    -------
    all_seeds : list[str]
        Names of all seed directories found (e.g. ["seed_0", "seed_1", ...]).
    successful_seeds : list[str]
        Subset of all_seeds for which 'predictions/{pdb_id}' exists.
    """
    if output_root is None:
        output_root = Path(f"./output/{model}")

    pdb_root = output_root / dataset / pdb_id / pdb_id

    if not pdb_root.is_dir():
        # No output directory at all
        return [], []

    # Find all seed_* directories
    seed_dirs = [
        d for d in pdb_root.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    ]

    if not seed_dirs:
        # No seed_* directories
        return [], []

    all_seeds: List[str] = [d.name for d in seed_dirs]
    successful_seeds: List[str] = []

    for seed_dir in seed_dirs:
        result_dir = seed_dir / "predictions"
        if result_dir.is_dir():
            successful_seeds.append(seed_dir.name)

    return all_seeds, successful_seeds

def check_boltz_output_for_pdb(
    model: str,
    dataset: str,
    pdb_id: str,
    output_root: Optional[Path] = None,
) -> bool:
    """
    Check whether a given pdb_id has a successful result under:

        ./output/model/{dataset}/{pdb_id}/seed_{seed}/
            boltz_results_{pdb_id}/predictions/{pdb_id}

    We consider this pdb_id successful if there exists at least one seed_* folder
    such that the 'predictions/{pdb_id}' directory exists.
    """
    all_seeds, successful_seeds = get_boltz_seed_info_for_pdb(
        model=model,
        dataset=dataset,
        pdb_id=pdb_id,
        output_root=output_root,
    )

    # If there are no seeds at all, or no successful seeds, this pdb_id is a failure
    return len(successful_seeds) > 0

def check_protenix_output_for_pdb(
    model: str,
    dataset: str,
    pdb_id: str,
    output_root: Optional[Path] = None,
) -> bool:
    """
    Check whether a given pdb_id has a successful result under:

        ./output/model/{dataset}/{pdb_id}/seed_{seed}/
            boltz_results_{pdb_id}/predictions/{pdb_id}

    We consider this pdb_id successful if there exists at least one seed_* folder
    such that the 'predictions/{pdb_id}' directory exists.
    """
    all_seeds, successful_seeds = get_protenix_seed_info_for_pdb(
        model=model,
        dataset=dataset,
        pdb_id=pdb_id,
        output_root=output_root,
    )

    # If there are no seeds at all, or no successful seeds, this pdb_id is a failure
    return len(successful_seeds) > 0

def test_output(dataset: str, model_type: str, model: str) -> Tuple[List[str], List[str]]:
    """
    Main logic:
    1. Collect pdb_ids for the given dataset/model.
    2. For model == "boltz", check outputs under:
           ./output/model/{dataset}
       using the rules described in check_boltz_output_for_pdb.
    3. Return (success_list, failure_list).
       No printing here; printing is handled in __main__.
    """
    if model_type == "boltz":

        pdb_ids = collect_pdb_ids(dataset, model_type)
        total = len(pdb_ids)

        if total == 0:
            # Let caller decide how to handle this; just return empty lists
            return [], []

        success: List[str] = []
        fail: List[str] = []

        for pdb_id in pdb_ids:
            ok = check_boltz_output_for_pdb(model, dataset, pdb_id)
            if ok:
                success.append(pdb_id)
            else:
                fail.append(pdb_id)

        return success, fail
    
    elif model_type == "protenix":

        pdb_ids = collect_pdb_ids(dataset, model_type)
        total = len(pdb_ids)

        if total == 0:
            # Let caller decide how to handle this; just return empty lists
            return [], []

        success: List[str] = []
        fail: List[str] = []

        for pdb_id in pdb_ids:
            ok = check_protenix_output_for_pdb(model, dataset, pdb_id)
            if ok:
                success.append(pdb_id)
            else:
                fail.append(pdb_id)

        return success, fail

    else:
        raise ValueError(f"test_output: model '{model_type}' is not supported yet.")


if __name__ == "__main__":
    # Expect: test_output.py DATASET_NAME MODEL_TYPE_NAME MODEL_NAME
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} DATASET_NAME MODEL_TYPE_NAME MODEL_NAME")
        print(f"Example: {sys.argv[0]} PoseBusters boltz boltz-1")
        sys.exit(1)

    dataset_name = sys.argv[1]
    model_type_name = sys.argv[2]   # e.g. "boltz"
    model_name = sys.argv[3]        # e.g. "boltz-1"

    print(
        f"[INFO] Testing outputs for dataset '{dataset_name}' "
        f"with model type '{model_type_name}' and model '{model_name}'..."
    )

    try:
        success, fail = test_output(dataset_name, model_type_name, model_name)
        total = len(success) + len(fail)

        if total == 0:
            print(f"[INFO] Dataset '{dataset_name}' has 0 pdb_ids (no YAML files).")
            sys.exit(0)

        # First print which ones succeeded / failed
        print("Successful pdb_ids:")
        if success:
            for pid in success:
                print(f"    {pid}")
        else:
            print("  (none)")
        print()

        print("Failed pdb_ids:")
        if fail:
            for pid in fail:
                print(f"    {pid}")
        else:
            print("  (none)")
        print()

        # 额外输出：每个 pdb_id 有多少个 seed，seed 名字分别是什么
        print("Per-pdb_id seed summary:")
        # 用 set 去重，然后按名字排序一下
        all_pdb_ids = sorted(set(success + fail))
        for pid in all_pdb_ids:
            if model_type_name == "boltz":
                all_seeds, successful_seeds = get_boltz_seed_info_for_pdb(
                    model=model_name,
                    dataset=dataset_name,
                    pdb_id=pid,
                )
            elif model_type_name == "protenix":
                all_seeds, successful_seeds = get_protenix_seed_info_for_pdb(
                    model=model_name,
                    dataset=dataset_name,
                    pdb_id=pid,
                )

            total_seeds = len(all_seeds)
            seeds_str = ", ".join(all_seeds) if all_seeds else "(none)"
            succ_str = ", ".join(successful_seeds) if successful_seeds else "(none)"

            print(f"   -  {pid}:")
            print(f"      total seeds      = {total_seeds}")
            print(f"      all seeds        = {seeds_str}")
            print(f"      successful seeds = {succ_str}")
        print()

        # Put the summary block at the end: success count is printed last
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Model:   {model_name}")
        print(f"Total pdb_ids: {total}")
        print(f"Success: {len(success)} / {total}")
        print("=" * 80)

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


