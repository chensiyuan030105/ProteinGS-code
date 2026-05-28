#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import yaml

def main() -> None:
    # ---------------------------
    # Argument parsing
    # ---------------------------
    parser = argparse.ArgumentParser(
        description="Run boltz predict with a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file",
    )
    args = parser.parse_args()

    # ---------------------------
    # Load config
    # ---------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Basic fields
    model = config["model"]
    dataset = config.get("dataset")  # optional, used for path templates

    # Directories (directly use paths from config.yaml)
    input_dir = Path(config["input_dir"])  # directly using Path without resolve_path
    output_dir = Path(config["output_dir"])  # directly using Path without resolve_path

    print("input_dir =", input_dir)

    diffusion_samples = config["diffusion_samples"]
    seeds = config["seeds"]
    gpus = config["gpus"]
    num_gpus = config["num_gpus"]

    use_potentials = config["use_potentials"]
    cache = config["cache"]
    checkpoint = config["checkpoint"]

    gamma_0 = config["gamma_0"]
    noise_scale = config["noise_scale"]
    step_scale = config["step_scale"]
    sampling_steps = config["sampling_steps"]

    # ---------------------------
    # NEW: Read pdb_id_list from txt file if provided
    pdb_id_list_path = config.get("pdb_id_list", None)
    pdb_id_set = None
    if pdb_id_list_path:
        p = Path(pdb_id_list_path)  # directly use Path for pdb_id_list
        if not p.exists():
            raise FileNotFoundError(f"pdb_id_list txt not found: {p}")
        pdb_id_set = set()
        with open(p, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                # accept "7eds ..." and take first token
                pdb_id_set.add(s.split()[0])
        if not pdb_id_set:
            raise ValueError(f"pdb_id_list txt is empty: {p}")

    # ---------------------------
    # Collect input files
    # ---------------------------
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    input_files = sorted(input_dir.glob("*"))
    if not input_files:
        print(f"No input files found in {input_dir}")
        return

    print(f"Found {len(input_files)} input files in {input_dir}")
    print(f"Output root: {output_dir}")
    print(f"GPUs: {gpus} (num_gpus={num_gpus})")
    print()

    # ---------------------------
    # Build job queue (respecting existing results)
    # ---------------------------
    jobs: list[dict] = []  # each job: {"input_file": Path, "basename": str, "seed": int, "out_dir": Path}

    for input_file in input_files:
        print("input_file =", input_file)
        basename = input_file.stem

        # NEW: Skip job if basename is not in pdb_id_list
        if pdb_id_set is not None and basename not in pdb_id_set:
            print(f"[SKIP] {basename} not in pdb_id_list txt")
            continue

        for seed in seeds:
            out_dir = output_dir / basename / f"seed_{seed}"

            # Prediction directory produced by boltz
            pred_dir = out_dir / f"boltz_results_{basename}" / "predictions" / basename

            # Skip if results already exist and are not empty
            if pred_dir.exists() and any(pred_dir.iterdir()):
                print(
                    f"[SKIP] {basename} | seed={seed} -> "
                    f"{pred_dir} already exists and is not empty, skipping."
                )
                continue

            jobs.append(
                {
                    "input_file": input_file,
                    "basename": basename,
                    "seed": seed,
                    "out_dir": out_dir,
                }
            )

    if not jobs:
        print("No jobs to run after skipping existing results.")
        return

    print(f"Total jobs to run: {len(jobs)}")
    print()

    # ---------------------------
    # Dynamic scheduler:
    #   - available_gpus: free GPU IDs
    #   - running: currently running processes
    # ---------------------------
    available_gpus: list[int] = list(gpus)
    running: list[dict] = []  # {"proc": Popen, "log_file": file, "desc": str, "gpu_id": int}
    failed_jobs: list[str] = []

    job_counter = 0  # just for display / indexing

    try:
        # We keep looping while there are pending jobs or running jobs
        while jobs or running:
            # 1) Launch new jobs while we have both free GPUs and pending jobs
            while available_gpus and jobs:
                job_spec = jobs.pop(0)
                input_file = job_spec["input_file"]
                basename = job_spec["basename"]
                seed = job_spec["seed"]
                out_dir = job_spec["out_dir"]

                out_dir.mkdir(parents=True, exist_ok=True)

                gpu_id = available_gpus.pop(0)  # take one free GPU

                log_path = out_dir / f"boltz_{basename}_seed{seed}.log"
                desc = f"{basename} | seed={seed} | GPU={gpu_id}"

                print(f"[JOB {job_counter}] {desc} -> {out_dir} (log: {log_path})")

                cmd = [
                    "boltz",
                    "predict",
                    str(input_file),
                    "--diffusion_samples",
                    str(diffusion_samples),
                    "--out_dir",
                    str(out_dir),
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--override",
                    "--devices",
                    "1",
                    "--use_msa_server",
                    "--gamma_0",
                    str(gamma_0),
                    "--noise_scale",
                    str(noise_scale),
                    "--step_scale",
                    str(step_scale),
                    "--sampling_steps",
                    str(sampling_steps),
                ]

                # Optional flags from config
                if use_potentials:
                    cmd.append("--use_potentials")
                if cache:
                    cmd.extend(["--cache", str(cache)])
                if checkpoint:
                    cmd.extend(["--checkpoint", str(checkpoint)])

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                log_file = open(log_path, "w")
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(out_dir),   # ensure relative outputs go into this out_dir
                )

                running.append(
                    {
                        "proc": proc,
                        "log_file": log_file,
                        "desc": desc,
                        "gpu_id": gpu_id,
                    }
                )
                job_counter += 1

            # 2) If nothing is running and no jobs are left, we are done
            if not running:
                break

            # 3) Poll running jobs to see which ones finished
            time.sleep(1.0)  # small delay to avoid busy-waiting

            still_running: list[dict] = []
            for job in running:
                ret = job["proc"].poll()
                if ret is None:
                    # still running
                    still_running.append(job)
                else:
                    # finished
                    job["log_file"].close()
                    available_gpus.append(job["gpu_id"])  # free this GPU

                    if ret == 0:
                        print(f"[DONE] {job['desc']}")
                    else:
                        print(f"[FAIL] {job['desc']} (return code {ret})")
                        failed_jobs.append(job["desc"])

            running = still_running

        # ---------------------------
        # Final summary
        # ---------------------------
        if failed_jobs:
            print("\n⚠️ Some jobs failed:")
            for d in failed_jobs:
                print("  -", d)
            # Uncomment if you want a non-zero exit code on failure:
            # sys.exit(1)
        else:
            print("✅ All jobs finished successfully.")

    except KeyboardInterrupt:
        print("\n⚠️ Caught KeyboardInterrupt, terminating running jobs...")
        for job in running:
            try:
                job["proc"].terminate()
            except Exception:
                pass
            job["log_file"].close()
        raise


if __name__ == "__main__":
    main()
