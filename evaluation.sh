#!/usr/bin/env bash

# get raw evaluation results for different datasets
set -euo pipefail

model_type=boltz
model=boltz-1x
dataset=dsDNA_Protein

echo "=== Evaluating dataset: $dataset ==="

python extern/PXMeter/benchmark/run_eval.py \
    -i "./output/$model/"$dataset"_DEBUG" \
    -o "./evaluation/eval_results_pxmeter/$model/"$dataset"_DEBUG" \
    -d "$dataset" \
    -m "$model_type" \
    -n 128

# aggregate complex lddt results
python extern/PXMeter/benchmark/aggregate/aggregate_complex_lddt.py PoseBusters \
  -t boltz boltz protenix boltz protenix protenix boltz \
  -m boltz-1 boltz-1x protenix boltz-2 protenix-mini-10step protenix-mini-5step boltz-gs \
  -s 101


model=boltz-1
dataset=dsDNA_Protein

PYTHONPATH=. python extern/PXMeter/benchmark/show_intersection_results.py \
    -i ./evaluation/aggregate_input/${dataset}/${model}.json \
    -o ./evaluation/aggregate_results/${dataset}/interface_lddt/raw/${model}/ \
    -p ./evaluation/aggregate_input/${dataset}/pdb_id_list.txt \
    -n 1 \
    --overwrite_agg

dataset=dsDNA_Protein

PYTHONPATH=. python extern/PXMeter/benchmark/show_intersection_results.py \
    -i ./evaluation/aggregate_input/${dataset}/aggregate.json \
    -o ./evaluation/aggregate_results/${dataset}/ \
    -p ./evaluation/aggregate_input/${dataset}/pdb_id_list.txt \
    -n 1 \
    --overwrite_agg

### RELEASE ###
python extern/PXMeter/benchmark/aggregate/aggregate_complex_lddt.py dsDNA_Protein \
  -t boltz boltz protenix boltz protenix protenix boltz \
  -m boltz-1 boltz-1x protenix boltz-2 protenix-mini-10step protenix-mini-5step boltz-gs \
  -s 101

dataset=dsDNA_Protein

PYTHONPATH=. python extern/PXMeter/benchmark/show_intersection_results.py \
    -i ./release/evaluation/aggregate_input/${dataset}/aggregate.json \
    -o ./release/evaluation/aggregate_results/${dataset}/ \
    -p ./release/evaluation/aggregate_input/${dataset}/pdb_id_list.txt \
    -n 1 \
    --overwrite_agg

model=boltz-gs
dataset=dsDNA_Protein_7t19
echo $model
python extern/PhysProtein/physcialsim_metrics.py \
  --tool $model \
  --num_samples 20 \
  --model_dir ./release/output/$model/$dataset \
  --output_csv ./release/evaluation/eval_results_PhysProtein/$model/${dataset}/physical_checks.csv \
  --output_txt ./release/evaluation/eval_results_PhysProtein/$model/${dataset}/successful_pdb_ids.txt

