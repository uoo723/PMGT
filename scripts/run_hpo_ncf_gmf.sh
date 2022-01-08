#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://115.145.135.72:5000

DATASET=VG
MODEL=GMF

args=(
    --study-name "NCF-GMF"
    --train-config-path './config/train_ncf_params.json'
    --hp-config-path './config/hpo_ncf_gmf_params.yaml'
    --train-func 'train_ncf'
)

python main.py hp-tuning "${args[@]}"
