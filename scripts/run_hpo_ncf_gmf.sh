#!/usr/bin/env bash

args=(
    --study-name "NCF-GMF"
    --train-config-path './config/train_ncf_params.json'
    --hp-config-path './config/hpo_ncf_gmf_params.yaml'
    --n-trials 16
    --train-func 'train_ncf'
)

python main.py hp-tuning "${args[@]}"
