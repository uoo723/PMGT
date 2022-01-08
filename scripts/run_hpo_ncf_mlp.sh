#!/usr/bin/env bash

args=(
    --study-name "NCF-MLP"
    --train-config-path './config/train_ncf_params.json'
    --hp-config-path './config/hpo_ncf_mlp_params.yaml'
    --train-func 'train_ncf'
)

python main.py hp-tuning "${args[@]}"
