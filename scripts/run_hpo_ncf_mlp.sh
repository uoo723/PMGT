#!/usr/bin/env bash

args=(
    --study-name "NCF-MLP"
    --train-config-path './config/hpo/train_ncf_mlp_params.json'
    --hp-config-path './config/hpo/hpo_ncf_mlp_params.yaml'
    --train-name 'ncf'
    --n-trials 50
)

python main.py hp-tuning "${args[@]}"
