#!/usr/bin/env bash

args=(
    --study-name "NCF-GMF-4"
    --train-config-path './config/hpo/train_ncf_gmf_params.json'
    --hp-config-path './config/hpo/hpo_ncf_gmf_params.yaml'
    --n-trials 30
    --train-name 'ncf'
)

python main.py hp-tuning "${args[@]}"
