#!/usr/bin/env bash

args=(
    --study-name "NCF-NeuMF-end-2"
    --train-config-path './config/hpo/train_ncf_neumf_end_params.json'
    --hp-config-path './config/hpo/hpo_ncf_neumf_end_params.yaml'
    --train-name 'ncf'
    --n-trials 50
)

python main.py hp-tuning "${args[@]}"
