#!/usr/bin/env bash

args=(
    --study-name "NCF-NeuMF-end"
    --train-config-path './config/hpo/train_ncf_neumf_end_params.json'
    --hp-config-path './config/hpo/hpo_ncf_neumf_end_params.yaml'
    --train-func 'train_ncf'
    --n-trials 100
)

python main.py hp-tuning "${args[@]}"