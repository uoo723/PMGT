#!/usr/bin/env bash

args=(
    --study-name "NeuMF-end-PMGT-6"
    --train-config-path './config/hpo/train_ncf_neumf_pmgt_params.json'
    --hp-config-path './config/hpo/hpo_ncf_neumf_pmgt_params.yaml'
    --train-name 'ncf'
    --n-trials 50
)

python main.py hp-tuning "${args[@]}"
