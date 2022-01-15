#!/usr/bin/env bash

args=(
    --study-name "DCN"
    --train-config-path './config/hpo/train_dcn_params.json'
    --hp-config-path './config/hpo/hpo_dcn_params.yaml'
    --train-name 'dcn'
    --n-trials 50
)

python main.py hp-tuning "${args[@]}"
