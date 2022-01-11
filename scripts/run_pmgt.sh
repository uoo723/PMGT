#!/usr/bin/env bash

DATASET=VG
MODEL=PMGT

args=(
    --run-script $0
    --dataset-name $DATASET
    --model-name $MODEL
    --lr 1e-4
    --decay 1e-2
    --num-epochs 30
    --train-batch-size 32
    --test-batch-size 64
    --early-criterion 'auc'
    --seed $1
    --early 5
    --gradient-max-norm 5.0
    --num-workers 16
    --experiment-name 'PMGT-test'
    --run-name $MODEL
    --mp-enabled
)

python main.py train-pmgt "${args[@]}"
