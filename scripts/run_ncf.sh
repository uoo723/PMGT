#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://115.145.135.72:5000

DATASET=VG
MODEL=GMF

args=(
    --run-script $0
    --dataset-name $DATASET
    --model-name $MODEL
    --num-epochs 20
    --train-batch-size 256
    --test-batch-size 256
    --early-criterion 'n20'
    --seed $1
    --early 5
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name 'NCF'
    --run-name "GMF"
    --mp-enabled
)

python main.py train-ncf "${args[@]}"
