#!/usr/bin/env bash

DATASET=TG
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
    --hidden-size 32
    --gradient-max-norm 5.0
    --num-workers 16
    --experiment-name 'PMGT'
    --run-name $MODEL
    --mp-enabled
    --beta 1.0
    --num-hidden-layers 3
    --valid-size 0.1
    --mode "inference"
    --inference-result-path "./data/$DATASET/node_feat3_32dim.npy"
    --run-id '4add51ee87c2489cba6d6318298b11fe'
)

python main.py train-pmgt "${args[@]}"
