#!/usr/bin/env bash

DATASET=VG
MODEL=GMF

args=(
    --run-script $0
    --dataset-name $DATASET
    --model-name $MODEL
    --lr 1e-4
    --decay 1e-2
    # --emb-dropout 0.7
    # --dropout 0.5
    --factor-num 8
    --num-ng 1
    --num-epochs 20
    --train-batch-size 256
    --test-batch-size 256
    --early-criterion 'n20'
    --seed $1
    --early 3
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name 'NCF'
    --run-name $MODEL
    # --GMF-run-id 'd6ef4672307347f6b7f24542c21aaeb8'
    # --MLP-run-id '5074f22dc6c943eca416877e80f47ddd'
    --mp-enabled
)

python main.py train-ncf "${args[@]}"
