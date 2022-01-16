#!/usr/bin/env bash

DATASET=TG
MODEL=NeuMF-end

args=(
    --run-script $0
    --dataset-name $DATASET
    --model-name $MODEL
    --lr 1e-4
    --decay 0
    --emb-dropout 0
    --dropout 0
    --factor-num 64
    --num-layers 2
    --num-ng 1
    --num-epochs 60
    --train-batch-size 128
    --test-batch-size 256
    --early-criterion 'n20'
    --seed $1
    --early 10
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name 'NeuMF-end'
    # --tags "study_name" "NCF-MLP-2"
    --run-name $MODEL
    # --use-layer-norm
    --item-init-emb-path "./data/$DATASET/node_feat_128dim.npy"
    --normalize-item-init-emb
    # --run-id "fdf03d4d35f246dbbdf2a9cc3e191c12"
    # --GMF-run-id '05be061737e6440c90615044d28eec67'
    # --MLP-run-id '0f1ba47276d941cc9fca71b5b032eeaf'
    # --mode 'eval'
)

python main.py train-ncf "${args[@]}"
