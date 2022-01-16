#!/usr/bin/env bash

DATASET=TG
MODEL=DCN

args=(
    --run-script $0
    --dataset-name $DATASET
    --model-name $MODEL
    --lr 1e-3
    --decay 1e-3
    --emb-dropout 0.2
    --dropout 0
    --factor-num 16
    --deep-net-num-layers 1
    --cross-net-num-layers 4
    --num-ng 1
    --num-epochs 60
    --train-batch-size 128
    --test-batch-size 256
    --early-criterion 'auc'
    --seed $1
    --early 10
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name 'DCN'
    # --tags "study_name" "NCF-MLP-2"
    --run-name $MODEL-PMGT
    --use-layer-norm
    --item-init-emb-path "./data/$DATASET/node_feat3_32dim.npy"
    --normalize-item-init-emb
    # --run-id "37932c0b158b48c1a18202c2082aab59"
    # --GMF-run-id '05be061737e6440c90615044d28eec67'
    # --MLP-run-id '0f1ba47276d941cc9fca71b5b032eeaf'
    # --mode 'eval'
)

python main.py train-dcn "${args[@]}"
