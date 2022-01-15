#!/usr/bin/env bash

DATASET=VG
MODEL=DCN

args=(
    --run-script $0
    --dataset-name $DATASET
    --model-name $MODEL
    --lr 1e-4
    --decay 1e-3
    --emb-dropout 0.2
    --dropout 0.4
    --factor-num 32
    --deep-net-num-layers 3
    --cross-net-num-layers 3
    --num-ng 5
    --num-epochs 40
    --train-batch-size 256
    --test-batch-size 256
    --early-criterion 'auc'
    --seed $1
    --early 10
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name 'DCN'
    # --tags "study_name" "NCF-MLP-2"
    --run-name $MODEL
    --use-layer-norm
    # --item-init-emb-path './data/VG/node_feat5_128dim.npy'
    # --normalize-item-init-emb
    # --run-id "fe007ec1e5004ce1acb65213bb0cddd3"
    # --GMF-run-id '05be061737e6440c90615044d28eec67'
    # --MLP-run-id '0f1ba47276d941cc9fca71b5b032eeaf'
    # --mode 'eval'
)

python main.py train-dcn "${args[@]}"
