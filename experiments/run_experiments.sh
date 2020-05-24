#!/bin/bash

while getopts m:d:l:c: option
do
    case "$option" in
        m) MODEL=${OPTARG};;
        d) DATASET=${OPTARG};;
        l) L_DIM=${OPTARG};;
        c) N_COMP=${OPTARG};;
    esac
done

if [ $MODEL = "RVAE" ]; then
    $N_COMP = 1
fi

python run.py --model $MODEL --dataset $DATASET --enc_layers 300 300 --dec_layers 300 300 --latent_dim $L_DIM --num_centers 350 --num_components $N_COMP --device cuda
