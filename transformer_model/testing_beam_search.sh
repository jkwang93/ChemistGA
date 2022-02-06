#!/usr/bin/env bash

dataset=USPTO-50K
model=${dataset}_model_step_90000.pt
beam_size=10
CUDA_VISIBLE_DEVICES=4 python translate.py -gpu 4 \
                    -model experiments/checkpoints/${dataset}/${model} \
                    -src data/${dataset}/src-test_sorted.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_beam${beam_size}_test_sorted.txt \
                    -batch_size 64 -replace_unk -max_length 200 -beam_size ${beam_size} -n_best ${beam_size}
