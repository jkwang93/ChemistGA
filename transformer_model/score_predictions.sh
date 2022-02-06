#!/usr/bin/env bash

dataset=USPTO-50K
model=${dataset}_model_step_50000.pt

python  score_predictions.py -targets data/${dataset}/tgt-test.txt -beam_size 5 -invalid_smiles \
                    -predictions experiments/results/predictions_${model}_on_${dataset}_beam10.txt
