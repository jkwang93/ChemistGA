#!/usr/bin/env bash
dataset_all=USPTO-50K_all # USPTO-50K_all
python preprocess.py -train_src data/${dataset_all}/src-train.txt \
                     -train_tgt data/${dataset_all}/tgt-train.txt \
                     -valid_src data/${dataset_all}/src-val.txt \
                     -valid_tgt data/${dataset_all}/tgt-val.txt \
                     -save_data data/${dataset_all}/${dataset_all} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab

dataset_rm=USPTO-50K_rm # USPTO-50K_rm
python preprocess.py -train_src data/${dataset_rm}/src-train.txt \
                     -train_tgt data/${dataset_rm}/tgt-train.txt \
                     -valid_src data/${dataset_rm}/src-val.txt \
                     -valid_tgt data/${dataset_rm}/tgt-val.txt \
                     -save_data data/${dataset_rm}/${dataset_rm} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab