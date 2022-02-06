# Reproducing results

training
```bash
sudo CUDA_VISIBLE_DEVICES=2 nohup python train.py -data data/USPTO-50K/USPTO-50K \
                -save_model experiments/checkpoints/USPTO-50K/transformer \
                -seed 42 -gpu_ranks 0 -save_checkpoint_steps 10000 -keep_checkpoint 20 \
                -train_steps 250000 -param_init 0 -param_init_glorot -max_generator_batches 32 \
                -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0 -accum_count 4 \
                -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 \
                -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
                -layers 6 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                -dropout 0.1 -position_encoding -share_embeddings \
                -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                -heads 8 -transformer_ff 2048 >> ~/transformer.out 2>&1 &

sudo CUDA_VISIBLE_DEVICES=2 nohup python train.py -data data/USPTO-50K/USPTO-50K -save_model experiments/checkpoints/USPTO-50K/transformer -seed 42 -gpu_ranks 0 -save_checkpoint_steps 10000 -keep_checkpoint 20 -train_steps 250000 -param_init 0 -param_init_glorot -max_generator_batches 32 -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0 -accum_count 4 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 -report_every 1000 -layers 6 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer -dropout 0.1 -position_encoding -share_embeddings -global_attention general -global_attention_function softmax -self_attn_type scaled-dot -heads 8 -transformer_ff 2048 >> ~/transformer.out 2>&1 &

tail -f transformer.out
```


average weights
```bash
sudo python tools/average_models.py -models \
		experiments/checkpoints/USPTO-50K/transformer_step_160000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_170000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_180000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_190000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_200000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_210000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_220000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_230000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_240000.pt \
		experiments/checkpoints/USPTO-50K/transformer_step_250000.pt \
		-output experiments/checkpoints/USPTO-50K/transformer_16w-25w.pt
```


translate
```bash
sudo CUDA_VISIBLE_DEVICES=1 python translate.py -gpu 0 \
			-model experiments/checkpoints/USPTO-50K/transformer_16w-25w.pt \
			-src data/USPTO-50K/src-test.txt -output experiments/results/transformer.txt \
			-batch_size 64 -replace_unk -max_length 200 -beam_size 50 -n_best 50
```



evaluate
```bash
sudo python score_predictions.py -targets data/USPTO-50K/tgt-test.txt -beam_size 50 -invalid_smiles \
			-predictions experiments/results/transformer.txt
```


# Results

```bash
Top-1: 56.4% || Invalid SMILES 1.82%
Top-2: 67.5% || Invalid SMILES 8.47%
Top-3: 70.9% || Invalid SMILES 13.92%
Top-4: 72.9% || Invalid SMILES 17.99%
Top-5: 74.2% || Invalid SMILES 21.06%
Top-6: 75.0% || Invalid SMILES 23.33%
Top-7: 75.7% || Invalid SMILES 25.21%
Top-8: 76.4% || Invalid SMILES 26.88%
Top-9: 76.8% || Invalid SMILES 28.30%
Top-10: 77.2% || Invalid SMILES 29.52%
Top-11: 77.5% || Invalid SMILES 30.68%
Top-12: 77.8% || Invalid SMILES 31.65%
Top-13: 78.2% || Invalid SMILES 32.55%
Top-14: 78.5% || Invalid SMILES 33.39%
Top-15: 78.7% || Invalid SMILES 34.16%
Top-16: 78.9% || Invalid SMILES 34.88%
Top-17: 79.1% || Invalid SMILES 35.51%
Top-18: 79.4% || Invalid SMILES 36.06%
Top-19: 79.5% || Invalid SMILES 36.65%
Top-20: 79.8% || Invalid SMILES 37.18%
Top-21: 79.8% || Invalid SMILES 37.70%
Top-22: 80.0% || Invalid SMILES 38.16%
Top-23: 80.1% || Invalid SMILES 38.61%
Top-24: 80.5% || Invalid SMILES 39.02%
Top-25: 80.6% || Invalid SMILES 39.40%
Top-26: 80.7% || Invalid SMILES 39.77%
Top-27: 80.8% || Invalid SMILES 40.12%
Top-28: 80.9% || Invalid SMILES 40.47%
Top-29: 81.0% || Invalid SMILES 40.80%
Top-30: 81.1% || Invalid SMILES 41.12%
Top-31: 81.2% || Invalid SMILES 41.45%
Top-32: 81.2% || Invalid SMILES 41.73%
Top-33: 81.2% || Invalid SMILES 42.03%
Top-34: 81.2% || Invalid SMILES 42.30%
Top-35: 81.3% || Invalid SMILES 42.54%
Top-36: 81.4% || Invalid SMILES 42.76%
Top-37: 81.4% || Invalid SMILES 43.01%
Top-38: 81.4% || Invalid SMILES 43.26%
Top-39: 81.5% || Invalid SMILES 43.49%
Top-40: 81.5% || Invalid SMILES 43.73%
Top-41: 81.5% || Invalid SMILES 43.94%
Top-42: 81.6% || Invalid SMILES 44.17%
Top-43: 81.6% || Invalid SMILES 44.45%
Top-44: 81.7% || Invalid SMILES 44.70%
Top-45: 81.8% || Invalid SMILES 44.96%
Top-46: 81.8% || Invalid SMILES 45.21%
Top-47: 81.8% || Invalid SMILES 45.46%
Top-48: 81.8% || Invalid SMILES 45.71%
Top-49: 81.8% || Invalid SMILES 45.98%
Top-50: 81.8% || Invalid SMILES 46.33%
```
