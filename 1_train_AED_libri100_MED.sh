#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train /DB --save-dir data \
  --config-yaml config.yaml --train-subset train-clean-100 --valid-subset dev-clean,dev-other \
  --num-workers 4 --max-tokens 20000 --max-update 30000 --num-update-start-sdt 15000 --beam-sdt 3 --nbest-sdt 3 \
  --task speech_to_text --criterion minimum_edit_distance --label-smoothing 0.1 --report-accuracy --wer-threshold 0.3 \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 4 --ddp-backend legacy_ddp --wandb-project "litsub19_fairseq_AED"
  
#  --wandb-project "Hi to Hi"
# fairseq-train /DB --save-dir data \
#   --config-yaml config.yaml --train-subset train-clean-100,train-clean-360,train-other-500 --valid-subset dev-clean,dev-other \
#   --num-workers 4 --max-tokens 40000 --max-update 300000 \
#   --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
#   --arch s2t_transformer_s --share-decoder-input-output-embed \
#   --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
#   --clip-norm 10.0 --seed 1 --update-freq 8