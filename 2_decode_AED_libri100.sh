#!/bin/bash
# 167, 249 (best checkpoints)
SAVE_DIR=data
LS_ROOT=/DB 
CHECKPOINT_FILENAME=avg_last_10_checkpoint_errlog044.pt
# CHECKPOINT_FILENAME=avg_158-167_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
for SUBSET in dev-clean dev-other test-clean test-other; do
  fairseq-generate ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 40 --scoring wer
done

# --checkpoint-upper-bound 167 \
