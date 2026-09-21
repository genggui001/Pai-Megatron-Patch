#!/bin/bash
set -xe

export KV_REPEAT_COUNT=4


bash scripts/qwen35/run_8xH20_new.sh \
A3B \
/mnt/nas/home/genggui/pretrain_weights/nlp/Qwen3.6-35B-A3B \
/mnt/nas/home/genggui/pretrain_weights/nlp/qwen36_35b_a3b-to-mcore-ng-8-mtp-new \
false \
true \
bf16 2>&1 | tee to-mcore_qwen36_35b_a3b.log


