#!/bin/bash
set -xe

export KV_REPEAT_COUNT=4


bash scripts/qwen35/run_8xH20_new.sh \
A3B \
/mnt/nas/home/genggui/pretrain_weights/nlp/Qwen3.5-35B-A3B \
/mnt/nas/home/genggui/pretrain_weights/nlp/qwen35_35b_a3b-to-mcore-ng-8-mtp  \
false \
true \
bf16 2>&1 | tee to-mcore_qwen35_35b_a3b.log


