#!/bin/bash
set -xe

export KV_REPEAT_COUNT=4


bash scripts/qwen35/run_8xH20_new.sh \
A10B \
/mnt/nas/home/genggui/pretrain_weights/nlp/Qwen3.5-122B-A10B \
/mnt/nas/home/genggui/pretrain_weights/nlp/qwen35_122b_a10b-to-mcore-ng-8-mtp  \
false \
true \
bf16 2>&1 | tee to-mcore_qwen35_122b_a10b.log


