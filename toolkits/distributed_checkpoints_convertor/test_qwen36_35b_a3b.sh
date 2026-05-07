#!/bin/bash
set -xe

export KV_REPEAT_COUNT=4


bash scripts/qwen35/run_8xH20_new.sh \
A3B \
/mnt/nas/home/genggui/code/llm-compressor/tmp/Qwen3.6-35B-A3B-awq-smoothed-merged \
/mnt/nas/home/genggui/pretrain_weights/nlp/qwen36_35b_a3b_smoothed-to-mcore-ng-8-mtp  \
false \
true \
bf16 2>&1 | tee to-mcore_qwen36_35b_a3b.log


