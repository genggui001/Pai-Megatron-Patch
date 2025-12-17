#!/bin/bash
set -xe

bash scripts/qwen3/run_8xH20_my.sh \
A3B \
/mnt/nas/home/genggui/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
/mnt/nas/home/genggui/pretrain_weights/nlp/Qwen3-30B-A3B-Thinking-2507-to-mcore-mtp  \
false \
true \
bf16 2>&1 | tee to-mcore.log
