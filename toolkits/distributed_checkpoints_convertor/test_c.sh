#!/bin/bash
set -xe

bash scripts/qwen3/run_A22B_16xH20_my.sh \
A22B \
/mnt/nas/home/genggui/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5 \
/mnt/nas/home/genggui/pretrain_weights/nlp/Qwen3-235B-A22B-Thinking-2507-to-mcore-mtp  \
false \
true \
bf16 2>&1 | tee to-mcore.log
