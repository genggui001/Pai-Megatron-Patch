#!/bin/bash
set -xe

bash scripts/qwen3/run_8xH20_my.sh \
A3B \
/mnt/nas/home/genggui/pretrain_weights/nlp/Qwen3-30B-A3B-Thinking-2507-INIT-MTP \
/mnt/nas/home/genggui/pretrain_weights/nlp/Qwen3-30B-A3B-Thinking-2507-to-mcore-init-mtp  \
false \
true \
bf16 2>&1 | tee to-mcore.log
