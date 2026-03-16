#!/bin/bash
set -xe

export KV_REPEAT_COUNT=1


bash scripts/qwen3_next/run_8xH20_new.sh \
A3B \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/pretrain_weights/nlp/pulse-v18-3-80b-a3b-hf \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/pretrain_weights/nlp/pulse-v18-3-80b-a3b-to-mcore-ng-8-mtp  \
false \
true \
bf16 2>&1 | tee to-mcore.log


