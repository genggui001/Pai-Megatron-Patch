#!/bin/bash
set -xe

bash scripts/qwen3/run_8xH20_my.sh \
A3B \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v18_3_30b_a3b_gemini_bf16/outputs_mtp_only/checkpoint/finetune-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-5-bs-1-gbs-6-seqlen-88064-pr-bf16-tp-8-pp-1-cp-1-ac-full-do-true-sp-true-ti-20480-wi-64/  \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v18_3_30b_a3b_gemini_bf16/outputs_mtp_only/checkpoint/finetune-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-5-bs-1-gbs-6-seqlen-88064-pr-bf16-tp-8-pp-1-cp-1-ac-full-do-true-sp-true-ti-20480-wi-64-hf/  \
true \
true \
bf16 \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v18_3_30b_a3b_gemini_bf16/outputs_mtp_only/checkpoint/finetune-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-5-bs-1-gbs-6-seqlen-88064-pr-bf16-tp-8-pp-1-cp-1-ac-full-do-true-sp-true-ti-20480-wi-64/ 2>&1 | tee to-hf.log


