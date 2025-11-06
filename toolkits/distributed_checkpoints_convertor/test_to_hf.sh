#!/bin/bash
set -xe

bash scripts/qwen3_next/run_8xH20_new.sh \
A3B \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v18_2_80b_a3b_next_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-bs-1-gbs-32-seqlen-16384-pr-bf16-tp-8-pp-4-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8/  \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v18_2_80b_a3b_next_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-bs-1-gbs-32-seqlen-16384-pr-bf16-tp-8-pp-4-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-hf/  \
true \
true \
bf16 \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v18_2_80b_a3b_next_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-bs-1-gbs-32-seqlen-16384-pr-bf16-tp-8-pp-4-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8/


