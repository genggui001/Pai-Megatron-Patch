#!/bin/bash
set -xe


export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

bash scripts/qwen3/run_8xH20_my.sh \
A3B \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/code/Megatron-Next/model_dir/pulse_v18_3_30b_a3b_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-moe-megatron-A3B-lr-7e-6-minlr-7e-7-bs-1-gbs-4-seqlen-131072-pr-bf16-tp-8-pp-2-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best/  \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/code/Megatron-Next/model_dir/pulse_v18_3_30b_a3b_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-moe-megatron-A3B-lr-7e-6-minlr-7e-7-bs-1-gbs-4-seqlen-131072-pr-bf16-tp-8-pp-2-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best-hf/  \
true \
true \
bf16 \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/code/Megatron-Next/model_dir/pulse_v18_3_30b_a3b_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-moe-megatron-A3B-lr-7e-6-minlr-7e-7-bs-1-gbs-4-seqlen-131072-pr-bf16-tp-8-pp-2-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best/ 2>&1 | tee to-hf.log


