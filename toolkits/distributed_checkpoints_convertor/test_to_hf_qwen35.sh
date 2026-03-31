#!/bin/bash
set -xe

bash scripts/qwen3_next/run_8xH20_35.sh \
A10B \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/code/Megatron-Next/model_dir/pulse_v19_2_122b_a10b_next_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-next-A10B-lr-7e-6-minlr-7e-7-bs-1-gbs-14-seqlen-37632-pr-bf16-tp-8-pp-2-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best/  \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/code/Megatron-Next/model_dir/pulse_v19_2_122b_a10b_next_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-next-A10B-lr-7e-6-minlr-7e-7-bs-1-gbs-14-seqlen-37632-pr-bf16-tp-8-pp-2-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best-hf/  \
true \
true \
bf16 \
/inspire/hdd/project/qproject-medicineresearch/public/genggui001/code/Megatron-Next/model_dir/pulse_v19_2_122b_a10b_next_gemini_bf16/outputs/checkpoint/finetune-mcore-qwen3-next-A10B-lr-7e-6-minlr-7e-7-bs-1-gbs-14-seqlen-37632-pr-bf16-tp-8-pp-2-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best/ 2>&1 | tee  to-hf.log


