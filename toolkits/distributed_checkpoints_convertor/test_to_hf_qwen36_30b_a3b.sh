#!/bin/bash
set -xe

bash scripts/qwen35/run_8xH20_35_to_hf.sh \
A3B \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v19_3_35b_a3b_next_gemini_bf16/outputs_kl_fake_nvfp4/checkpoint/finetune-kl-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-kl-0.05-bs-1-gbs-8-seqlen-65536-pr-bf16-tp-8-pp-3-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best/  \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v19_3_35b_a3b_next_gemini_bf16/outputs_kl_fake_nvfp4/checkpoint/finetune-kl-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-kl-0.05-bs-1-gbs-8-seqlen-65536-pr-bf16-tp-8-pp-3-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best-hf/  \
true \
true \
bf16 \
/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v19_3_35b_a3b_next_gemini_bf16/outputs_kl_fake_nvfp4/checkpoint/finetune-kl-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-kl-0.05-bs-1-gbs-8-seqlen-65536-pr-bf16-tp-8-pp-3-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best/ 2>&1 | tee  to-hf.log


