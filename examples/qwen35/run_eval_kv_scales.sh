#!/bin/bash
set -e
ENV=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-260226:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

if [ $ENV = dsw ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### BASE CONFIG ###
MODEL_SIZE=$2
BATCH_SIZE=$3
GLOBAL_BATCH_SIZE=$4
SEQ_LEN=$5
PAD_LEN=$6
PR=$7
### BASE CONFIG ###

### PARALLEL OPTION ###
TP=${8}
PP=${9}
CP=${10}
ETP=${11}
EP=${12}
SP=${13}
FL=${14}
### PARALLEL OPTION ###

### OTHERS ###
EVAL_ITERS=${15}
DATASET_PATH=${16}
PRETRAIN_CHECKPOINT_PATH=${17}
OUTPUT_BASEPATH=${18}
PATCH_TOKENIZER_PATH=${19}
### OTHERS ###

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    attn_backend_option=" \
        --attention-backend flash
    "
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    attn_backend_option=" \
        --attention-backend fused
    "
fi

if [ $MODEL_SIZE = A3B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    NUM_LAYERS=80
    INTERMEDIATE_SIZE=5120
    MOE_INTERMEDIATE_SIZE=512
    MAX_POSITION_EMBEDDINGS=262144
    NUM_KEY_VALUE_HEADS=8
    ROPE_THETA=10000000
    NUM_EXPERTS=256
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-shared-expert-gate \
        --moe-shared-expert-intermediate-size ${MOE_INTERMEDIATE_SIZE} \
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    hybrid_model_options=" \
                --hybrid-attention-ratio 0.125 \
                --hybrid-mlp-ratio 0.5 \
                --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*- \
                --is-hybrid-model \
                --mamba-state-dim 128 \
                --mamba-head-dim 128 \
                --mamba-num-groups 16 \
                --mamba-num-heads 32
    "
elif [ $MODEL_SIZE = A10B ]; then
    HIDDEN_SIZE=3072
    NUM_ATTENTION_HEADS=32
    NUM_LAYERS=96
    INTERMEDIATE_SIZE=7680
    MOE_INTERMEDIATE_SIZE=1024
    MAX_POSITION_EMBEDDINGS=262144
    NUM_KEY_VALUE_HEADS=8
    ROPE_THETA=10000000
    NUM_EXPERTS=256
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-shared-expert-gate \
        --moe-shared-expert-intermediate-size ${MOE_INTERMEDIATE_SIZE} \
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    hybrid_model_options=" \
                --hybrid-attention-ratio 0.125 \
                --hybrid-mlp-ratio 0.5 \
                --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*- \
                --is-hybrid-model \
                --mamba-state-dim 128 \
                --mamba-head-dim 128 \
                --mamba-num-groups 16 \
                --mamba-num-heads 64
    "
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

if [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_option=" \
		    --sequence-parallel"
elif [ $SP = false ]; then
    sp_option=" \
                    "
fi

if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi
    uneven_split_option=" \
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    uneven_split_option=""
fi

load_option=" \
        --load $PRETRAIN_CHECKPOINT_PATH"

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
      --reset-position-ids \
      --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

dataset_options=" \
    --data-path ${DATASET_PATH} \
    --patch-tokenizer-path ${PATCH_TOKENIZER_PATH} \
    --split 99,1,0 \
    --dataloader-type external \
    --dataset ENERGON"

##### Prepare output dir #######
mkdir -p "${OUTPUT_BASEPATH}"

megatron_options="  \
        --save ${OUTPUT_BASEPATH} \
        --lr 1e-6 \
        --min-lr 1e-7 \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters 1 \
        --lr-warmup-iters 0 \
        --train-iters 1 \
        --eval-iters ${EVAL_ITERS} \
        --save-interval 999999 \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --mtp-num-layers 1 \
        --mtp-steps 2 \
        --mtp-loss-scaling-factor 0.1 \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --eval-interval 1 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --pipeline-model-parallel-layout ${PARALLEL_LAYOUT} \
        --context-parallel-size ${CP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 4 \
        --padded-vocab-size 248320 \
        --patch-tokenizer-type Qwen3Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --distributed-timeout-minutes 60 \
        --ckpt-format torch_dist \
        --transformer-impl transformer_engine \
        --cross-entropy-loss-fusion \
        --qk-layernorm \
        --kv-channels 256 \
        --rotary-percent 0.25 \
        --apply-layernorm-1p \
        --moe-expert-fake-type fake_int4 \
        --moe-expert-fake-group-size 128 \
        --moe-expert-fake-activation-type fp8_e4m3 \
        --attention-fake-kv-quantization-type fp8_e4m3 \
        --train-mode finetune \
        --eod-mask-loss
        "

run_cmd="torchrun $DISTRIBUTED_ARGS eval_kv_scales.py
 ${megatron_options} ${dataset_options} ${pr_options} ${load_option} \
 ${sp_option} ${moe_options} ${packing_options} \
 ${uneven_split_option} ${attn_backend_option} ${tie_option} ${gqa_options} ${hybrid_model_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
