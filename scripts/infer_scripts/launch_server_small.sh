#!/bin/bash

# VLLM 单实例启动脚本
# 用于在单个实例上启动 VLLM 服务

set -euo pipefail
set -x

# 必须指定的配置
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"gpt-5-codex-qwencoder_model_0"}
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-"/tmp/baseline/test"}
HDFS_MODEL_PATH=${HDFS_MODEL_PATH:-"/mnt/bn/tiktok-mm-5/aiic/users/guqingshui/serving_on_merlin/models/Qwen3-235B-A22B"}


# 并行配置
TP_SIZE=${TP_SIZE:-2}
DP_SIZE=${DP_SIZE:-1}
DP_SIZE_LOCAL=${DP_SIZE_LOCAL:-1}
WORKERS_PER_INSTANCE=${DP_SIZE:-2}

# 模型配置
MAX_MODEL_LEN=${MAX_MODEL_LEN:-80000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}

# 功能开关
ENABLE_EP=${ENABLE_EP:-False}
ENABLE_DP=${ENABLE_DP:-False}
ENABLE_TOOL=${ENABLE_TOOL:-True}
ENABLE_LONG_CONTEXT=${ENABLE_LONG_CONTEXT:-False}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
ENABLE_REASON=${ENABLE_REASON:-False}


# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="${SCRIPT_DIR}/utils"
# 其余变量配置
VLLM_ENV_PATH=${VLLM_ENV_PATH:-/tmp/VLLM_V14_env}
COPY_TO_LOCAL=${COPY_TO_LOCAL:-"True"}
SERVER_PORT=${SERVER_PORT:-$PORT0}
export MODEL_TYPE
export VLLM_ENV_PATH
export SERVED_MODEL_NAME
export LOCAL_MODEL_PATH
export HDFS_MODEL_PATH
export COPY_TO_LOCAL

export TP_SIZE
export DP_SIZE
export DP_SIZE_LOCAL
export WORKERS_PER_INSTANCE

export MAX_MODEL_LEN
export GPU_MEM_UTIL

export ENABLE_EP
export ENABLE_DP
export ENABLE_TOOL
export ENABLE_LONG_CONTEXT
export ENFORCE_EAGER
export ENABLE_REASON

# export REASON_PARSER
# export TOOL_PARSER
export UTILS_DIR
export SERVER_PORT
bash $UTILS_DIR/kill_vllm.sh || true
# install vllm env
bash $UTILS_DIR/install_vllm_uvenv.sh

# 从 HDFS 复制模型（如果需要）
bash $UTILS_DIR/get_hdfs_model.sh

build_base_config_args() {
    # Build the arguments string
    local args="--tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --host $ARNOLD_WORKER_0_HOST \
    --port $SERVER_PORT \
    --served-model-name $SERVED_MODEL_NAME \
    --async-scheduling"
    if [[ $MODEL_TYPE == qwen35* ]]; then
        args+="   --mm-processor-cache-type "shm""
    fi
    
    if [[ $ENABLE_LONG_CONTEXT == "True" ]]; then
    # vllm serve Qwen/Qwen3-8B --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 
        args+="  --hf-overrides.rope_parameters.rope_theta 1000000.0"
        args+="  --hf-overrides.rope_parameters.factor 4.0"
        args+="  --hf-overrides.rope_parameters.original_max_position_embeddings 32768"
        args+="  --hf-overrides.rope_parameters.rope_type yarn"
    fi
   
    if [[ $ENFORCE_EAGER == "True" ]]; then
        args+= "  --enforce-eager"
    fi
    if [[ $ENABLE_EP == "True" ]]; then
        args+="  --enable-expert-parallel"
    fi

    if [[ $ENABLE_REASON == "True" ]]; then
        args+="  --reasoning-parser $REASON_PARSER"
    fi
    if [[ $MAX_MODEL_LEN -gt 0 ]]; then
        args+="  --max-model-len $MAX_MODEL_LEN"
    fi

    if [[ $ENABLE_TOOL == "True" ]]; then
        args+="  --enable-auto-tool-choice"
        args+="  --tool-call-parser $TOOL_PARSER"
    fi

    echo "$args"
}

bash $UTILS_DIR/up_and_bind_psm.sh

ARGS=$(build_base_config_args)
source "${VLLM_ENV_PATH}/bin/activate"
vllm serve "$LOCAL_MODEL_PATH" $ARGS


# if [[ $ARNOLD_ID -eq 0 ]]; then
#     #launch vllm-server
#     bash $UTILS_DIR/launch_server.sh &

#     #install and launch vllm-router
#     bash $UTILS_DIR/install_vllm_router.sh
#     bash $UTILS_DIR/launch_router.sh &
# else
#     bash $UTILS_DIR/launch_server.sh
# fi
