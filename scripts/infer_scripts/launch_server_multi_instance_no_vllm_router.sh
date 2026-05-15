#!/bin/bash

# VLLM 单实例启动脚本
# 用于在单个实例上启动 VLLM 服务

set -euo pipefail
set -x


# 必须指定的配置
MODEL_TYPE=${MODEL_TYPE:?Missing env var MODEL_TYPE. Options: minimax21|deepseek_v32|glm47|k2-thinging|k25|qwen3coder|qwen3-235b-5m|qwen3-235b-7m|qwen35*}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"gpt-5-codex-qwencoder_model_0"}
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-"/tmp/baseline/test"}
HDFS_MODEL_PATH=${HDFS_MODEL_PATH:-"/mnt/bn/tiktok-mm-5/aiic/users/guqingshui/serving_on_merlin/models/Qwen3-235B-A22B"}


# 并行配置
TP_SIZE=${TP_SIZE:-8}
DP_SIZE=${DP_SIZE:-2}
DP_SIZE_LOCAL=${DP_SIZE_LOCAL:-1}
WORKERS_PER_INSTANCE=${DP_SIZE:-2}

# 模型配置
MAX_MODEL_LEN=${MAX_MODEL_LEN:-120000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}

# 功能开关
ENABLE_EP=${ENABLE_EP:-True}
ENABLE_DP=${ENABLE_DP:-True}
ENABLE_TOOL=${ENABLE_TOOL:-True}
ENABLE_LONG_CONTEXT=${ENABLE_LONG_CONTEXT:-False}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
ENABLE_REASON=${ENABLE_REASON:-False}

# # 工具配置
# TOOL_PARSER=${TOOL_PARSER:-"hermes"}
# REASON_PARSER=${REASON_PARSER:-"kimi_k2"}


# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="${SCRIPT_DIR}/utils"
# 其余变量配置
VLLM_ENV_PATH=${VLLM_ENV_PATH:-/tmp/VLLM_V14_env}
COPY_TO_LOCAL=${COPY_TO_LOCAL:-"True"}
SERVER_PORT=${SERVER_PORT:-$PORT4}

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

if [[ $ARNOLD_ID -eq 0 ]]; then
    #launch vllm-server
    bash $UTILS_DIR/launch_server.sh &

    #install and launch vllm-router
    # bash $UTILS_DIR/install_vllm_router.sh
    # bash $UTILS_DIR/launch_router.sh &
    bash $UTILS_DIR/up_and_bind_psm.sh
else
    bash $UTILS_DIR/launch_server.sh
fi
