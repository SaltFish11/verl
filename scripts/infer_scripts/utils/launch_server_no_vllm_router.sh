set -x
source "${VLLM_ENV_PATH}/bin/activate"

# parallel config
TP_SIZE=${TP_SIZE}
DP_SIZE_LOCAL=${DP_SIZE_LOCAL:-1}

ENABLE_EP=${ENABLE_EP:-True}
ENABLE_DP=${ENABLE_DP:-True}
ENABLE_TOOL=${ENABLE_TOOL:-True}
ENABLE_LONG_CONTEXT=${ENABLE_LONG_CONTEXT:-True}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
ENABLE_REASON=${ENABLE_REASON:-False}
ENABLE_TP_ENCODER=${ENABLE_TP_ENCODER:-True}
# model config
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-"/tmp/baseline/Qwen3-235B-A22B"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-120000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}

# tool config
case "$MODEL_TYPE" in
    deepseek_v32)
        TOOL_PARSER=${TOOL_PARSER:-"deepseek_v32"}
        # REASON_PARSER=${REASON_PARSER:-"deepseek_v3"}
        ;;
    glm47)
        TOOL_PARSER=${TOOL_PARSER:-"glm47"}
        REASON_PARSER=${REASON_PARSER:-"glm45"}
        ;;
    qwen3coder)
        TOOL_PARSER=${TOOL_PARSER:-"qwen3_coder"}
        REASON_PARSER=${REASON_PARSER:-"qwen3"}
        ;;
    qwen3-235b-5m|qwen3-235b-7m)
        TOOL_PARSER=${TOOL_PARSER:-"hermes"}
        REASON_PARSER=${REASON_PARSER:-"qwen3"}
        ;;
    k2-thinging|k25|k2-instruct)
        TOOL_PARSER=${TOOL_PARSER:-"kimi_k2"}
        REASON_PARSER=${REASON_PARSER:-"kimi_k2"}
        ;;
    minimax21)
        TOOL_PARSER=${TOOL_PARSER:-"minimax_m2"}
        # REASON_PARSER=${REASON_PARSER:-"minimax_m2"}
        ;;
    *)
        TOOL_PARSER=${TOOL_PARSER:-"kimi_k2"}
        REASON_PARSER=${REASON_PARSER:-"kimi_k2"}
        ;;
esac

WORKERS_PER_INSTANCE=${WORKERS_PER_INSTANCE:-2}
DP_SIZE=$WORKERS_PER_INSTANCE
TOTAL_WORKERS=$ARNOLD_WORKER_NUM
COPY_TO_LOCAL=${COPY_TO_LOCAL:-"False"}
# Define a variable for the local model path for easier management

######## launch vllm server
worker_id=$ARNOLD_ID
instance_id=$((worker_id / WORKERS_PER_INSTANCE))
role_in_instance=$((worker_id % WORKERS_PER_INSTANCE))
head_worker_id=$((instance_id * WORKERS_PER_INSTANCE))

get_worker_host() {
    local wid="$1"
    local var="ARNOLD_WORKER_${wid}_HOST"
    echo "${!var:-}"
}

get_worker_ports() {
    local wid="$1"
    local var="ARNOLD_WORKER_${wid}_PORT"
    echo "${!var:-}"
}

split_ports() {
    local ports_csv="$1"
    local p1
    local p2
    p1=$(echo "$ports_csv" | cut -d',' -f1)
    p2=$(echo "$ports_csv" | cut -d',' -f2)
    if [[ -z "$p1" || -z "$p2" ]]; then
        echo "ERROR: Expected two comma-separated ports, got: '$ports_csv'" >&2
        return 1
    fi
    echo "$p1 $p2"
}

WORKER_HOST=$(get_worker_host "$worker_id")
WORKER_PORTS=$(get_worker_ports "$worker_id")

read -r WORKER_SERVER_PORT WORKER_RPC_PORT <<<"$(split_ports "$WORKER_PORTS")"

HEAD_HOST=$(get_worker_host "$head_worker_id")
HEAD_PORTS=$(get_worker_ports "$head_worker_id")
read -r HEAD_SERVER_PORT HEAD_RPC_PORT <<<"$(split_ports "$HEAD_PORTS")"

DP_ADDRESS=$HEAD_HOST
DP_RPC_PORT=$HEAD_RPC_PORT

# Bind address/port for THIS worker process
HOST_IP=$WORKER_HOST
HOST_PORT=$WORKER_SERVER_PORT

build_base_config_args() {
    # ==============================================================================
    # VLLM Server Configurations
    # You can override these by setting environment variables before running the script.
    # e.g., TP_SIZE=4 ./deploy.sh
    # ==============================================================================

    # Build the arguments string
    local args="--tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --host $HOST_IP \
    --port $HOST_PORT \
    --served-model-name $SERVED_MODEL_NAME \
    --async-scheduling"
    
    if [[ $MODEL_TYPE == qwen35* ]]; then
        args+="   --mm-processor-cache-type "shm""
    fi
    # 模型特定配置
    case "$MODEL_TYPE" in
    # minimax_m2
        "minimax21")
            # DeepSeekV32 特殊配置
            args+="   --trust-remote-code"
            ;;
        "deepseek_v32")
            # DeepSeekV32 特殊配置
            args+="  --dtype bfloat16"
            args+="  --tokenizer-mode deepseek_v32"
            args+="  --default-chat-template-kwargs \"{\"enable_thinking\": true}\""
            ;;
        "qwen3-235b-5m")
            # Qwen3-235B-5M 特殊配置
            if [[ $ENABLE_LONG_CONTEXT == "True" ]]; then
                args+="  --hf-overrides.rope_parameters.rope_theta 100000"
                args+="  --hf-overrides.rope_parameters.factor 4.0"
                args+="  --hf-overrides.rope_parameters.original_max_position_embeddings 32768"
                args+="  --hf-overrides.rope_parameters.rope_type yarn"
            fi
            ;;
        "k25")
            # k25 特殊配置
            if [[ $ENABLE_TP_ENCODER == "True" ]]; then
                args+="  --mm-encoder-tp-mode data"
            fi
            args+="   --trust-remote-code"
            ;;
             
    esac
   
    if [[ $ENFORCE_EAGER == "True" ]]; then
        args+= "  --enforce-eager"
    fi
    if [[ $ENABLE_EP == "True" ]]; then
        args+="  --enable-expert-parallel"
    fi

    if [[ $ENABLE_DP == "True" ]]; then
        args+="  --data-parallel-size $DP_SIZE"
        args+="  --data-parallel-size-local $DP_SIZE_LOCAL"
        args+="  --data-parallel-address $DP_ADDRESS"
        args+="  --data-parallel-rpc-port $DP_RPC_PORT"
        args+="  --data-parallel-external-lb"
        args+="  --data-parallel-rank $role_in_instance"
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

build_role_config_args() {
    local args=""
    if [[ $role_in_instance -ne 0 ]]; then
        args+="  --data-parallel-start-rank $START_RANK"
        args+="  --headless"
    fi
    echo "$args"
}

build_serve_args() {
    local args=""
    args+=$(build_base_config_args)
    # args+=$(build_role_config_args)
    echo "$args"
}

serve_model() {
    
    local python_args=$(build_serve_args)
    echo "python_args: $python_args"

    if [[ $worker_id -eq 0 ]]; then
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve "$LOCAL_MODEL_PATH" $python_args
    else
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve "$LOCAL_MODEL_PATH" $python_args
    fi

}

serve_model
