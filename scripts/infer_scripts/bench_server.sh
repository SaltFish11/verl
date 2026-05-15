VLLM_ENV_PATH="${VLLM_ENV_PATH:-/tmp/VLLM_V13_env}"
source "${VLLM_ENV_PATH}/bin/activate"

MODEL_PATH=${MODEL_PATH:-"/tmp/baseline/235b-5m/Qwen3-235B-A22B/"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"gpt-oss-qwencoder_model_9"} 

HOSTS=${ARNOLD_WORKER_0_HOST}
PORTS=$ARNOLD_WORKER_0_PORT
PORT=$(echo "$PORTS" | cut -d',' -f1)

bench_model() {
    vllm bench serve \
        --model $MODEL_PATH \
        --served-model-name $SERVED_MODEL_NAME \
        --base-url "http://[$HOSTS]:$PORT" \
        --endpoint /v1/chat/completions \
        --backend openai-chat \
        --random-input-len 8096 \
        --random-output-len 1024 \
        --request-rate 8 \
        --num-prompts 400 \
        --max-concurrency 8
}

bench_model