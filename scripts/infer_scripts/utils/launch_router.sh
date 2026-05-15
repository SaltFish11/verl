set -x
VLLM_ENV_PATH="${VLLM_ENV_PATH:-/tmp/VLLM_V14_env}"
source "${VLLM_ENV_PATH}/bin/deactivate"

PSM=tiktok.aiic.$SERVED_MODEL_NAME
# python $UTILS_DIR/register_psm.py register ${PSM} ${SERVER_PORT}

worker_urls=()
for worker_id in $(seq 0 $((ARNOLD_WORKER_NUM - 1))); do
    # 1. 获取当前 worker 的端口环境变量（先拼接变量名，再间接取值）
    port_var="ARNOLD_WORKER_${worker_id}_PORT"
    worker_port=${!port_var}  # 间接取值，替代 echo + cut 的冗余操作
    # 2. 截取端口的第一个值（处理端口可能是逗号分隔的情况）
    worker_port=$(echo "$worker_port" | cut -d',' -f1)
    
    # 3. 获取当前 worker 的主机环境变量
    host_var="ARNOLD_WORKER_${worker_id}_HOST"
    worker_host=${!host_var}
    
    # 4. 拼接 URL 并加入数组
    worker_urls+=("http://[${worker_host}]:${worker_port}")
done

source "${VLLM_ENV_PATH}/bin/activate"
# Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value
vllm-router \
    --worker-urls ${worker_urls[@]} \
    --policy cache_aware \
    --intra-node-data-parallel-size 1 \
    --host $ARNOLD_WORKER_0_HOST \
    --port $SERVER_PORT \
    --worker-startup-timeout-secs 3600
