set -x
VLLM_ENV_PATH="${VLLM_ENV_PATH:-/tmp/VLLM_V14_env}"
pip install uv --user
# uv venv "${VLLM_V13}" --python 3.12 --seed -i http://bytedpypi.byted.org/simple --trusted-host bytedpypi.byted.org 
uv venv "${VLLM_ENV_PATH}" --python 3.12 --seed -i http://bytedpypi.byted.org/simple --trusted-host bytedpypi.byted.org --clear
source "${VLLM_ENV_PATH}/bin/activate"
if [[ $MODEL_TYPE == "k25" ]]; then
    # uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
    uv pip install vllm==v0.18.0
elif  [[ $MODEL_TYPE == "deepseek_v32" ]]; then
    uv pip install vllm==v0.13.0
    uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
elif [[ $MODEL_TYPE == "small" ]]; then
    uv pip install vllm==v0.14.0
elif [[ $MODEL_TYPE == qwen35* ]]; then
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/0.20.2/cu129 --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
else 
    # uv pip install vllm==v0.14.0
    uv pip install git+https://github.com/SaltFish11/vllm.git@v0.14.0_add_log
fi
# cd /mnt/bn/tiktok-mm-4/aiic/users/zhangbiao.168/vllm
# VLLM_USE_PRECOMPILED=1 uv pip install -e .