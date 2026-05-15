set -x

# #curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
bash $UTILS_DIR/bash.sh
source $HOME/.cargo/env
sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a --force-confdef --force-confold
sudo apt-get install -y protobuf-compiler libprotobuf-dev libssl-dev
# #Build Rust components: if precompiled, not run
# # option 
##solve openssl install error
sudo apt install pkg-config

## option 2
# pip install setuptools-rust wheel build --user
# python -m build
VLLM_ENV_PATH="${VLLM_ENV_PATH:-/tmp/VLLM_V13_env}"
source "${VLLM_ENV_PATH}/bin/activate"

uv pip install $UTILS_DIR/*.whl

# # Rebuild & reinstall in one step during development
# python -m build && pip install --force-reinstall dist/*.whl --user