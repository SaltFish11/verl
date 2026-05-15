#!/bin/bash
source "$(dirname "$0")/../../setup_env.sh"

# Trainer settings
NNODES=${ARNOLD_WORKER_NUM:-1} # Use cluster variable if available, otherwise default to 1
N_GPUS_PER_NODE=$ARNOLD_WORKER_GPU

# Data settings - use paths from setup_env.sh
# Default data file, can be overridden with --data_path
DATA_PATH="simplelr_math_35/train.parquet"
PROMPT_KEY="prompt"
# rollout_n
ROLLOUT_N=16

# Output paths will be generated dynamically below if not provided by the user.
OUTPUT_PATH=""
RAW_RESPONSE_PATH=""

# Model settings - use paths from setup_env.sh
# Default model, can be overridden with --model_name
MODEL_NAME=Qwen2.5-7B
TEMPERATURE=1.0
TOP_K=-1
TOP_P=1.0
PROMPT_LENGTH=2048
RESPONSE_LENGTH=4096
GPU_MEMORY_UTILIZATION=0.9
TENSOR_MODEL_PARALLEL_SIZE=1
CUSTOM_SUFFIX=""

# setup reward manager
REWARD_MANAGER="math"

# --- Step 3: Helper function to generate a suffix for the run name ---
generate_suffix() {
  local suffix=""
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --rollout_n) suffix+="_n$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --top_k) suffix+="_topk$2"; shift 2 ;;
      --top_p) suffix+="_topp$2"; shift 2 ;;
      --prompt_length) suffix+="_prlen$2"; shift 2 ;;
      --response_length) suffix+="_reslen$2"; shift 2 ;;
      --custom_suffix) suffix+="_$2"; shift 2 ;;
      --reward_manager) suffix+="_rm$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  echo "$suffix"
}

echo "Arguments received by script: $@"

# --- Step 4: Parse command-line arguments to override defaults ---
# Store original arguments to pass to generate_suffix
ORIGINAL_ARGS=("$@")

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --nnodes) NNODES="$2"; shift 2 ;;
    --data_path) DATA_PATH="$2"; shift 2 ;;
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --prompt_length) PROMPT_LENGTH="$2"; shift 2 ;;
    --response_length) RESPONSE_LENGTH="$2"; shift 2 ;;
    --tp_size) TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --custom_suffix) CUSTOM_SUFFIX="$2"; shift 2 ;;
    --reward_manager) REWARD_MANAGER="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --- Step 5: Generate dynamic Output and Raw Response Paths ---
# This step runs *after* all arguments are parsed, so it uses the final values.

# Generate a unique suffix based on the run's hyperparameters
SUFFIX=$(generate_suffix "${ORIGINAL_ARGS[@]}")

# Extract the base directory from the final data path
# e.g., for "simplelr_math_35/train.parquet", this will be "simplelr_math_35"
BASE_DATA_DIR=$(dirname "$DATA_PATH")

# Sanitize model name to create a clean filename component
# Replaces characters like '/' with '_' (e.g., 'org/model' becomes 'org_model')
MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | tr '/' '_')

# Always generate the raw response path dynamically based on parameters.
RAW_RESPONSE_PATH="${BASE_DATA_DIR}/${MODEL_NAME_CLEAN}_raw_response${SUFFIX}.json"

# If an output path was NOT provided via command line, generate a dynamic one.
# Otherwise, we will use the one provided by the user.
if [ -z "$OUTPUT_PATH" ]; then
  OUTPUT_PATH="${BASE_DATA_DIR}/${MODEL_NAME_CLEAN}_train_with_solverate${SUFFIX}.parquet"
  echo "INFO: Output path not specified, dynamically generating: $OUTPUT_PATH"
else
  echo "INFO: Using user-specified output path: $OUTPUT_PATH"
fi

echo "==================================="
echo "FINAL RUN CONFIGURATION"
echo "==================================="
echo "--- Paths (from setup_env.sh) ---"
echo "HDFS Data Path: $HDFS_DATA_PATH"
echo "HDFS Model Path: $HDFS_MODEL_PATH"
echo
echo "--- Inference Parameters ---"
echo "Nodes: $NNODES"
echo "GPUs per Node: $N_GPUS_PER_NODE"
echo "Model Name (Final): $MODEL_NAME"
echo "Data Path (Final): $DATA_PATH"
echo "Raw Response Path: $RAW_RESPONSE_PATH"
echo "Output Path (Final): $OUTPUT_PATH"
echo "Number of Samples: $N_SAMPLES"
echo "Reward Manager: $REWARD_MANAGER"
echo "Temperature: $TEMPERATURE"
echo "Top K: $TOP_K"
echo "Top P: $TOP_P"
echo "Tensor Parallel Size: $TENSOR_MODEL_PARALLEL_SIZE"
echo "==================================="

# Add a small delay to allow user to review parameters
sleep 3

max_num_batched_tokens=$(expr $PROMPT_LENGTH + $RESPONSE_LENGTH + 1000)

# --- Step 6: Construct and run the final command ---
# NOTE: The python script entry point `verl.inference.main` is an assumption.
# Please change it to your actual script's entry point if it's different.
python -m aiic_verl.trainer.code.main_grading \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    data.path=$HDFS_DATA_PATH/$DATA_PATH \
    data.prompt_key=$PROMPT_KEY \
    data.n_samples=$ROLLOUT_N \
    data.max_prompt_length=$PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.raw_response_path=$HDFS_DATA_PATH/$RAW_RESPONSE_PATH \
    data.output_path=$HDFS_DATA_PATH/$OUTPUT_PATH \
    model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    rollout.temperature=$TEMPERATURE \
    rollout.reward_manager=$REWARD_MANAGER \
    rollout.top_k=$TOP_K \
    rollout.top_p=$TOP_P \
    rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    rollout.max_num_batched_tokens=$max_num_batched_tokens

echo "Inference finished. Final graded data saved to $HDFS_DATA_PATH/$OUTPUT_PATH"
echo "Raw model responses (if generated by the script) would be at $HDFS_DATA_PATH/$RAW_RESPONSE_PATH"
