set -x
echo "INFO: Copying model from HDFS to local path: $LOCAL_MODEL_PATH"

# Use "mkdir -p" to create parent dirs and avoid errors if the path already exists.
mkdir -p "$LOCAL_MODEL_PATH" || { echo "ERROR: Failed to create directory '$LOCAL_MODEL_PATH'."; exit 1; }

# Change to the directory. Exit if it fails to prevent downloading to the wrong place.
cd "$LOCAL_MODEL_PATH" || { echo "ERROR: Could not change to directory '$LOCAL_MODEL_PATH'."; exit 1; }

echo "INFO: Starting download from ${HDFS_MODEL_PATH}..."

if [[ $HDFS_MODEL_PATH == "hdfs://"* ]]; then
    MODEL_COPY_WORKERS=${MODEL_COPY_WORKERS:-16}
    MODEL_COPY_MAX_RETRIES=${MODEL_COPY_MAX_RETRIES:-3}
    MODEL_COPY_RETRY_DELAY_SEC=${MODEL_COPY_RETRY_DELAY_SEC:-1}
    PYTHON_BIN=${PYTHON_BIN:-python3}
    "$PYTHON_BIN" "${UTILS_DIR}/parallel_model_copy.py" \
        --src "${HDFS_MODEL_PATH}" \
        --dst "$LOCAL_MODEL_PATH" \
        --workers "${MODEL_COPY_WORKERS}" \
        --max-retries "${MODEL_COPY_MAX_RETRIES}" \
        --retry-delay-sec "${MODEL_COPY_RETRY_DELAY_SEC}" \
        --source-type hdfs
elif [[ -d "${HDFS_MODEL_PATH}" ]]; then
    MODEL_COPY_WORKERS=${MODEL_COPY_WORKERS:-16}
    MODEL_COPY_MAX_RETRIES=${MODEL_COPY_MAX_RETRIES:-3}
    MODEL_COPY_RETRY_DELAY_SEC=${MODEL_COPY_RETRY_DELAY_SEC:-1}
    PYTHON_BIN=${PYTHON_BIN:-python3}
    "$PYTHON_BIN" "${UTILS_DIR}/parallel_model_copy.py" \
        --src "${HDFS_MODEL_PATH}" \
        --dst "$LOCAL_MODEL_PATH" \
        --workers "${MODEL_COPY_WORKERS}" \
        --max-retries "${MODEL_COPY_MAX_RETRIES}" \
        --retry-delay-sec "${MODEL_COPY_RETRY_DELAY_SEC}" \
        --source-type local
else
    echo "ERROR: Unsupported HDFS_MODEL_PATH: ${HDFS_MODEL_PATH}"
    exit 1
fi

echo "INFO: Download complete."
