
# 海外办公网  
# export LITELLM_BASE_URL="http://litellm.tiktok-row.net"
# 国内生产网/办公网通用
# export LITELLM_BASE_URL="litellm-gateway.bytedance.net"
# 海外生产网
export LITELLM_BASE_URL="http://maas.byteintl.net/gateway"
export LITELLM_API_KEY="sk-1234"
export PSM=tiktok.aiic.${SERVED_MODEL_NAME}
#make model_list.yaml
#!/bin/bash

# 确保环境变量 MODEL_NAME 已设置
if [ -z "$SERVED_MODEL_NAME" ]; then
    echo "错误：请先设置环境变量 SERVED_MODEL_NAME"
    exit 1
fi

#detect psm is used
# python $UTILS_DIR/detect_psm.py -l ${PSM}
# check_ip_alive_status=$?
# if [[ $check_ip_alive_status -ne 0 ]]; then
#     echo "the psm ${PSM} has been used"
#     exit 1
# fi

# 写入 YAML 文件
cat > /tmp/model_list.yaml << EOF
model_list:
  - model_name: $SERVED_MODEL_NAME
    litellm_params:
      model: openai/$SERVED_MODEL_NAME
      api_base: psm://tiktok.aiic.$SERVED_MODEL_NAME/v1
      api_key: "sk-1234"
      extra_headers:
        extra: '{ "Destination-Service": "tiktok.aiic.$SERVED_MODEL_NAME" }'
EOF

echo "✅ 已生成 model_list.yaml，使用模型名：$SERVED_MODEL_NAME"

#register psm to litellm
python $UTILS_DIR/upsert_model.py -c /tmp/model_list.yaml



#bind vllm server to the registed psm on litellm
python $UTILS_DIR/register_psm.py register ${PSM} ${SERVER_PORT}

