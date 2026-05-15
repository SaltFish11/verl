# Infer Scripts 目录说明

本目录包含用于模型推理、部署和基准测试的脚本集合，主要基于 VLLM 框架实现高效的模型推理服务。

## 目录结构

```
examples/infer_scripts/
├── README.md                          # 项目说明文档
├── bench_server.sh                    # VLLM 服务器基准测试脚本
├── infer_grading.sh                   # 模型推理和评分脚本
├── launch_server_multi_instance.sh # 多实例部署脚本
├── launch_server_one_instance.sh      # 单实例部署脚本
└── utils/                             # 工具脚本目录
    ├── bash.sh                        # vllm-router 安装的依赖脚本
    ├── get_hdfs_model.sh              # 从 HDFS 获取模型的脚本
    ├── install_vllm_router.sh         # 安装 VLLM 路由器的脚本
    ├── install_vllm_uvenv.sh          # 安装 VLLM 虚拟环境的脚本
    ├── launch_router_and_registerpsm.sh  # 启动路由器并注册 PSM 的脚本
    └── launch_server.sh               # 通用服务器启动脚本
```

## 脚本功能说明

### 1. 服务器部署脚本
#### `launch_server_multi_instance.sh`
- **功能**：在多实例环境中启动 VLLM 服务器
- **支持特性**：
  - 多实例部署（每个实例使用多个 worker）
  - 自动安装 VLLM 虚拟环境
  - 自动安装 VLLM 路由器（仅在主实例）
  - 从 HDFS 自动下载模型（可选）
  - 长上下文支持（默认禁用）
  - 工具调用支持
  - 专家并行支持
  - 数据并行支持（默认启用）
- **环境变量**：
  - `MODEL_TYPE`：模型类型（必填；可选项：minimax21|deepseek_v32|glm47|k2-thinging|k25|qwen3coder|qwen3-235b-5m|qwen3-235b-7m）
  - `VLLM_ENV_PATH`：VLLM 环境路径（默认：/tmp/VLLM_V14_env）
  - `SERVED_MODEL_NAME`：服务模型名称（默认：gpt-5-codex-qwencoder_model_0）
  - `LOCAL_MODEL_PATH`：本地模型路径（默认：/tmp/baseline/Qwen3-235B-A22B）
  - `HDFS_MODEL_PATH`：HDFS 模型路径（默认：/mnt/bn/tiktok-mm-5/aiic/users/guqingshui/serving_on_merlin/models/Qwen3-235B-A22B）
  - `COPY_TO_LOCAL`：是否从 HDFS 复制模型（默认：True）
  - `TP_SIZE`：张量并行度（默认：8）
  - `DP_SIZE`：数据并行度（默认：2）
  - `DP_SIZE_LOCAL`：本地数据并行度（默认：1）
  - `WORKERS_PER_INSTANCE`：每个实例的 worker 数量（默认：DP_SIZE）
  - `MAX_MODEL_LEN`：最大模型长度（默认：120000）
  - `GPU_MEM_UTIL`：GPU 内存利用率（默认：0.75）
  - `ENABLE_EP`：是否启用专家并行（默认：True）
  - `ENABLE_DP`：是否启用数据并行（默认：True）
  - `ENABLE_TOOL`：是否启用工具调用（默认：True）
  - `ENABLE_LONG_CONTEXT`：是否启用长上下文（默认：False, qwen3-235b-5m 需要将其打开）
  - `ENFORCE_EAGER`：是否强制 eager 模式（默认：False）
  - `TOOL_PARSER`：工具解析器（默认跟随 `MODEL_TYPE`）
  - `REASON_PARSER`：reasoning 解析器
  - `ARNOLD_ID`：实例 ID（用于确定主实例）

### 2. 测试和评估脚本

#### `bench_server.sh`
- **功能**：对 VLLM 服务器进行基准测试
- **测试参数**：
  - 随机输入长度：8096
  - 随机输出长度：1024
  - 请求速率：8
  - 提示数量：400
  - 最大并发数：8
- **输出**：测试报告，包含延迟、吞吐量等指标

#### `infer_grading.sh`
- **功能**：模型推理和评分
- **支持特性**：
  - 多节点推理
  - 动态输出路径生成
  - 可配置的推理参数
- **推理参数**：
  - `ROLLOUT_N`：rollout 数量
  - `TEMPERATURE`：温度参数
  - `TOP_K`：top-k 采样
  - `TOP_P`：top-p 采样
  - `PROMPT_LENGTH`：提示长度
  - `RESPONSE_LENGTH`：响应长度

### 3. 工具脚本

#### `utils/get_hdfs_model.sh`
- **功能**：从 HDFS 下载模型到本地
- **使用场景**：当 `COPY_TO_LOCAL` 为 `True` 时自动调用

#### `utils/install_vllm_uvenv.sh`
- **功能**：安装 VLLM 虚拟环境
- **步骤**：
  1. 安装 uv 工具
  2. 创建 Python 3.12 虚拟环境
  3. 安装 vllm==v0.14.0

#### `utils/install_vllm_router.sh`
- **功能**：安装 VLLM 路由器
- **依赖**：
  - Rust 环境
  - protobuf-compiler
  - libprotobuf-dev
  - libssl-dev
  - pkg-config

#### `utils/launch_router_and_registerpsm.sh`
- **功能**：启动 VLLM 路由器并注册 PSM 服务
- **步骤**：
  1. 注册 PSM 服务
  2. 收集所有 worker 的 URL
  3. 启动 VLLM 路由器
- **使用场景**：仅在主实例（ARNOLD_ID=0）上调用

#### `utils/launch_server.sh`
- **功能**：通用服务器启动脚本
- **支持特性**：
  - 动态配置生成
  - 支持多种并行策略
  - 长上下文支持
  - 工具调用支持

## 使用方法

### 1. 单实例部署

```bash
# 设置环境变量（可选）
export VLLM_ENV_PATH=/path/to/vllm/env
export LOCAL_MODEL_PATH=/path/to/model
export HDFS_MODEL_PATH=/path/to/hdfs/model
export COPY_TO_LOCAL=True
export TP_SIZE=8
export DP_SIZE=2
export ENABLE_DP=True
export ENABLE_LONG_CONTEXT=False

# 启动服务器
./launch_server_one_instance.sh
```

### 2. 多实例部署

```bash
# 设置环境变量（可选）
export VLLM_ENV_PATH=/path/to/vllm/env
export LOCAL_MODEL_PATH=/path/to/model
export HDFS_MODEL_PATH=/path/to/hdfs/model
export COPY_TO_LOCAL=True
export TP_SIZE=8
export DP_SIZE=2
export WORKERS_PER_INSTANCE=2
export ENABLE_DP=True
export ENABLE_LONG_CONTEXT=False

# 启动服务器
./launch_server_multi_instance.sh
```

### 3. 基准测试

```bash
# 设置环境变量
export MODEL_PATH=/path/to/model
export SERVED_MODEL_NAME=gpt-oss-qwencoder_model_9

# 运行基准测试
./bench_server.sh
```

### 4. 模型推理和评分

```bash
# 设置环境变量
export MODEL_NAME=Qwen2.5-7B
export DATA_PATH=simplelr_math_35/train.parquet

# 运行推理和评分
./infer_grading.sh --rollout_n 16 --temperature 1.0
```

## 技术栈

- **主要框架**：VLLM
- **脚本语言**：Bash
- **依赖**：
  - Python 3.8+
  - PyTorch
  - VLLM
  - Rust (用于 VLLM 路由器)
  - protobuf

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查 HDFS 路径是否正确
   - 确保网络连接正常
   - 检查本地磁盘空间是否充足

2. **端口冲突**
   - 确保 `ARNOLD_WORKER_${worker_id}_PORT` 环境变量正确设置
   - 检查端口是否被其他进程占用

3. **VLLM 路由器安装失败**
   - 确保 Rust 环境正确安装
   - 检查依赖包是否安装完整
   - 注意：路由器仅在主实例（ARNOLD_ID=0）上安装

4. **长上下文支持问题**
   - 确保 `ENABLE_LONG_CONTEXT` 环境变量设置为 `True`
   - 检查 `MAX_MODEL_LEN` 是否设置合理
   - 默认情况下，长上下文支持是禁用的（qwen3-235b-5m 需要将其打开）

### 调试技巧

- 脚本默认已开启调试模式（set -x）
- 检查环境变量是否正确设置
- 查看 VLLM 服务器日志获取详细错误信息
- 使用 `bench_server.sh` 验证服务器性能和可用性
- 注意：路由器仅在主实例（ARNOLD_ID=0）上启动

## 注意事项

- 本脚本集合主要针对分布式环境设计，在单机环境下可能需要调整配置
- 对于大模型（如 Qwen3-235B），确保有足够的 GPU 内存和网络带宽
- 首次运行时，从 HDFS 下载模型可能需要较长时间，请耐心等待
- 请根据实际硬件配置调整 `GPU_MEM_UTIL` 和 `TP_SIZE` 等参数

## 联系信息

如有问题或建议，请联系相关维护人员。

---

*本文档仅供参考，具体使用请根据实际环境和需求进行调整。*
