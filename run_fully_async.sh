# cd /opt/tiger
# git clone https://github.com/verl-project/verl.git

# dependency: vllm==0.18.0, transformers@<cc7ab9be>

# cd /mnt/bn/tiktok-mm-5/aiic/users/wujinming/projects/CUA/code/rl/verl_cua;
# pip instsall tilelang --user
# pip install --upgrade "cupy-cuda12x" --user
export WANDB_API_KEY=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
# cd /opt/tiger/verl
set -x

# Fully async requires vLLM V1 engine
export VLLM_USE_V1=1

RAY_DATA_HOME=/mnt/bn/tiktok-mm-5/aiic/users/wujinming/projects/CUA/code/rl
# MODEL_PATH=/mnt/hdfs/tiktok_aiic/user/zhangbofei.1030/Qwen3.5-9B
MODEL_PATH=/opt/tiger/Qwen3.5-9B
project_name='fully_async_debug'
gen_tp=1
sp_size=1
ENGINE=${1:-vllm}
verl_src_path="/mnt/bn/tiktok-mm-4/aiic/users/zhangbiao.168/verl"
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-27B"}
CKPTS_DIR=${CKPTS_DIR:-"${verl_src_path}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/verl_quickstart/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/verl_quickstart/geo3k/test.parquet"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}


# Fully async: split GPUs between Rollouter and Trainer (1 node x 8 GPUs -> 4 + 4)
n_gpus_rollout=4
n_gpus_training=4

# Fully async specific hyper-parameters (official recommended defaults)
use_trainer_do_validate=False
bypass_mode=True
# Multi-turn tool-calling parameters
max_turns=8
# tool_config_path="${verl_src_path}/examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml"
tool_config_path="${verl_src_path}/geo3k_tool_config.yaml"
max_prompt_length=4096
max_response_length=2048
actor_max_token_len_per_gpu=$((3 * (max_prompt_length + max_response_length) / 2))
require_batches=1

partial_rollout=True
staleness_threshold=0.5
trigger_parameter_sync_step=4

exp_name=GRPO-Qwen3_5-9B-FullyAsync-trigger_parameter_sync_step_${trigger_parameter_sync_step}-staleness_threshold_${staleness_threshold}-partial_rollout_${partial_rollout}


python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.bypass_mode=$bypass_mode \
    data.train_files=$TRAIN_FILE \
    data.train_batch_size=0 \
    data.val_files=$TEST_FILE \
    data.gen_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key='images' \
    data.shuffle=False \
    data.return_raw_chat=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${n_gpus_training} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.ref.use_torch_compile=True \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    rollout.total_rollout_steps=1000000 \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout} \
    async_training.use_trainer_do_validate=${use_trainer_do_validate} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=${n_gpus_training} \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=500 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.resume_from_path=null \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$CKPTS_DIR

    # +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_type=shm \

