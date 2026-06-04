# dependency: vllm==0.18.0, transformers@<cc7ab9be>

# cd /mnt/bn/tiktok-mm-5/aiic/users/wujinming/projects/CUA/code/rl/verl_cua;
# pip instsall tilelang --user
export WANDB_API_KEY=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
# CAUSAL_CONV1D_FORCE_BUILD=TRUE \
# pip install causal-conv1d --no-build-isolation -v
# peft-0.19.1
set -x
RAY_DATA_HOME=/mnt/bn/tiktok-mm-5/aiic/users/wujinming/projects/CUA/code/rl
MODEL_PATH=/mnt/hdfs/tiktok_aiic/user/zhangbofei.1030/Qwen3.5-9B
project_name='tree_attentin_debug_baseline_again'
date=$(date +%Y%m%d)
exp_name="tree_attention_$date"
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


    # trainer.use_legacy_worker_impl=disable \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    algorithm.rollout_correction.bypass_mode=true \
    +actor_rollout_ref.actor.fsdp_config.enable_tree_training=true \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.actor.fsdp_config.tree_max_tokens_per_tree=8192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key="images" \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.ref.use_torch_compile=True \
    actor_rollout_ref.ref.fsdp_config.offload_policy=True \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=500 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.resume_from_path=null \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$CKPTS_DIR


