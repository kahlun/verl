#!/usr/bin/env bash
# E2E PPO (GAE) test on Intel XPU — mirrors tests/special_npu/run_qwen3_06b_ppo.sh
#
# Validates PPO training with critic model on XPU:
#   FSDP training (actor + critic + ref) → vLLM rollout → GAE reward → train
#
# Usage:
#   NUM_GPUS=4 bash tests/special_xpu/run_ppo_xpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-4}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}

# oneCCL workarounds for multi-GPU (pre-DLE 2026.0 driver)
export CCL_ATL_SHM=${CCL_ATL_SHM:-1}
export CCL_BUFFER_CACHE=${CCL_BUFFER_CACHE:-0}
# Disable topology recognition — assume XeLink across devices
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

# XPU device selection: use ZE_AFFINITY_MASK (Level Zero) for device restriction.
# vLLM 0.17+ XPU platform uses ZE_AFFINITY_MASK as device_control_env_var; setting
# ONEAPI_DEVICE_SELECTOR=level_zero:N,M breaks the FLA/triton SYCL JIT init path.
_DEVICES=$(seq 0 $((NUM_GPUS-1)) | paste -sd',')
export ZE_AFFINITY_MASK="${_DEVICES}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_xpu_ppo_e2e' \
    trainer.experiment_name='qwen2_5_05b_xpu_ppo' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 $@
