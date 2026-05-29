#!/usr/bin/env bash
# E2E GRPO test on Intel XPU with sglang rollout engine
#
# Validates the full RL training loop on XPU using sglang instead of vLLM:
#   FSDP training (actor + ref) → sglang rollout → reward → train
#
# Prerequisites:
#   - PyTorch with XPU support (torch.xpu.is_available() == True)
#   - sglang 0.5.12 installed
#   - sgl-kernel-xpu built from source (replaces sgl-kernel CUDA package)
#   - oneCCL for xccl distributed backend
#
# Usage:
#   NUM_GPUS=2 bash tests/special_xpu/run_grpo_sglang_xpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}

# oneCCL workarounds for multi-GPU
export CCL_ATL_SHM=${CCL_ATL_SHM:-1}
export CCL_BUFFER_CACHE=${CCL_BUFFER_CACHE:-0}
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_TOPO_ALGO=0

# XPU device selection via ZE_AFFINITY_MASK
_DEVICES=$(seq 0 $((NUM_GPUS-1)) | paste -sd',')
export ZE_AFFINITY_MASK="${_DEVICES}"
unset ONEAPI_DEVICE_SELECTOR

export RAY_NUM_PRESTART_PYTHON_WORKERS=0
# Raise OOM kill threshold to avoid false kills during weight sync CPU buffers
# Disable Ray OOM monitor: CPU weight buffers briefly spike to ~22 GB during
# FSDP→sglang weight sync. Safe to disable for single-step e2e smoke tests.
export RAY_memory_monitor_refresh_ms=0

# Load XPU platform plugin
export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-verl_xpu}

# Enable sglang Intel XPU kernel backend (sgl-kernel-xpu)
export SGLANG_USE_SGL_XPU=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    ++actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.custom_engine_path=verl_xpu.fsdp_engine_xpu \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.custom_engine_path=verl_xpu.fsdp_engine_xpu \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_xpu_grpo_sglang_e2e' \
    trainer.experiment_name='qwen2_5_05b_xpu_grpo_sglang' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} $@
