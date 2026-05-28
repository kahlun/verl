#!/usr/bin/env bash
# XPU PPO E2E test — FSDP + XCCL backend
# Mirror of tests/special_npu/run_qwen3_06b_ppo.sh for Intel XPU (Arc / Data Center GPU Max).
#
# Env vars:
#   MODEL_ID        HuggingFace model ID (default: Qwen/Qwen3-0.6B)
#   MODEL_PATH      Path to local model weights (default: HF cache)
#   TRAIN_FILES     GSM8K train parquet (default: ~/data/gsm8k/train.parquet)
#   VAL_FILES       GSM8K test parquet  (default: ~/data/gsm8k/test.parquet)
#   NUM_XPUS        Number of XPU devices to use (default: 2)

set -xeuo pipefail

# ---- CCL stability flags for PCIe-connected XPU (platform_bugs.md Bug #5) ----
# CCL_ATL_SHM=1     : use shared-memory ATL transport (avoids ZE_RESULT_ERROR_UNKNOWN
#                     that torchrun/mp.spawn exhibit with the default TCP transport)
# CCL_BUFFER_CACHE=0: disable CCL buffer reuse cache that triggers ZE_ERROR on repeated
#                     all-gather/reduce-scatter calls in FSDP sharded training
export CCL_ATL_SHM=1
export CCL_BUFFER_CACHE=0

# Per-rank ONEAPI_DEVICE_SELECTOR and ZE_AFFINITY_MASK are set automatically by Ray's
# IntelGPUAcceleratorManager (Ray 2.10+); do not override them here.

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

NUM_XPUS=${NUM_XPUS:-2}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    critic.use_dynamic_bsz=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_ppo_example_gsm8k_qwen3_xpu' \
    trainer.experiment_name='qwen3_06b_fsdp_xpu' \
    trainer.n_gpus_per_node="${NUM_XPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    "$@"
