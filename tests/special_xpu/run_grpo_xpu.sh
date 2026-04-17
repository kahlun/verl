#!/usr/bin/env bash
# E2E GRPO test on Intel XPU — mirrors tests/special_npu/run_qwen2_5_05b_grpo.sh
#
# Validates the full RL training loop on XPU:
#   FSDP training (actor + ref) → vLLM rollout → reward → train
#
# Prerequisites:
#   - PyTorch with XPU support (torch.xpu.is_available() == True)
#   - vLLM >= 0.17 with XPU platform support
#   - oneCCL for xccl distributed backend
#
# Known workarounds (pre-DLE 2026.0):
#   CCL_ATL_SHM=1 CCL_BUFFER_CACHE=0  (Level Zero IPC bug on PCIe cards)
#
# Usage:
#   NUM_GPUS=2 bash tests/special_xpu/run_grpo_xpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-2}
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

# XPU fix: Ray pre-starts ~100 idle workers, each opens an L0 context on the GPU.
# torch.xpu.mem_get_info() counts their combined context overhead (~20 GB) as "used"
# even though actual allocations succeed fine. This causes vLLM's pre-flight memory
# check to abort with a false OOM. Two fixes:
#   1. Reduce Ray prestarted workers to avoid L0 context memory pressure
#   2. Patch vLLM's request_memory to skip the check on XPU
export RAY_NUM_PRESTART_PYTHON_WORKERS=0

# XPU: vLLM's _run_in_subprocess model inspection spawns a fresh Python process.
# If HuggingFace Hub is unreachable through the proxy, the subprocess hangs until
# timeout. Setting HF_HUB_OFFLINE=1 forces it to use cached model config instantly.
export HF_HUB_OFFLINE=1

# Redirect Ray temp dir if /tmp is nearly full (avoids "over 95% full" errors)
if [ -d /workspace ]; then
    export RAY_TMPDIR=${RAY_TMPDIR:-/workspace/.ray_tmp}
    mkdir -p "$RAY_TMPDIR"
fi

# XPU vLLM patches are applied automatically inside the training process
# (vllm_async_server.py calls xpu_patches.apply() before vLLM initialisation).
# No pre-step needed here.

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
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_xpu_grpo_e2e' \
    trainer.experiment_name='qwen2_5_05b_xpu_grpo' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    ray_kwargs.ray_init.num_gpus=${NUM_GPUS} $@
