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

# SYCL device selector: must be explicit indices (Ray rejects '*').
# Always override — the Dockerfile ENV may contain 'level_zero:*' which Ray rejects.
_DEVICES=$(seq 0 $((NUM_GPUS-1)) | paste -sd',')
export ONEAPI_DEVICE_SELECTOR="level_zero:${_DEVICES}"

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

python3 << 'XPUPATCH'
# XPU WORKAROUND — two runtime patches for colocated FSDP+vLLM on Intel XPU.
#
# Root cause: XPU has no CuMemAllocator equivalent (CUDA uses a shared memory pool
# so FSDP and vLLM coordinate ownership; on XPU they see raw driver free memory).
# Upstream fix in progress: https://github.com/vllm-project/vllm/pull/37149
# (XpuMemAllocator / transparent sleep mode — requires torch-xpu 2.11, still draft)
#
# Patch 1 (vllm/v1/worker/utils.py — request_memory):
#   vLLM checks free_memory >= requested on startup. When FSDP already occupies the
#   GPU this fails with ValueError even though FSDP will CPU-offload before inference.
#   Workaround: skip the ValueError on XPU only.
#
# Patch 2 (vllm/v1/worker/gpu_worker.py — profiling assert):
#   vLLM asserts free memory did not grow during profiling. When FSDP offloads to CPU
#   during vLLM init, free memory increases, tripping the assert.
#   Workaround: convert the assert to a no-op on XPU only.
#   (Same assert hit ROCm: https://github.com/vllm-project/vllm/pull/36720;
#    xinyu-intel noted "Similar on XPU" but no XPU fix was submitted.)
#
# Remove both patches once PR #37149 merges and torch-xpu 2.11 is available.

import pathlib, sys
vllm_pkg = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
if not vllm_pkg.exists():
    sys.exit(0)

# Patch 1: Skip false OOM in request_memory (L0 context overhead inflates "used" memory)
f = vllm_pkg / "v1/worker/utils.py"
src = f.read_text()
if "xpu_skip" not in src:
    src = src.replace(
        "    if init_snapshot.free_memory < requested_memory:\n        raise ValueError(",
        "    if init_snapshot.free_memory < requested_memory:  # xpu_skip\n"
        "        import torch as _t\n"
        "        if not (hasattr(_t, 'xpu') and _t.xpu.is_available()):\n"
        "            raise ValueError(",
    )
    f.write_text(src)

# Patch 2: Skip profiling assert (FSDP offloads to CPU during vLLM init, freeing GPU)
f2 = vllm_pkg / "v1/worker/gpu_worker.py"
src2 = f2.read_text()
if "xpu_skip" not in src2:
    src2 = src2.replace(
        "        assert self.init_snapshot.free_memory >= free_gpu_memory, (",
        "        # xpu_skip: FSDP offloads params during vLLM startup, free memory grows\n"
        "        if self.init_snapshot.free_memory < free_gpu_memory:\n"
        "            import torch as _t2\n"
        "            if not (hasattr(_t2, 'xpu') and _t2.xpu.is_available()):\n"
        "                assert False, (",
    )
    f2.write_text(src2)

print("[XPU] Patched vLLM memory checks for XPU")


XPUPATCH

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
