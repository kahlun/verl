#!/usr/bin/env bash
# XPU E2E validation contract (README-derived)
#
# Covers:
#   1. Import smoke test (verl + is_xpu_available)
#   2. Single-GPU SFT (verl.trainer.sft_trainer via torchrun, 2 steps)
#   3. Single-GPU PPO (verl.trainer.main_ppo via Ray, 2 steps)
#   4. Multi-GPU PPO  (verl.trainer.main_ppo via Ray, 2 GPUs, FSDP+XCCL, 2 steps)
#
# Required environment:
#   MODEL_PATH         Path to a local HF model checkpoint (e.g. Qwen/Qwen3-0.6B cache)
#   TRAIN_FILES        GSM8K train parquet
#   VAL_FILES          GSM8K test parquet
#   SFT_TRAIN_FILES    gsm8k_sft train parquet (for SFT section)
#   SFT_VAL_FILES      gsm8k_sft test parquet  (for SFT section)
#
# All paths default to standard verl quickstart locations under $HOME.
# The script exits 0 (skip) when required data/model files are absent so CI
# jobs without data mounted are skipped cleanly rather than failing the CI gate.
# This matches the pattern used by tests/special_e2e/ scripts.

set -xeuo pipefail

# ---- Global XPU environment ----
# CCL stability flags for PCIe-connected XPU (platform_bugs.md Bug #5)
export CCL_ATL_SHM=1
export CCL_BUFFER_CACHE=0
# ZE_SERIALIZE_MODE avoids Level Zero command-list ordering races under heavy load
export ZE_SERIALIZE_MODE=4

# Per-rank ONEAPI_DEVICE_SELECTOR and ZE_AFFINITY_MASK are set by Ray; do not
# override here so Ray's IntelGPUAcceleratorManager can manage device assignment.

# ---- Paths (override via env var before calling this script) ----
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

SFT_TRAIN_FILES=${SFT_TRAIN_FILES:-${HOME}/data/gsm8k_sft/train.parquet}
SFT_VAL_FILES=${SFT_VAL_FILES:-${HOME}/data/gsm8k_sft/test.parquet}

# ---- Pre-flight: check required paths exist ----
for f in "${MODEL_PATH}" "${TRAIN_FILES}" "${VAL_FILES}" "${SFT_TRAIN_FILES}" "${SFT_VAL_FILES}"; do
    if [ ! -e "${f}" ]; then
        echo "[test_xpu_e2e] SKIP: required path not found: ${f}"
        echo "[test_xpu_e2e] Set MODEL_PATH / TRAIN_FILES / VAL_FILES / SFT_TRAIN_FILES / SFT_VAL_FILES to existing paths."
        exit 0
    fi
done

# ========================================================================
# Section 1: Import smoke test
# Validates: is_xpu_available, basic verl import, get_nccl_backend returns xccl
# ========================================================================
echo "=== [1/4] Import smoke test ==="
python3 - <<'PYEOF'
import verl
from verl.utils.device import is_xpu_available, get_nccl_backend, get_device_name
assert is_xpu_available, "is_xpu_available is False — XPU not detected or device.py not patched"
assert get_device_name() == "xpu", f"get_device_name() returned {get_device_name()!r}, expected 'xpu'"
backend = get_nccl_backend()
assert "xccl" in backend, f"get_nccl_backend() returned {backend!r}, expected composite backend containing 'xccl'"
print(f"[OK] is_xpu_available=True  device={get_device_name()}  backend={backend}")
PYEOF

# ========================================================================
# Section 2: Single-GPU SFT  (torchrun, 1 XPU, 2 steps)
# Entry point: verl.trainer.sft_trainer  (matches README quickstart SFT path)
# torchrun is safe for single-process (no inter-process XPU collectives).
# engine=fsdp engine.strategy=fsdp2 exercises the FSDP2 path enabled by PR-004.
# ========================================================================
echo "=== [2/4] Single-GPU SFT (2 steps) ==="
SFT_SAVE_DIR=$(mktemp -d /tmp/verl_xpu_sft_XXXXXX)
trap 'rm -rf "${SFT_SAVE_DIR}"' EXIT

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files="${SFT_TRAIN_FILES}" \
    data.val_files="${SFT_VAL_FILES}" \
    data.train_batch_size=4 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=1024 \
    data.messages_key=messages \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    engine=fsdp \
    engine.strategy=fsdp2 \
    optim=fsdp \
    optim.lr=1e-5 \
    trainer.default_local_dir="${SFT_SAVE_DIR}" \
    trainer.project_name=xpu-e2e-sft \
    trainer.experiment_name=xpu-sft-smoke \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.logger=console \
    trainer.resume_mode=disable
echo "[OK] Single-GPU SFT passed"

# ========================================================================
# Section 3: Single-GPU PPO  (Ray, 1 XPU, 2 steps)
# Entry point: verl.trainer.main_ppo  (matches README quickstart PPO path)
# Ray avoids the torchrun/mp.spawn ZE_RESULT_ERROR_UNKNOWN (Bug #5).
# enforce_eager=True disables vLLM CUDA graph capture (NVIDIA-only).
# ========================================================================
echo "=== [3/4] Single-GPU PPO (2 steps) ==="
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=64 \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    critic.use_dynamic_bsz=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=xpu-e2e-ppo \
    trainer.experiment_name=xpu-ppo-1gpu-smoke \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2
echo "[OK] Single-GPU PPO passed"

# ========================================================================
# Section 4: Multi-GPU PPO  (Ray, 2 XPUs, FSDP+XCCL, 2 steps)
# Entry point: verl.trainer.main_ppo
# Critical distributed path: FSDP sharding uses XCCL collectives via the
# composite backend cpu:gloo,xpu:xccl from PR-001 get_nccl_backend().
# CCL_ATL_SHM + CCL_BUFFER_CACHE already exported above (Bug #5 mitigation).
# ReduceOp.AVG replaced with SUM+divide by PR-005/007 (Bug #6 mitigation).
# ========================================================================
echo "=== [4/4] Multi-GPU PPO (2 XPUs, FSDP+XCCL, 2 steps) ==="
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    data.max_response_length=64 \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    critic.use_dynamic_bsz=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=xpu-e2e-ppo \
    trainer.experiment_name=xpu-ppo-2gpu-smoke \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2
echo "[OK] Multi-GPU PPO (FSDP+XCCL) passed"

echo ""
echo "=============================="
echo "  XPU E2E validation: PASSED  "
echo "=============================="
