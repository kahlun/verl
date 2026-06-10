#!/usr/bin/env bash
# Reproduce the UNPATCHED sglang + XPU weight-sync bug.
#
# Purpose:
#   Run veRL GRPO with sglang rollout on XPU with NO custom patches applied.
#   Expected failure: update_weights takes ~88s (full tensor HTTP serialization)
#   because PyTorch's ForkingPickler has no _share_xpu_() path, and
#   monkey_patch_torch_reductions() corrupts the XPU pickle tuple.
#
# To see the patched (fast) result instead, use run_grpo_xpu.sh with
# actor_rollout_ref.rollout.name=sglang.
#
# Requirements:
#   - PT 2.13 nightly XPU  (torch==2.13.0.dev20260607+xpu or similar)
#   - sglang installed (unmodified — no SHM patch in utils/common.py)
#   - 1 GPU only (2-GPU sglang OOMs on RAM — separate issue)
#
# Usage:
#   bash tests/special_xpu/run_grpo_sglang_unpatched.sh

set -x

# ── env ────────────────────────────────────────────────────────────────────────
export CCL_ATL_SHM=${CCL_ATL_SHM:-1}
export CCL_BUFFER_CACHE=${CCL_BUFFER_CACHE:-0}
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_TOPO_ALGO=0
export ZE_AFFINITY_MASK=0          # 1 GPU only
unset  ONEAPI_DEVICE_SELECTOR      # prevent oneDNN OpenCL crash
export RAY_NUM_PRESTART_PYTHON_WORKERS=0

# offline — no internet on this machine; model is in ~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── INTENTIONALLY remove the SHM / IPC patches ────────────────────────────────
#
# The two patches we applied in-place to the sglang venv are:
#
# 1. sglang/srt/utils/patch_torch.py — monkey_patch_torch_reductions() early-return on XPU
#    Without this: sglang patches ForkingPickler at index 6 (CUDA device UUID slot).
#    XPU's ForkingPickler tuple is shorter; accessing index 6 raises IndexError:
#
#      File ".../sglang/srt/utils/patch_torch.py", line 79, in _reduce_tensor_modified
#        output_args = _modify_tuple(output_args, 6, _device_to_uuid)
#      File ".../sglang/srt/utils/patch_torch.py", line 108, in _modify_tuple
#        return *t[:6], modifier(t[6]), *t[7:]
#      IndexError: tuple index out of range
#
# 2. sglang/srt/utils/common.py — MultiprocessingSerializer XPU SHM path
#    Without this: _USE_SHM=False (XPU not detected, or old code path).
#    ForkingPickler is called on XPU tensor → falls through to rebuild_tensor
#    (CPU/generic path) which serializes the full storage bytes into the pipe.
#    For a 0.5B model ~2 GB must be sent over localhost HTTP per step.
#
# To reproduce the crash/slowness, set this env var which the patched
# common.py checks to disable SHM even on XPU:
export SGLANG_DISABLE_XPU_SHM=1

# ── run ────────────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
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
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_xpu_sglang_unpatched' \
    trainer.experiment_name='qwen2_5_05b_xpu_sglang_unpatched' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=1 $@
