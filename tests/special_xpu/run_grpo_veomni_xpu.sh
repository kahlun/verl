#!/usr/bin/env bash
# E2E GRPO test on Intel XPU with VeOmni engine
#
# Validates the full RL training loop using VeOmni (FSDP2) on XPU:
#   VeOmni training (actor + ref) → vLLM rollout → reward → train
#
# Prerequisites:
#   - PyTorch with XPU support (torch.xpu.is_available() == True)
#   - veomni >= 0.1.9 with XPU patches (B1/B2/B3/B5)
#   - vLLM >= 0.17 with XPU platform support
#   - oneCCL for xccl distributed backend
#
# Usage:
#   NUM_GPUS=2 bash tests/special_xpu/run_grpo_veomni_xpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}

# oneCCL workarounds for multi-GPU (pre-DLE 2026.0 driver)
export CCL_ATL_SHM=${CCL_ATL_SHM:-1}
export CCL_BUFFER_CACHE=${CCL_BUFFER_CACHE:-0}
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_TOPO_ALGO=0

# XPU device selection
_DEVICES=$(seq 0 $((NUM_GPUS-1)) | paste -sd',')
export ZE_AFFINITY_MASK="${_DEVICES}"

# Reduce Ray prestart workers to avoid L0 context memory pressure
export RAY_NUM_PRESTART_PYTHON_WORKERS=0

python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
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
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.veomni.fsdp_size=${NUM_GPUS} \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.actor.veomni.force_use_huggingface=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.ref.veomni.optimizer_offload=True \
    actor_rollout_ref.ref.veomni.force_use_huggingface=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_xpu_veomni_e2e' \
    trainer.experiment_name='qwen2_5_05b_xpu_veomni_grpo' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} $@
