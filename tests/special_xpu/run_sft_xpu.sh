#!/usr/bin/env bash
# SFT smoke test on Intel XPU — validates FSDP multi-GPU training without vLLM.
#
# Usage:
#   NUM_GPUS=4 bash tests/special_xpu/run_sft_xpu.sh

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
# SFT uses torchrun (no vLLM) but keeping the same env var as GRPO/PPO avoids
# confusion and works correctly with Level Zero device renumbering.
_DEVICES=$(seq 0 $((NUM_GPUS-1)) | paste -sd',')
export ZE_AFFINITY_MASK="${_DEVICES}"

# Use python -m torch.distributed.run instead of torchrun so the correct
# Python interpreter is used regardless of PATH (e.g. inside conda envs).
PYTHON3=$(python3 -c "import torch; import sys; print(sys.executable)" 2>/dev/null || command -v python3)
$PYTHON3 -m torch.distributed.run --nproc-per-node=${NUM_GPUS} --standalone \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train_sft.parquet \
    data.val_files=$HOME/data/gsm8k/test_sft.parquet \
    data.train_batch_size=8 \
    data.max_length=1024 \
    model.path="${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.use_remove_padding=False \
    +model.override_config.attn_implementation=sdpa \
    trainer.default_local_dir=./checkpoints/xpu_sft_test \
    trainer.project_name='verl_xpu_sft_e2e' \
    trainer.experiment_name='qwen2_5_05b_xpu_sft' \
    trainer.logger=console \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    data.micro_batch_size_per_gpu=1 \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=${NUM_GPUS} $@
