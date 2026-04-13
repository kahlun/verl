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

torchrun --nproc-per-node=${NUM_GPUS} --standalone \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.max_length=1024 \
    model.path="${MODEL_PATH}" \
    model.trust_remote_code=True \
    trainer.default_local_dir=./checkpoints/xpu_sft_test \
    trainer.project_name='verl_xpu_sft_e2e' \
    trainer.experiment_name='qwen2_5_05b_xpu_sft' \
    trainer.logger=console \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    trainer.micro_batch_size_per_gpu=1 \
    trainer.save_freq=-1 $@
