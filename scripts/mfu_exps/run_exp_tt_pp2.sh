#!/usr/bin/env bash
# MFU Experiment: TorchTitan 4-GPU Llama-3.1-8B PP=2, FSDP2=2
set -euo pipefail

LLAMA_PATH=/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
DATA_DIR=/root/data/gsm8k_sft
LOG_DIR=/workspace/verl/mfu_comparison
CKPT_DIR=/root/verl/ckpts/mfu-tt-llama8b-pp2-fsdp2

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd /workspace/verl

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m verl.trainer.sft_trainer \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size=128 \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu=2048 \
  data.messages_key=messages \
  data.ignore_input_ids_mismatch=True \
  model.use_remove_padding=True \
  model=hf_model \
  model.path="${LLAMA_PATH}" \
  engine=torchtitan \
  optim=torchtitan \
  optim.lr=1e-5 \
  optim.lr_warmup_steps_ratio=0.2 \
  optim.weight_decay=0.1 \
  'optim.betas=[0.9,0.95]' \
  optim.clip_grad=1.0 \
  optim.min_lr_factor=0.1 \
  optim.decay_type=cosine \
  optim.total_training_steps=1000 \
  engine.tensor_parallel_size=1 \
  engine.pipeline_parallel_size=2 \
  engine.context_parallel_size=1 \
  engine.data_parallel_shard_size=2 \
  engine.use_torch_compile=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  'trainer.logger=[console,file]' \
  trainer.project_name=mfu_comparison \
  trainer.experiment_name=tt-llama8b-pp2-fsdp2 \
  trainer.total_epochs=999 \
  trainer.total_training_steps=30 \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode=disable \
  2>&1 | tee "${LOG_DIR}/exp_tt_pp2.log"
