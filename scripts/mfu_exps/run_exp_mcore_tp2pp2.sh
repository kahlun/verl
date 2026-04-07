#!/usr/bin/env bash
# MFU Experiment: Megatron 4-GPU Llama-3.1-8B TP=2, PP=2, DP=1
set -euo pipefail

LLAMA_PATH=/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
DATA_DIR=/root/data/gsm8k_sft
LOG_DIR=/workspace/verl/mfu_comparison
CKPT_DIR=/root/verl/ckpts/mfu-mcore-llama8b-tp2-pp2

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd /workspace/verl

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m verl.trainer.sft_trainer \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=1 \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu=1024 \
  data.messages_key=messages \
  data.ignore_input_ids_mismatch=True \
  model.use_remove_padding=True \
  model=hf_model \
  model.path="${LLAMA_PATH}" \
  model.trust_remote_code=True \
  engine=megatron \
  optim=megatron \
  optim.lr=1e-5 \
  optim.min_lr=1e-6 \
  optim.lr_warmup_steps=6 \
  optim.weight_decay=0.1 \
  'optim.betas=[0.9,0.95]' \
  optim.clip_grad=1.0 \
  engine.tensor_model_parallel_size=2 \
  engine.pipeline_model_parallel_size=2 \
  engine.virtual_pipeline_model_parallel_size=null \
  engine.context_parallel_size=1 \
  engine.sequence_parallel=True \
  engine.use_mbridge=True \
  engine.vanilla_mbridge=True \
  engine.dtype=bfloat16 \
  engine.use_distributed_optimizer=True \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  'trainer.logger=[console,file]' \
  trainer.project_name=mfu_comparison \
  trainer.experiment_name=mcore-llama8b-tp2-pp2 \
  trainer.total_epochs=999 \
  trainer.total_training_steps=30 \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode=disable \
  2>&1 | tee "${LOG_DIR}/exp_mcore_tp2pp2.log"
