#!/usr/bin/env python3
"""Real VLM training on XPU — Qwen2-VL-2B with image+text data.

Uses verl's actual components:
  - MultiTurnSFTDataset (with real image processing, position_ids, loss_mask)
  - verl monkey patches (qwen2_vl_attn_forward → xpu_varlen_sdpa)
  - Pokemon image-caption dataset (real images, real tokenization)
  - Standard cross-entropy loss on assistant tokens only

This exercises the full VLM forward/backward path that verl's SFT trainer
would use, minus the FSDP/Ray orchestration layer.

Usage:
  ZE_AFFINITY_MASK=3 python train_vlm_real_xpu.py
"""

import os
import sys
import time
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "xpu"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DATA_PATH = "/home/sdp/data/pokemon-gpt4o-captions/train.parquet"
MAX_LENGTH = 512          # keep short for single-GPU memory
MICRO_BATCH_SIZE = 1      # single sample per step (images are large)
NUM_STEPS = 10
LR = 1e-5


def main():
    print(f"{'='*70}")
    print(f"  Real VLM Training: Qwen2-VL-2B on XPU")
    print(f"  ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', 'not set')}")
    print(f"  PyTorch {torch.__version__}, dtype={DTYPE}")
    print(f"{'='*70}\n")

    assert torch.xpu.is_available(), "XPU not available"
    print(f"  Device: {torch.xpu.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 1. Load model + processor
    # ------------------------------------------------------------------
    print("\n[1/5] Loading model and processor...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        attn_implementation="eager",
    ).to(DEVICE)
    model.train()
    model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded: {num_params/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    mem_after_load = torch.xpu.memory_allocated() / 1e9
    print(f"  Memory after load: {mem_after_load:.2f} GB")

    # ------------------------------------------------------------------
    # 2. Apply verl monkey patches (this is what verl does in training)
    # ------------------------------------------------------------------
    print("\n[2/5] Applying verl monkey patches...")
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    apply_monkey_patch(
        model,
        ulysses_sp_size=1,
        use_remove_padding=True,    # enables packed-sequence attention path
        use_fused_kernels=False,
    )
    print("  Monkey patches applied — xpu_attn.py is now in the attention path")

    # ------------------------------------------------------------------
    # 3. Build dataset using verl's MultiTurnSFTDataset
    # ------------------------------------------------------------------
    print("\n[3/5] Building VLM dataset...")
    from omegaconf import OmegaConf
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    data_config = OmegaConf.create({
        "messages_key": "messages",
        "image_key": "images",
        "video_key": "videos",
        "tools_key": "tools",
        "max_length": MAX_LENGTH,
        "truncation": "right",
        "pad_mode": "right",          # fixed-length padding for simple batching
        "enable_thinking": None,
        "print_sample_freq": 0,        # don't print every sample
        "filter_overlong_samples": True,
        "filter_no_loss_samples": True,
    })

    dataset = MultiTurnSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        config=data_config,
        processor=processor,
        max_samples=50,  # small subset for testing
    )
    print(f"  Dataset: {len(dataset)} samples (filtered from pokemon-gpt4o-captions)")

    # Check a sample
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  position_ids shape: {sample['position_ids'].shape}")
    has_mm = "multi_modal_inputs" in sample
    if has_mm:
        mm_keys = list(sample["multi_modal_inputs"].keys())
        print(f"  multi_modal_inputs keys: {mm_keys}")
        if "pixel_values" in sample["multi_modal_inputs"]:
            pv = sample["multi_modal_inputs"]["pixel_values"]
            print(f"  pixel_values shape: {pv.shape}, dtype: {pv.dtype}")
        if "image_grid_thw" in sample["multi_modal_inputs"]:
            print(f"  image_grid_thw: {sample['multi_modal_inputs']['image_grid_thw']}")
    else:
        print("  WARNING: no multi_modal_inputs — images may not have loaded")

    # Simple collate (fixed-length padded, so default_collate works for core tensors)
    def vlm_collate_fn(batch):
        """Collate that handles multi_modal_inputs separately."""
        core_keys = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
        result = {}
        for k in core_keys:
            if k in batch[0]:
                result[k] = torch.stack([b[k] for b in batch])
        # multi_modal_inputs: concat pixel_values, image_grid_thw across batch
        if "multi_modal_inputs" in batch[0]:
            mm = {}
            mm_keys = batch[0]["multi_modal_inputs"].keys()
            for k in mm_keys:
                vals = [b["multi_modal_inputs"][k] for b in batch if k in b.get("multi_modal_inputs", {})]
                if vals and isinstance(vals[0], torch.Tensor):
                    mm[k] = torch.cat(vals, dim=0)
            result["multi_modal_inputs"] = mm
        return result

    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        collate_fn=vlm_collate_fn,
        num_workers=0,  # images need main process
    )

    # ------------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------------
    print("\n[4/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    print(f"  AdamW lr={LR}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"\n[5/5] Training for {NUM_STEPS} steps...\n")
    print(f"  {'Step':>4}  {'Loss':>8}  {'GradNorm':>10}  {'Mem(GB)':>8}  {'Time(s)':>8}")
    print(f"  {'-'*50}")

    step = 0
    losses = []
    data_iter = iter(dataloader)

    for step_i in range(NUM_STEPS):
        step += 1
        t0 = time.time()

        # Get next batch (cycle if dataset is small)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(DEVICE)
        position_ids = batch["position_ids"].to(DEVICE)
        loss_mask = batch["loss_mask"].to(DEVICE)

        # Qwen2-VL position_ids: dataset returns (4, seq), collate stacks to (B, 4, seq)
        # Model expects (4, B, seq) — transpose the first two dims
        if position_ids.ndim == 3:
            position_ids = position_ids.permute(1, 0, 2)  # (B, 4, S) → (4, B, S)

        # Prepare model kwargs
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        # Add multi-modal inputs (pixel_values, image_grid_thw, etc.)
        if "multi_modal_inputs" in batch:
            for k, v in batch["multi_modal_inputs"].items():
                if isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.to(DEVICE)

        # Forward pass
        with torch.autocast(device_type="xpu", dtype=DTYPE):
            outputs = model(**model_kwargs)

        # Compute cross-entropy loss on assistant tokens only
        logits = outputs.logits  # (B, S, V)
        # Shift: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous().float()

        # Per-token cross entropy
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        # Masked average (only assistant tokens)
        num_loss_tokens = shift_loss_mask.sum().clamp(min=1)
        loss = (per_token_loss * shift_loss_mask).sum() / num_loss_tokens

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Grad norm (for monitoring)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        # Step
        optimizer.step()

        dt = time.time() - t0
        mem = torch.xpu.max_memory_allocated() / 1e9
        loss_val = loss.item()
        losses.append(loss_val)

        has_nan = torch.isnan(loss).item() or (grad_norm != grad_norm)
        nan_tag = " *** NaN! ***" if has_nan else ""

        print(f"  {step:4d}  {loss_val:8.4f}  {grad_norm:10.4f}  {mem:8.2f}  {dt:8.2f}{nan_tag}")

        # Free memory
        del outputs, logits, shift_logits, per_token_loss, loss
        torch.xpu.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Training Complete")
    print(f"  Steps: {step}")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    any_nan = any(l != l for l in losses)
    print(f"  NaN detected: {'YES' if any_nan else 'No'}")
    print(f"  Peak memory: {torch.xpu.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  Loss trend: {'decreasing ✓' if losses[-1] < losses[0] else 'not decreasing (expected for few steps)'}")
    print(f"{'='*70}")

    # Cleanup
    del model, optimizer
    gc.collect()
    torch.xpu.empty_cache()

    return 0 if not any_nan else 1


if __name__ == "__main__":
    sys.exit(main())
