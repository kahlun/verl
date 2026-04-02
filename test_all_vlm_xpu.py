#!/usr/bin/env python3
"""Test ALL verl-supported VLM models on XPU — real image training, 3 steps each.

For each model:
  1. Load model + processor (bf16, eager attn)
  2. Apply verl monkey patches
  3. Build dataset from pokemon-gpt4o-captions (real images)
  4. Run 3 training steps with cross-entropy loss on assistant tokens
  5. Report: loss, grad_norm, NaN status, peak memory, time/step

Models tested:
  - Qwen2-VL-2B-Instruct    (model_type=qwen2_vl,   XPU attn patched)
  - Qwen2.5-VL-3B-Instruct  (model_type=qwen2_5_vl, XPU attn patched)
  - Qwen3-VL-2B-Instruct    (model_type=qwen3_vl,   HF native)
  - Kimi-VL-A3B-Instruct    (model_type=kimi_vl,     XPU attn patched)

GLM-4V (glm4v) excluded: smallest GLM-4V is 9B — won't fit on 24GB with grad ckpt.

Usage:
  ZE_AFFINITY_MASK=3 python test_all_vlm_xpu.py [--steps 3] [--max-length 384]

Also generates: test_all_vlm_cuda.py (identical script for NVIDIA GPU comparison)
"""

import os
import sys
import time
import gc
import json
import argparse
import traceback

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = os.environ.get("VLM_TEST_DEVICE", "xpu" if torch.xpu.is_available() else "cuda")
DTYPE = torch.bfloat16
DATA_PATH = "/home/sdp/data/pokemon-gpt4o-captions/train.parquet"

# Models to test: (name, model_type, xpu_attn_patched, max_length, notes)
VLM_MODELS = [
    {
        "name": "Qwen/Qwen2-VL-2B-Instruct",
        "model_type": "qwen2_vl",
        "xpu_attn": True,
        "max_length": 384,
        "notes": "XPU attn: xpu_varlen_sdpa + xpu_flash_attention_forward",
    },
    {
        "name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_type": "qwen2_5_vl",
        "xpu_attn": True,
        "max_length": 384,
        "notes": "Shares qwen2_vl.py code path with Qwen2-VL",
    },
    {
        "name": "Qwen/Qwen3-VL-2B-Instruct",
        "model_type": "qwen3_vl",
        "xpu_attn": False,
        "max_length": 384,
        "notes": "Uses HF native eager attention (no xpu_attn patch needed)",
    },
    {
        "name": "moonshotai/Kimi-VL-A3B-Instruct",
        "model_type": "kimi_vl",
        "xpu_attn": True,
        "max_length": 256,  # Kimi-VL has more overhead from MoE
        "notes": "DeepSeek-V3 arch with MoE, xpu_flash_attention_forward patched",
    },
]


def get_model_class(model_name):
    """Auto-detect the right model class from transformers."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    mt = config.model_type

    if mt == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    elif mt == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    elif mt == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif mt == "kimi_vl":
        # KimiVL uses trust_remote_code — use AutoModelForCausalLM
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    else:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq


def build_dataset(model_name, tokenizer, processor, max_length, max_samples=20):
    """Build verl's MultiTurnSFTDataset with the pokemon data."""
    from omegaconf import OmegaConf
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    data_config = OmegaConf.create({
        "messages_key": "messages",
        "image_key": "images",
        "video_key": "videos",
        "tools_key": "tools",
        "max_length": max_length,
        "truncation": "right",
        "pad_mode": "right",
        "enable_thinking": None,
        "print_sample_freq": 0,
        "filter_overlong_samples": True,
        "filter_no_loss_samples": True,
    })

    dataset = MultiTurnSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        config=data_config,
        processor=processor,
        max_samples=max_samples,
    )
    return dataset


def vlm_collate_fn(batch):
    """Collate that handles multi_modal_inputs separately."""
    core_keys = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
    result = {}
    for k in core_keys:
        if k in batch[0]:
            result[k] = torch.stack([b[k] for b in batch])
    if "multi_modal_inputs" in batch[0]:
        mm = {}
        for k in batch[0]["multi_modal_inputs"].keys():
            vals = [b["multi_modal_inputs"][k] for b in batch
                    if k in b.get("multi_modal_inputs", {})]
            if vals and isinstance(vals[0], torch.Tensor):
                mm[k] = torch.cat(vals, dim=0)
        result["multi_modal_inputs"] = mm
    return result


def train_one_model(model_info, num_steps, override_max_length=None):
    """Run training on a single VLM model. Returns results dict."""
    model_name = model_info["name"]
    max_length = override_max_length or model_info["max_length"]
    results = {
        "model": model_name,
        "model_type": model_info["model_type"],
        "xpu_attn": model_info["xpu_attn"],
        "status": "FAIL",
        "steps": [],
        "error": None,
    }

    try:
        # 1. Load model
        print(f"    Loading model...")
        from transformers import AutoProcessor

        ModelClass = get_model_class(model_name)
        model = ModelClass.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(DEVICE)
        model.train()
        model.gradient_checkpointing_enable()

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        mem_load = torch.xpu.memory_allocated() / 1e9 if DEVICE == "xpu" else torch.cuda.memory_allocated() / 1e9
        results["params_m"] = round(num_params, 1)
        results["mem_after_load_gb"] = round(mem_load, 2)
        print(f"    {num_params:.0f}M params, {mem_load:.2f} GB after load")

        # 2. Apply verl monkey patches
        print(f"    Applying monkey patches...")
        from verl.models.transformers.monkey_patch import apply_monkey_patch
        apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=True, use_fused_kernels=False)

        # 3. Build dataset
        print(f"    Building dataset (max_length={max_length})...")
        dataset = build_dataset(model_name, tokenizer, processor, max_length)
        if len(dataset) == 0:
            results["error"] = "Dataset empty after filtering"
            return results
        print(f"    Dataset: {len(dataset)} samples")

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                collate_fn=vlm_collate_fn, num_workers=0)

        # 4. Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

        # 5. Training loop
        print(f"    Training {num_steps} steps...")
        data_iter = iter(dataloader)

        for step_i in range(num_steps):
            t0 = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(DEVICE)
            position_ids = batch["position_ids"].to(DEVICE)
            loss_mask = batch["loss_mask"].to(DEVICE)

            # Qwen2-VL/Qwen2.5-VL: (B, 4, S) → (4, B, S)
            if position_ids.ndim == 3:
                position_ids = position_ids.permute(1, 0, 2)

            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

            if "multi_modal_inputs" in batch:
                for k, v in batch["multi_modal_inputs"].items():
                    if isinstance(v, torch.Tensor):
                        model_kwargs[k] = v.to(DEVICE)

            # Forward
            with torch.autocast(device_type=DEVICE.split(":")[0], dtype=DTYPE):
                outputs = model(**model_kwargs)

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_loss_mask = loss_mask[:, 1:].contiguous().float()

            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            num_loss_tokens = shift_loss_mask.sum().clamp(min=1)
            loss = (per_token_loss * shift_loss_mask).sum() / num_loss_tokens

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()

            dt = time.time() - t0
            if DEVICE == "xpu":
                mem = torch.xpu.max_memory_allocated() / 1e9
            else:
                mem = torch.cuda.max_memory_allocated() / 1e9
            loss_val = loss.item()
            has_nan = torch.isnan(loss).item() or (grad_norm != grad_norm)

            step_result = {
                "step": step_i + 1,
                "loss": round(loss_val, 4),
                "grad_norm": round(grad_norm, 4),
                "mem_gb": round(mem, 2),
                "time_s": round(dt, 2),
                "nan": has_nan,
            }
            results["steps"].append(step_result)

            tag = "NaN!" if has_nan else ""
            print(f"      step {step_i+1}: loss={loss_val:.4f} grad_norm={grad_norm:.2f} "
                  f"mem={mem:.1f}GB t={dt:.1f}s {tag}")

            del outputs, logits, shift_logits, per_token_loss, loss
            if DEVICE == "xpu":
                torch.xpu.empty_cache()
            else:
                torch.cuda.empty_cache()

        results["status"] = "PASS"
        results["peak_mem_gb"] = results["steps"][-1]["mem_gb"] if results["steps"] else 0
        results["any_nan"] = any(s["nan"] for s in results["steps"])

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"    ERROR: {e}")

    finally:
        # Cleanup
        try:
            del model, optimizer
        except NameError:
            pass
        gc.collect()
        if DEVICE == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats()
        else:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model indices (0-based) or 'all'")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  VERL VLM Model Test Suite — All Supported Models")
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  dtype: {DTYPE}")
    print(f"  Steps per model: {args.steps}")
    print(f"{'='*70}\n")

    if DEVICE == "xpu":
        print(f"  XPU: {torch.xpu.get_device_name(0)}")
        print(f"  ZE_AFFINITY_MASK: {os.environ.get('ZE_AFFINITY_MASK', 'not set')}")
    elif DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()

    # Select models
    if args.models and args.models != "all":
        indices = [int(x) for x in args.models.split(",")]
        models_to_test = [VLM_MODELS[i] for i in indices]
    else:
        models_to_test = VLM_MODELS

    all_results = []
    for i, minfo in enumerate(models_to_test):
        print(f"  [{i+1}/{len(models_to_test)}] {minfo['name']} (type={minfo['model_type']})")
        print(f"    {minfo['notes']}")
        result = train_one_model(minfo, args.steps, args.max_length)
        all_results.append(result)
        print(f"    → {result['status']}" +
              (f" (error: {result['error']})" if result['error'] else "") +
              "\n")

    # ──────────── Summary Table ────────────
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Type':<12} {'Status':<6} {'Loss':<14} {'NaN':<4} {'Mem(GB)':<8} {'t/step':<6}")
    print(f"  {'-'*95}")

    for r in all_results:
        name = r["model"].split("/")[-1][:34]
        mt = r["model_type"]
        status = r["status"]
        if r["steps"]:
            loss_str = f"{r['steps'][0]['loss']:.2f}→{r['steps'][-1]['loss']:.2f}"
            nan_str = "Yes" if r.get("any_nan") else "No"
            mem_str = f"{r.get('peak_mem_gb', 0):.1f}"
            avg_t = sum(s["time_s"] for s in r["steps"]) / len(r["steps"])
            t_str = f"{avg_t:.1f}s"
        else:
            loss_str = "N/A"
            nan_str = "N/A"
            mem_str = "N/A"
            t_str = "N/A"

        print(f"  {name:<35} {mt:<12} {status:<6} {loss_str:<14} {nan_str:<4} {mem_str:<8} {t_str:<6}")

    print(f"  {'-'*95}")
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    print(f"  {passed} PASS, {failed} FAIL out of {len(all_results)} models")
    print(f"{'='*70}\n")

    # Save results JSON for comparison
    results_file = os.path.join(os.path.dirname(__file__),
                                f"vlm_test_results_{DEVICE.replace(':','_')}.json")
    with open(results_file, "w") as f:
        json.dump({
            "device": DEVICE,
            "device_name": (torch.xpu.get_device_name(0) if DEVICE == "xpu"
                           else torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"),
            "pytorch": torch.__version__,
            "dtype": str(DTYPE),
            "results": all_results,
        }, f, indent=2)
    print(f"  Results saved to: {results_file}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
