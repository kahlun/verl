# VERL VLM XPU vs NVIDIA Comparison Test Plan

## Objective
Validate that verl's XPU attention backend (`xpu_attn.py`) produces correct training
behavior by comparing against NVIDIA GPU baseline using identical model, data, and
training parameters.

---

## Test Matrix

| Model                     | Type         | Params | XPU Status | CUDA Expected | Notes                            |
|---------------------------|-------------|--------|------------|---------------|----------------------------------|
| Qwen2-VL-2B-Instruct     | qwen2_vl    | 2.2B   | ✅ PASS     | ✅ PASS        | Core xpu_attn test (varlen+flash)|
| Qwen2.5-VL-3B-Instruct   | qwen2_5_vl  | 3.8B   | ❌ OOM 24GB | ✅ PASS (≥32GB)| XPU B60 24GB too small           |
| Qwen3-VL-2B-Instruct     | qwen3_vl    | 2.1B   | ✅ PASS     | ✅ PASS        | Uses HF native attn, not xpu_attn|
| Kimi-VL-A3B-Instruct     | kimi_vl     | ~16B*  | ❌ OOM 24GB | ⚠️ Needs ≥40GB | MoE arch, vision encoder large   |
| GLM-4V-9B                | glm4v       | 9B     | ❌ Excluded | ✅ PASS (≥48GB)| Smallest GLM-4V is 9B            |

*Kimi-VL total params include MoE inactive experts; active params ~3B but full model still loads ~20GB.

---

## What We're Comparing

1. **Correctness**: Loss values should be similar (not identical — different SDPA implementations have different numeric precision)
2. **NaN-free training**: Both devices should produce zero NaN losses and gradient norms
3. **Memory efficiency**: Compare peak GPU memory usage
4. **Throughput**: Compare time per training step (first step excluded — JIT warmup)

### Expected Differences
- **Loss values will NOT match exactly** — XPU uses SDPA, CUDA uses FlashAttention v2. Different attention kernels produce different floating-point rounding.
- **Acceptable loss delta**: < 0.5 between XPU and CUDA for same model/data/seed is considered equivalent.
- **First step will be slower** on both platforms due to compilation/warmup.

---

## Files to Copy to NVIDIA Machine

```
# Required files:
test_all_vlm_xpu.py          # The test script (works on both XPU and CUDA)
compare_vlm_results.py        # Comparison tool (reads both JSON files)

# Required data:
data/pokemon-gpt4o-captions/train.parquet  # 1499 samples with images

# Required verl installation:
# The verl repo with monkey_patch support
```

---

## NVIDIA Setup Instructions

### 1. Environment
```bash
# Recommended: NVIDIA A100/A10G/H100 with ≥24GB VRAM
# For all models including Kimi-VL: need ≥40GB

conda create -n verl_test python=3.12
conda activate verl_test

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0 flash-attn>=2.5.0 omegaconf tiktoken qwen-vl-utils
```

### 2. Install verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl && git checkout xpu/2-training-integration  # or main if merged
pip install -e .
```

### 3. Copy Data
```bash
# Copy the pokemon dataset from the XPU machine
scp -r user@xpu-machine:/home/sdp/data/pokemon-gpt4o-captions/ data/

# Or adjust DATA_PATH in the script
```

### 4. Run the Test
```bash
# Set the device to CUDA (the script auto-detects, but you can force it)
export VLM_TEST_DEVICE=cuda

# Test models that passed on XPU (for direct comparison):
python test_all_vlm_xpu.py --steps 3 --models 0,2

# Test all models (CUDA has enough VRAM for larger models):
CUDA_VISIBLE_DEVICES=0 python test_all_vlm_xpu.py --steps 3

# For models needing more VRAM (Qwen2.5-VL-3B, Kimi-VL):
python test_all_vlm_xpu.py --steps 3 --models 1,3
```

### 5. Compare Results
```bash
# Copy vlm_test_results_cuda.json back to XPU machine, then:
python compare_vlm_results.py --xpu vlm_test_results_xpu.json --cuda vlm_test_results_cuda.json

# Or copy vlm_test_results_xpu.json to NVIDIA machine and run there
```

---

## XPU Results (Intel Arc Pro B60, 24GB)

| Model                    | Status | Loss       | NaN | Peak Mem | Avg t/step |
|--------------------------|--------|------------|-----|----------|------------|
| Qwen2-VL-2B-Instruct    | PASS   | 2.31→2.66  | No  | 22.2 GB  | 2.2s       |
| Qwen2.5-VL-3B-Instruct  | OOM    | —          | —   | >24 GB   | —          |
| Qwen3-VL-2B-Instruct    | PASS   | 2.15→2.38  | No  | 21.4 GB  | 1.1s       |
| Kimi-VL-A3B-Instruct    | OOM    | —          | —   | >24 GB   | —          |

### Key Observations
- **2/4 models PASS** on 24GB Intel Arc Pro B60
- **0 NaN** across all passing steps
- **xpu_attn.py correctly handles**:
  - Qwen2-VL packed sequence attention (varlen SDPA)
  - Qwen3-VL native HF attention (no xpu_attn needed, but _flash_attention_forward still patched)
- **Memory-limited**: 24GB insufficient for 3B+ models with full fine-tuning + gradient checkpointing

---

## Interpretation Guide

### If losses match within ~0.5:
✅ XPU attention is numerically equivalent to CUDA flash_attn. The SDPA fallback produces correct gradients.

### If XPU produces NaN but CUDA doesn't:
❌ Bug in xpu_attn.py — likely in the varlen SDPA packed-sequence handling or mask construction.

### If CUDA produces NaN but XPU doesn't:
⚠️ Likely a flash_attn version issue on CUDA side. Ensure flash-attn >= 2.5.0.

### If memory differs significantly (>2x):
📊 Expected — SDPA eager attention uses more memory than FlashAttention due to materializing the full attention matrix. On NVIDIA, flash_attn is memory-efficient (O(N) vs O(N²)).

### If throughput differs significantly:
📊 Expected — FlashAttention on NVIDIA is heavily optimized with custom CUDA kernels. XPU SDPA is a general-purpose implementation.

---

## Kimi-VL Transformers Compatibility Note

Kimi-VL's remote model code (`modeling_kimi_vl.py`) imports `PytorchGELUTanh` which was
renamed to `GELUTanh` in transformers >= 4.57. If you encounter this error on CUDA:

```python
# Fix in ~/.cache/huggingface/modules/transformers_modules/moonshotai/Kimi_hyphen_VL_hyphen_A3B_hyphen_Instruct/*/modeling_kimi_vl.py
# Line 56: change PytorchGELUTanh → GELUTanh
# Line 2350: change PytorchGELUTanh() → GELUTanh()
```
