---
name: xpu-vlm-test
description: Test VLM model (Qwen2-VL) on XPU — verifies xpu_attn.py correctness, monkey_patch integration, forward/backward, sequence packing. Use when xpu_attn.py or monkey_patch.py is modified.
argument-hint: "[quick|full] default: full"
---

Test VLM XPU support at `/home/sdp/kl/verl_test_xpu`.

Mode: $ARGUMENTS (quick = imports + correctness only; full = imports + correctness + model training)

## Step 1: Verify XPU imports in VLM files

```bash
cd /home/sdp/kl/verl_test_xpu
python3 -c "
from verl.utils.device import is_xpu_available
assert is_xpu_available

from verl.models.transformers.xpu_attn import xpu_varlen_sdpa, xpu_flash_attention_forward
print('xpu_attn.py: OK')

from verl.models.transformers import qwen2_vl, glm4v, kimi_vl
assert 'xpu_attn' in qwen2_vl.flash_attn_varlen_func.__module__, f'qwen2_vl not redirected: {qwen2_vl.flash_attn_varlen_func}'
assert 'xpu_attn' in glm4v.flash_attn_varlen_func.__module__, f'glm4v not redirected'
assert 'xpu_attn' in kimi_vl._flash_attention_forward.__module__, f'kimi_vl not redirected'
print('VLM redirects: OK')
" 2>&1 | grep -v WARNING
```

## Step 2: xpu_varlen_sdpa correctness

```bash
cd /home/sdp/kl/verl_test_xpu
python3 -c "
import torch, torch.nn.functional as F
from verl.models.transformers.xpu_attn import xpu_varlen_sdpa, xpu_flash_attention_forward

# Test 1: varlen SDPA — 3 packed sequences
torch.manual_seed(42)
q = torch.randn(13, 4, 32, device='xpu', dtype=torch.bfloat16)
k = torch.randn(13, 4, 32, device='xpu', dtype=torch.bfloat16)
v = torch.randn(13, 4, 32, device='xpu', dtype=torch.bfloat16)
cu = torch.tensor([0, 4, 10, 13], device='xpu', dtype=torch.int32)
out = xpu_varlen_sdpa(q, k, v, cu, cu, 6, 6, causal=True)

for i in range(3):
    s, e = cu[i].item(), cu[i+1].item()
    ref = F.scaled_dot_product_attention(
        q[s:e].unsqueeze(0).transpose(1,2),
        k[s:e].unsqueeze(0).transpose(1,2),
        v[s:e].unsqueeze(0).transpose(1,2), is_causal=True
    ).squeeze(0).transpose(0,1)
    diff = (out[s:e] - ref).abs().max().item()
    assert diff < 1e-3, f'Seq {i}: diff={diff}'
    print(f'  seq{i} (len={e-s}): diff={diff:.8f} OK')

# Test 2: xpu_flash_attention_forward — cross-sequence isolation
pos = torch.tensor([[0,1,2,3,4,0,1,2]], device='xpu')
q2 = torch.randn(1,8,4,32, device='xpu', dtype=torch.bfloat16)
k2 = torch.randn(1,8,4,32, device='xpu', dtype=torch.bfloat16)
v2 = torch.randn(1,8,4,32, device='xpu', dtype=torch.bfloat16)
out2 = xpu_flash_attention_forward(q2, k2, v2, None, 8, is_causal=True, position_ids=pos)
ref2 = F.scaled_dot_product_attention(q2[:,5:].transpose(1,2), k2[:,5:].transpose(1,2),
    v2[:,5:].transpose(1,2), is_causal=True).transpose(1,2)
diff2 = (out2[:,5:] - ref2).abs().max().item()
assert diff2 < 1e-3, f'Cross-seq leakage: diff={diff2}'
print(f'Cross-seq isolation: diff={diff2:.8f} OK')
print('xpu_attn.py correctness: PASS')
" 2>&1 | grep -E "seq|Cross|PASS|Error|assert"
```

## Step 3: monkey_patch integration (model must be available)

```bash
cd /home/sdp/kl/verl_test_xpu
python3 -c "
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoConfig
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.device import get_default_attention_implementation

config = AutoConfig.from_pretrained('Qwen/Qwen2-VL-2B-Instruct',
    attn_implementation=get_default_attention_implementation())
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct', config=config, torch_dtype=torch.bfloat16).to('xpu')
apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=True)
print('monkey_patch: PASS')
" 2>&1 | grep -E "PASS|Error|XPU|Monkey patch"
```

## Step 4: Full VLM training (only in 'full' mode)

```bash
cd /home/sdp/kl/verl_test_xpu
python3 -c "
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct', torch_dtype=torch.bfloat16,
    attn_implementation='sdpa').to('xpu')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is 2+2?'}]}]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors='pt').to('xpu')
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for step in range(3):
    out = model(**inputs, labels=inputs.input_ids)
    assert not out.loss.isnan().item(), f'NaN at step {step}'
    out.loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); optimizer.zero_grad()
    print(f'step={step} loss={out.loss.item():.4f} nan=False')
print(f'Peak: {torch.xpu.max_memory_allocated()/1024**3:.2f} GB')
print('VLM SDPA training: PASS')
" 2>&1 | grep -E "step=|Peak|PASS|Error|NaN"
```

## Expected results

- All redirects: `xpu_attn` module
- varlen SDPA diff: < 1e-3 (typically 0.0)
- Cross-sequence leakage: 0.0
- VLM 3-step loss: decreasing, zero NaN
- Peak memory: ~15-16 GB for 2B model
