---
name: xpu-test
description: Run XPU validation tests — device abstractions, single-GPU training, VLM forward, distributed seqlen_balancing. Use after making code changes to verify nothing is broken on XPU. Pass an argument to run a specific subset.
argument-hint: "[all|unit|gpu|vlm|distributed] default: all"
---

Run XPU validation tests at `/home/sdp/kl/verl_test_xpu`.

Test set: $ARGUMENTS (default: all — but skip distributed unless user explicitly asks)

## Device smoke test (always run first)

```bash
cd /home/sdp/kl/verl_test_xpu
python3 -c "
from verl.utils.device import (get_device_name, get_nccl_backend, is_xpu_available,
    get_default_attention_implementation, get_resource_name)
import torch
assert is_xpu_available, 'XPU not available!'
assert get_device_name() == 'xpu'
assert get_nccl_backend() == 'xccl'
assert get_default_attention_implementation() == 'eager'
assert get_resource_name() == 'xpu'
from verl.utils.device import get_torch_device
assert get_torch_device() == torch.xpu
from verl.workers.engine import EngineRegistry
assert 'xpu' in list(EngineRegistry._engines.get('language_model', {}).get('fsdp', {}).keys())
print('Device smoke test: PASS')
" 2>&1 | grep -v WARNING
```

## CPU unit tests (fast, ~30 seconds)

```bash
cd /home/sdp/kl/verl_test_xpu
python3 -m pytest \
  tests/test_base_config_on_cpu.py \
  tests/test_protocol_on_cpu.py \
  tests/trainer/ppo/test_core_algos_on_cpu.py \
  tests/trainer/ppo/test_metric_utils_on_cpu.py \
  tests/workers/config/ \
  tests/interactions/test_interaction_registry.py \
  tests/utils/test_import_utils_on_cpu.py \
  -v --tb=line -q 2>&1 | tail -10
```

## Single-GPU XPU training test (~2 min)

```bash
cd /home/sdp/kl/verl_test_xpu
LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29510 RANK=0 WORLD_SIZE=1 python3 -c "
import torch
from verl.utils.distributed import initialize_global_process_group
from verl.utils.device import get_default_attention_implementation
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

initialize_global_process_group()
config = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',
    attn_implementation=get_default_attention_implementation())
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',
    config=config, torch_dtype=torch.bfloat16).to('xpu')
model = FSDP(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
inputs = tokenizer('Test XPU training', return_tensors='pt').to('xpu')
model.train()
for step in range(3):
    out = model(**inputs, labels=inputs.input_ids)
    out.loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); optimizer.zero_grad()
    assert not out.loss.isnan().item(), f'NaN at step {step}!'
    print(f'step={step} loss={out.loss.item():.4f} grad_norm={gn.item():.4f}')
print(f'Peak mem: {torch.xpu.max_memory_allocated()/1024**3:.2f} GB')
print('Single-GPU FSDP training: PASS')
torch.distributed.destroy_process_group()
" 2>&1 | grep -E "step=|Peak|PASS|Error|assert|NaN"
```

## Fused kernel test (if use_fused_kernels=True is being considered)

```bash
cd /home/sdp/kl/verl_test_xpu
python3 -c "
import torch
from verl.utils.kernel.kernels import efficient_entropy_forward
hidden = torch.randn(8, 128, device='xpu', dtype=torch.bfloat16).contiguous()
weight = torch.randn(1024, 128, device='xpu', dtype=torch.bfloat16).contiguous()
labels = torch.randint(0, 1024, (8,), device='xpu')
result = efficient_entropy_forward(hidden, weight, labels)
logprobs, entropy = result[0], result[1]
assert not entropy.isnan().any(), 'NaN in fused kernel output!'
print(f'Fused entropy kernel: PASS, sample={entropy[:3].tolist()}')
" 2>&1 | grep -E "PASS|Error|NaN|assert"
```

## Distributed test (2-GPU, only run if user approves multi-GPU)

**Note: Only run if user has approved multi-GPU testing.**

```bash
cd /home/sdp/kl/verl_test_xpu
torchrun --standalone --nproc_per_node=2 \
  -m pytest tests/utils/test_seqlen_balancing.py -v --tb=short 2>&1 | tail -10
```

## Report format

For each test, print: PASS / FAIL and key metrics (loss, grad_norm, peak_mem, NaN count).
Flag any regressions vs known-good baseline from `benchmark/results_xpu_b60.json`.
