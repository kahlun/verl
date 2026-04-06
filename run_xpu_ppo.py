#!/usr/bin/env python3
"""Pre-initialize Ray with XPU GPU count, then run VERL main_ppo.
Workaround: Ray doesn't auto-detect Intel XPU as GPU resources.
Usage: python3 run_xpu_ppo.py [all normal verl hydra args...]
"""
import sys
import torch
import ray

n_xpu = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0
if n_xpu > 0 and not ray.is_initialized():
    # Register "xpu" custom resource — VERL workers request {"xpu": N}, not num_gpus
    ray.init(resources={"xpu": n_xpu})
    print(f"[run_xpu_ppo] Ray initialized with resources={{xpu: {n_xpu}}} (XPU devices)")

# Now run main_ppo — it will see ray.is_initialized()==True and skip re-init
from verl.trainer.main_ppo import main
main()
