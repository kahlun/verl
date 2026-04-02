#!/usr/bin/env python3
"""Compare VLM test results between XPU and CUDA runs.

Reads vlm_test_results_xpu.json and vlm_test_results_cuda.json,
prints a side-by-side comparison table.

Usage:
    python compare_vlm_results.py [--xpu xpu_results.json] [--cuda cuda_results.json]
"""

import argparse
import json
import os
import sys


def load_results(path):
    with open(path) as f:
        return json.load(f)


def fmt_loss(steps):
    if not steps:
        return "N/A"
    return f"{steps[0]['loss']:.2f}→{steps[-1]['loss']:.2f}"


def fmt_mem(result):
    if result.get("peak_mem_gb"):
        return f"{result['peak_mem_gb']:.1f}"
    if result.get("steps"):
        return f"{result['steps'][-1]['mem_gb']:.1f}"
    return "N/A"


def fmt_time(steps):
    if not steps:
        return "N/A"
    avg = sum(s["time_s"] for s in steps) / len(steps)
    return f"{avg:.1f}s"


def main():
    parser = argparse.ArgumentParser(description="Compare XPU vs CUDA VLM test results")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--xpu", default=os.path.join(script_dir, "vlm_test_results_xpu.json"))
    parser.add_argument("--cuda", default=os.path.join(script_dir, "vlm_test_results_cuda.json"))
    args = parser.parse_args()

    have_xpu = os.path.exists(args.xpu)
    have_cuda = os.path.exists(args.cuda)

    if not have_xpu and not have_cuda:
        print("No result files found. Run test_all_vlm_xpu.py on both devices first.")
        sys.exit(1)

    xpu_data = load_results(args.xpu) if have_xpu else None
    cuda_data = load_results(args.cuda) if have_cuda else None

    # Build model→result maps
    xpu_map = {}
    cuda_map = {}
    if xpu_data:
        for r in xpu_data["results"]:
            xpu_map[r["model"]] = r
    if cuda_data:
        for r in cuda_data["results"]:
            cuda_map[r["model"]] = r

    all_models = sorted(set(list(xpu_map.keys()) + list(cuda_map.keys())))

    # Header
    print()
    print("=" * 120)
    print("  VERL VLM Test Results — XPU vs CUDA Comparison")
    print("=" * 120)

    if xpu_data:
        print(f"  XPU: {xpu_data.get('device_name', '?')} | PyTorch {xpu_data.get('pytorch', '?')}")
    if cuda_data:
        print(f"  CUDA: {cuda_data.get('device_name', '?')} | PyTorch {cuda_data.get('pytorch', '?')}")
    print()

    # Table header
    hdr = f"  {'Model':<30} │ {'XPU Status':<10} {'XPU Loss':<14} {'XPU Mem':<8} {'XPU t/s':<8}"
    hdr += f"│ {'CUDA Status':<11} {'CUDA Loss':<14} {'CUDA Mem':<9} {'CUDA t/s':<8}"
    hdr += f"│ {'Speedup':<8}"
    print(hdr)
    print(f"  {'─' * 29}─┼─{'─' * 42}┼─{'─' * 43}┼─{'─' * 8}")

    for model in all_models:
        name = model.split("/")[-1][:29]
        xr = xpu_map.get(model, {})
        cr = cuda_map.get(model, {})

        x_status = xr.get("status", "—")
        x_loss = fmt_loss(xr.get("steps", []))
        x_mem = fmt_mem(xr)
        x_time = fmt_time(xr.get("steps", []))

        c_status = cr.get("status", "—")
        c_loss = fmt_loss(cr.get("steps", []))
        c_mem = fmt_mem(cr)
        c_time = fmt_time(cr.get("steps", []))

        # Compute speedup (CUDA time / XPU time)
        speedup = "—"
        if xr.get("steps") and cr.get("steps"):
            x_avg = sum(s["time_s"] for s in xr["steps"]) / len(xr["steps"])
            c_avg = sum(s["time_s"] for s in cr["steps"]) / len(cr["steps"])
            if x_avg > 0:
                ratio = c_avg / x_avg
                speedup = f"{ratio:.2f}x"

        row = f"  {name:<30} │ {x_status:<10} {x_loss:<14} {x_mem:<8} {x_time:<8}"
        row += f"│ {c_status:<11} {c_loss:<14} {c_mem:<9} {c_time:<8}"
        row += f"│ {speedup:<8}"
        print(row)

    print(f"  {'─' * 29}─┼─{'─' * 42}┼─{'─' * 43}┼─{'─' * 8}")
    print()

    # Detailed per-step comparison for passing models
    for model in all_models:
        xr = xpu_map.get(model, {})
        cr = cuda_map.get(model, {})
        if not (xr.get("steps") and cr.get("steps")):
            continue

        name = model.split("/")[-1]
        print(f"  Per-step detail: {name}")
        print(f"    {'Step':<6} │ {'XPU Loss':<10} {'XPU Grad':<10} {'XPU t(s)':<10} │ {'CUDA Loss':<10} {'CUDA Grad':<10} {'CUDA t(s)':<10} │ {'Loss Δ':<10}")
        print(f"    {'─' * 5}─┼─{'─' * 30}┼─{'─' * 31}┼─{'─' * 10}")

        for xs, cs in zip(xr["steps"], cr["steps"]):
            loss_delta = abs(xs["loss"] - cs["loss"])
            row = f"    {xs['step']:<6} │ {xs['loss']:<10.4f} {xs['grad_norm']:<10.2f} {xs['time_s']:<10.2f} │ "
            row += f"{cs['loss']:<10.4f} {cs['grad_norm']:<10.2f} {cs['time_s']:<10.2f} │ {loss_delta:<10.4f}"
            print(row)
        print()

    # Summary stats
    x_pass = sum(1 for r in xpu_map.values() if r.get("status") == "PASS") if xpu_map else 0
    c_pass = sum(1 for r in cuda_map.values() if r.get("status") == "PASS") if cuda_map else 0
    print(f"  XPU: {x_pass}/{len(xpu_map)} PASS" if xpu_map else "  XPU: no results")
    print(f"  CUDA: {c_pass}/{len(cuda_map)} PASS" if cuda_map else "  CUDA: no results")
    print()


if __name__ == "__main__":
    main()
