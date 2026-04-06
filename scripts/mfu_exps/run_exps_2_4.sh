#!/usr/bin/env bash
# Sequential orchestrator for MFU experiments 2-4
# Run this after Exp1 (TT TP=1 FSDP4) finishes
set -euo pipefail

echo "=== Starting Exp2: TorchTitan 4-GPU Llama8B TP=2 FSDP2=2 at $(date) ==="
bash /workspace/verl/scripts/mfu_exps/run_exp2.sh
echo "=== Exp2 done at $(date) ==="
sleep 5

echo "=== Starting Exp3: AutoModel 4-GPU Llama8B TP=1 at $(date) ==="
bash /workspace/verl/scripts/mfu_exps/run_exp3.sh
echo "=== Exp3 done at $(date) ==="
sleep 5

echo "=== Starting Exp4: AutoModel 4-GPU Llama8B TP=4 at $(date) ==="
bash /workspace/verl/scripts/mfu_exps/run_exp4.sh
echo "=== Exp4 done at $(date) ==="

echo ""
echo "All experiments complete. MFU summary:"
echo "Exp1 (TT TP=1 FSDP4):"
grep "step:" /root/verl/logs/mfu-tt-llama8b-tp1-fsdp4.log 2>/dev/null | grep "mfu:" | tail -5
echo "Exp2 (TT TP=2 FSDP2):"
grep "step:" /root/verl/logs/mfu-tt-llama8b-tp2-fsdp2.log 2>/dev/null | grep "mfu:" | tail -5
echo "Exp3 (AM TP=1):"
grep "mfu\|step" /root/verl/logs/mfu-am-llama8b-tp1.log 2>/dev/null | tail -5
echo "Exp4 (AM TP=4):"
grep "mfu\|step" /root/verl/logs/mfu-am-llama8b-tp4.log 2>/dev/null | tail -5
