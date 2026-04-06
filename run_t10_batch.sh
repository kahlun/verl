#!/usr/bin/env bash
# T10 batch test runner for VERL XPU gap coverage
# Runs inside docker container pt, GPU 3 only (ZE_AFFINITY_MASK=3)
# Each test: 1 epoch, skip validation, 15 min timeout
set -euo pipefail

WORKDIR=/host/home/sdp/kl/verl_test_xpu
LOGDIR=/tmp/t10_logs
mkdir -p "$LOGDIR"

export ZE_AFFINITY_MASK=3
export CCL_BUFFER_CACHE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled

cd "$WORKDIR"

# Base args shared by all tests
BASE_ARGS=(
    data.train_files=/host/home/sdp/data/gsm8k/train.parquet
    data.val_files=/host/home/sdp/data/gsm8k/test.parquet
    data.train_batch_size=8 data.max_prompt_length=256 data.max_response_length=128
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    trainer.total_epochs=1 trainer.save_freq=-1 trainer.test_freq=-1
    trainer.val_before_train=false
    trainer.n_gpus_per_node=1 trainer.device=xpu
    trainer.use_legacy_worker_impl=enable
    "trainer.logger=[console]"
)

run_test() {
    local test_id="$1"
    local desc="$2"
    shift 2
    local extra_args=("$@")

    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] T10.$test_id START: $desc"
    echo "================================================================"

    local logfile="$LOGDIR/t10_${test_id}.log"

    # Stop ray between tests
    ray stop --force 2>/dev/null || true
    sleep 2

    # Run with 900s (15 min) timeout - enough for init + 1 training step
    if timeout 900 python3 run_xpu_ppo.py "${BASE_ARGS[@]}" "${extra_args[@]}" \
        > "$logfile" 2>&1; then
        # Check if training step happened
        if grep -q "step:" "$logfile"; then
            echo "[$(date '+%H:%M:%S')] T10.$test_id PASS (training step completed)"
            echo "PASS" > "$LOGDIR/t10_${test_id}.result"
        else
            echo "[$(date '+%H:%M:%S')] T10.$test_id PARTIAL (no step output)"
            echo "PARTIAL" > "$LOGDIR/t10_${test_id}.result"
        fi
    else
        local rc=$?
        if [ $rc -eq 124 ]; then
            # Timeout - check if at least 1 step completed
            if grep -q "step:" "$logfile"; then
                echo "[$(date '+%H:%M:%S')] T10.$test_id PASS (timeout after training step)"
                echo "PASS" > "$LOGDIR/t10_${test_id}.result"
            else
                echo "[$(date '+%H:%M:%S')] T10.$test_id TIMEOUT (no step completed)"
                echo "TIMEOUT" > "$LOGDIR/t10_${test_id}.result"
            fi
        else
            echo "[$(date '+%H:%M:%S')] T10.$test_id FAIL (rc=$rc)"
            echo "FAIL" > "$LOGDIR/t10_${test_id}.result"
        fi
    fi

    # Print key output lines
    grep -E "step:|Error|Traceback|advantage|Train step|update_weight|ValueError" "$logfile" 2>/dev/null | tail -5
    echo ""
}

echo "===== VERL XPU T10 GAP-COVERAGE TESTS ====="
echo "Date: $(date)"
echo "GPU: ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
echo ""

# T10.1: OPO advantage estimator
run_test 1 "OPO advantage estimator" \
    algorithm.adv_estimator=opo

# T10.3: GRPO_PASSK advantage estimator
run_test 3 "GRPO_PASSK advantage estimator" \
    algorithm.adv_estimator=grpo_passk

# T10.4: RLOO_VECTORIZED advantage estimator
run_test 4 "RLOO_VECTORIZED advantage estimator" \
    algorithm.adv_estimator=rloo_vectorized

# T10.5: GRPO_VECTORIZED advantage estimator
run_test 5 "GRPO_VECTORIZED advantage estimator" \
    algorithm.adv_estimator=grpo_vectorized

# T10.6: File logger
run_test 6 "File logger backend" \
    algorithm.adv_estimator=grpo \
    "trainer.logger=[console,file]"

# T10.7: Tensorboard logger
run_test 7 "Tensorboard logger backend" \
    algorithm.adv_estimator=grpo \
    "trainer.logger=[console,tensorboard]"

# T10.8: DAPO reward manager
run_test 8 "DAPO reward manager" \
    algorithm.adv_estimator=grpo \
    reward.reward_manager.name=dapo

echo "===== SUMMARY ====="
for f in "$LOGDIR"/t10_*.result; do
    test_id=$(basename "$f" | sed 's/t10_//;s/.result//')
    result=$(cat "$f")
    echo "T10.$test_id: $result"
done
echo "==================="
