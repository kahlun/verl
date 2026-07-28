#!/usr/bin/env bash
# Launch the colocated-reward-model GRPO test (run_grpo_colocate_rm_intel_gpu.sh)
# inside the validated verl-intel-gpu:vllm-0.22.1 XPU image.
#
# Mirrors tests/special_intel_gpu/docker_run_ppo.sh, plus the 2 known image-gap
# pip fixes (triton-xpu collision, TransferQueue missing) applied inline since
# the prebuilt image predates them being baked into the Dockerfile.
#
# Usage:
#   NUM_GPUS=2 bash tests/special_intel_gpu/docker_run_colocate_rm.sh

set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-verl-intel-gpu:vllm-0.22.1}
NUM_GPUS=${NUM_GPUS:-2}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
DATA_DIR=${DATA_DIR:-$HOME/data}
RENDER_GID=${RENDER_GID:-$(getent group render | cut -d: -f3)}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

docker run --rm --name verl_colocate_rm_test \
    --device /dev/dri --group-add "${RENDER_GID}" \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --shm-size 16g --network host --tmpfs /tmp:exec,size=8g \
    -v "${REPO_DIR}:/workspace/verl" \
    -v "${DATA_DIR}:/root/data" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 \
    -e ZE_AFFINITY_MASK=0,1 \
    -e RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 \
    -e NUM_GPUS="${NUM_GPUS}" \
    "${IMAGE_TAG}" \
    bash -c '
        uv pip uninstall triton triton-xpu >/dev/null 2>&1 || true
        uv pip install triton-xpu==3.7.1 --extra-index-url https://download.pytorch.org/whl/xpu >/dev/null
        uv pip install --no-deps TransferQueue==0.1.8 >/dev/null
        cd /workspace/verl
        NUM_GPUS='"${NUM_GPUS}"' bash tests/special_intel_gpu/run_grpo_colocate_rm_intel_gpu.sh \
            +actor_rollout_ref.rollout.enable_sleep_mode=False \
            +reward.reward_model.rollout.enable_sleep_mode=False
    '
