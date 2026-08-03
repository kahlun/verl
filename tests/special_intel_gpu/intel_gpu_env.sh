#!/usr/bin/env bash

# Shared runtime environment setup for Intel GPU special tests.
#
# Optional overrides can be placed in:
#   tests/special_intel_gpu/.env.intel_gpu
# or provided with:
#   XPU_ENV_FILE=/path/to/file
#
# Supported modes:
#   configure_xpu_runtime sft
#   configure_xpu_runtime vllm

configure_xpu_runtime() {
    local mode="${1:-sft}"

    if [[ -z "${NUM_GPUS:-}" ]]; then
        echo "NUM_GPUS must be set before calling configure_xpu_runtime" >&2
        return 1
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local env_file="${XPU_ENV_FILE:-${script_dir}/.env.intel_gpu}"

    if [[ -f "${env_file}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi

}
