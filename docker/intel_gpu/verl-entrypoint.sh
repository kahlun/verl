#!/usr/bin/env bash
set -e

# The DLE 2026.1.0 base bakes CCL_ROOT, LD_LIBRARY_PATH, and CMAKE_PREFIX_PATH
# for oneCCL 2022.1 directly into the image ENV — no sourcing needed at runtime.

if [ -f "${VERL_PRIMARY_ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${VERL_PRIMARY_ENV_FILE}"
    set +a
elif [ -f "${VERL_FALLBACK_ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${VERL_FALLBACK_ENV_FILE}"
    set +a
fi

exec "$@"
