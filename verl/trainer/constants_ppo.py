# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        # Intel XPU: propagate ONEAPI_DEVICE_SELECTOR to all Ray worker processes.
        # Ray workers are spawned fresh and do not inherit the shell environment;
        # without this, vLLM subprocesses get 'level_zero:' (empty after ':')
        # from Intel setvars.sh and SIGABRT immediately.
        "ONEAPI_DEVICE_SELECTOR": os.environ.get("ONEAPI_DEVICE_SELECTOR", "level_zero:0,1"),
    },
}
def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # Intel XPU: always explicitly propagate ONEAPI_DEVICE_SELECTOR to Ray workers.
    # Ray workers are spawned as fresh processes and do NOT inherit the parent shell
    # environment. The filter above would drop keys already in os.environ, but for
    # ONEAPI_DEVICE_SELECTOR that is exactly the value we need workers to receive.
    oneapi_selector = os.environ.get("ONEAPI_DEVICE_SELECTOR")
    if oneapi_selector:
        runtime_env["env_vars"]["ONEAPI_DEVICE_SELECTOR"] = oneapi_selector

    # Propagate HF_HUB_OFFLINE to prevent vLLM subprocess model inspection hangs.
    hf_offline = os.environ.get("HF_HUB_OFFLINE")
    if hf_offline:
        runtime_env["env_vars"]["HF_HUB_OFFLINE"] = hf_offline

    return runtime_env
