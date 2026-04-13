# veRL on Intel XPU

## Supported Hardware

- Intel Data Center GPU Max Series (Ponte Vecchio)
- Intel Arc Pro B-Series (Battlemage)
- Intel Arc A-Series (Alchemist)

## Quick Start

### Build the Docker image

```bash
# Standard build
docker build -t verl-xpu:latest -f docker/xpu/Dockerfile.xpu .

# Behind a corporate proxy
docker build \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -t verl-xpu:latest -f docker/xpu/Dockerfile.xpu .
```

### Run with GPU access

```bash
# Find render group GID on host
RENDER_GID=$(getent group render | cut -d: -f3)

docker run -it --rm \
  --device /dev/dri --group-add ${RENDER_GID} \
  --shm-size 16g \
  -v $HOME/data:/root/data \
  verl-xpu:latest
```

### Run e2e tests inside the container

```bash
# SFT smoke test (4 GPU)
NUM_GPUS=4 bash tests/special_xpu/run_sft_xpu.sh

# GRPO e2e with vLLM rollout (2 GPU)
NUM_GPUS=2 bash tests/special_xpu/run_grpo_xpu.sh

# PPO e2e with critic model (4 GPU)
NUM_GPUS=4 bash tests/special_xpu/run_ppo_xpu.sh
```

## Dependencies

| Package | Version | Source |
|---------|---------|--------|
| PyTorch | 2.11.0+xpu | `https://download.pytorch.org/whl/xpu` |
| oneccl-bind-pt | (matches torch) | `https://download.pytorch.org/whl/xpu` |
| triton-xpu | 3.7.0 | `https://download.pytorch.org/whl/xpu` |
| vLLM | >= 0.17 | Built from source with `VLLM_TARGET_DEVICE=xpu` |
| vllm-xpu-kernels | 0.1.5 | GitHub release (pulled by vLLM XPU build) |
| transformers | latest | PyPI |

## Known Workarounds (pre-DLE 2026.0 driver)

Multi-GPU requires these environment variables due to Level Zero IPC limitations:

```bash
export CCL_ATL_SHM=1        # Route collectives via /dev/shm
export CCL_BUFFER_CACHE=0    # Prevent stale IPC handle cache
```

These are set automatically in the Dockerfile. They will be removed once Intel ships the DLE 2026.0 driver update.

## Attention Backend

XPU uses `flash_attention_2` via the HuggingFace `kernels` package
([transformers PR #41956](https://github.com/huggingface/transformers/pull/41956)).
This auto-downloads `kernels-community/flash-attn2` (Intel SYCL kernel) on first use —
no CUDA `flash-attn` package needed. **2.59x faster than SDPA** on XPU.

### Executable stack workaround

The pre-compiled `flash-attn2` `.so` has `GNU_STACK RWE` (executable stack), which
hardened Linux kernels block. The Dockerfile patches this automatically with `patchelf`.
If installing outside Docker, run after `pip install kernels`:

```bash
pip install patchelf
find ~/.cache/huggingface/ -name "*.so" -path "*flash_attn*" \
    -exec patchelf --clear-execstack {} \;
```

Upstream fix pending: `kernels-community/flash-attn2` needs `-z noexecstack` in build.
