# Intel GPU (XPU) Dockerfile of verl

Intel GPU support is provided by the external
[verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin) package.
This image installs verl and the plugin together, so the `xpu` platform is
auto-discovered with no extra configuration.

For usage, supported features and troubleshooting, see the plugin's
[Intel XPU User Guide](https://github.com/verl-project/verl-hardware-plugin/blob/main/docs/accelerators/xpu/README.md).

## Supported Hardware

- Intel Arc Pro B60 (Battlemage, 24 GB)

## Key Versions

| Component | Version |
|-----------|---------|
| Base image | `intel/deep-learning-essentials:2026.1.0-devel-ubuntu24.04` |
| Python | 3.12 |
| torch | 2.13.0+xpu |
| vLLM | 0.29.0+xpu (prebuilt XPU wheel) |
| triton-xpu | 3.7.2 (follows torch) |
| transformers | 5.17.0 |
| oneCCL | 2022.0.0 (pip wheel; the base image's 2022.1 is not used by the venv) |
| compute-runtime / Level Zero | 26.18.38308.1 / 1.28.2 |

## What the Image Contains

- verl, editable install at `/opt/verl`; dependencies from verl's own `requirements.txt` plus the XPU overlay `requirements-xpu.txt`
- verl-hardware-plugin, editable install at `/opt/verl-hardware-plugin`
- vLLM for rollout; FSDP / FSDP2 for training

## Building Locally

`requirements-xpu.txt` is read from the cloned verl, so the ref you build (`VERL_REF`) must contain it. While
it is unmerged, build with `--build-arg VERL_GIT_REPO=<your fork> --build-arg VERL_REF=<branch>`.

```sh
docker build -f docker/xpu/Dockerfile.stable.vllm -t verl:xpu-vllm029 .
```

### Useful build arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `VERL_GIT_REPO` / `VERL_REF` | `verl-project/verl` / `main` | verl source to install |
| `PLUGIN_GIT_REPO` / `PLUGIN_REF` | `verl-project/verl-hardware-plugin` / pinned commit | plugin source to install |
| `VLLM_VERSION` | `0.29.0` | vLLM XPU wheel version |
| `APT_MIRROR` | empty | apt mirror, e.g. `https://mirrors.tuna.tsinghua.edu.cn` |

## Running

```sh
docker run -it --rm \
    --device /dev/dri \
    --group-add $(getent group render | cut -d: -f3) \
    --shm-size 16g \
    -v $HOME/data:/root/data \
    verl:xpu-vllm029
```

Check that the platform resolves to Intel:

```sh
python3 -c "from verl.plugin.platform import get_platform; p = get_platform(); print(p.device_name, p.vendor_name)"
# xpu intel
```

## Installing Without Docker

The image is not required. On a host with the Intel GPU driver (compute-runtime >= 26.18 and Level Zero; a system
oneAPI install is **not** needed), from a verl checkout, with [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
export UV_INDEX_URL=https://download.pytorch.org/whl/xpu UV_EXTRA_INDEX_URL=https://pypi.org/simple \
       UV_INDEX_STRATEGY=unsafe-best-match

# 1. vLLM XPU wheel (brings torch 2.13.0+xpu); the PyPI vllm wheel has no XPU kernels
uv pip install "vllm==0.29.0+xpu" \
    --extra-index-url https://wheels.vllm.ai/0.29.0/xpu --extra-index-url https://wheels.vllm.ai/xpu \
    --extra-index-url https://pypi.org/simple

# 2. verl's requirements + the XPU overlay
grep -v '^transformers' requirements.txt > /tmp/requirements-core.txt
echo 'pyzes>=0.1.2' > /tmp/override-pyzes.txt
uv --no-config pip install -r /tmp/requirements-core.txt -r requirements-xpu.txt --override /tmp/override-pyzes.txt
uv --no-config pip install --no-deps -e .

# 3. triton: keep triton-xpu as the only owner of triton/, at the version the resolver chose
TRITON_XPU="$(uv pip freeze | grep -i '^triton-xpu==')"
uv pip uninstall triton
uv pip install --reinstall --no-deps "${TRITON_XPU}"

# 4. the plugin
uv pip install --no-deps -e /path/to/verl-hardware-plugin
```

Why these exact flags:

- `uv`, not `pip`: `pip` has no `--override`, and `pyzes>=0.1.2` must beat torch's own `pyzes==0.1.1`.
- `--no-config`: inside a verl checkout uv reads `pyproject.toml` `[tool.uv].override-dependencies`, which pin
  `transformers` and `kernels` for the CUDA stack and silently beat the XPU pins in `requirements-xpu.txt`.
- `grep -v '^transformers'`: uv cannot override a direct requirement, and `requirements.txt` caps
  `transformers<5.13` while XPU uses 5.17.0 (the flash-attn2 hub kernel build).
- `--no-deps`: otherwise the install may replace the XPU torch with a CUDA or newer build.
- The `triton` step: `liger-kernel` (in verl's `requirements.txt`) depends on `triton`, and the `triton` and
  `triton-xpu` distributions ship the same `triton/` files. Left alone, a second owner of those files can
  overwrite the XPU backend. Uninstalling `triton` and reinstalling `triton-xpu` at the resolved version leaves one
  owner. The Dockerfile does the same.

`uv pip check` then reports a few expected mismatches (verl's `transformers<5.13`, torch's `pyzes==0.1.1`,
and the `triton` / `triton-xpu` package names). They are intentional.

## Release History

- 2026/10/15: initial image, vLLM 0.29.0, torch 2.13.0+xpu
