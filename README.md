# vllm-omni-torchtpu

`vllm-omni-torchtpu` (`vllm_omni_torchtpu`) is the Google Cloud TPU platform
plugin for [vLLM-Omni](https://github.com/vllm-project/vllm-omni). Maintainers
extracted this repository from
[`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu)
and preserved the commit history with `Upstream-commit` trailers.

This package provides `OmniTpuPlatform`, TPU diffusion attention backends, RoPE
helpers, multimodal autoregressive workers, and runtime model patches. Core TPU
kernels and `TpuPlatform` remain in `vllm-torchtpu-core`. See the
[plugin overview](src/vllm_omni_torchtpu/README.md) and the
[roadmap](src/vllm_omni_torchtpu/ROADMAP.md) for scope details.

## Installation (Python 3.12)

Create a Python 3.12 virtual environment with `uv` (`>= 0.6`) and install the
prebuilt `vllm-torchtpu==0.33.0` runtime before overlaying `core-git` and `test`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --torch-backend cpu "vllm-torchtpu==0.33.0"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0 \
  uv pip install --torch-backend cpu \
  -e . --group core-git --group test
```

The `core-git` group builds `vllm-torchtpu-core` from the upstream commit that
matches this extract (`#2065` and `#3304`, newer than `0.33.0`).
`vllm-torchtpu==0.33.0` pins `vllm-torchtpu-core==0.33.0`, so install it in a
separate step before `-e .` (`vllm-torchtpu-core>=0.34.0.dev0`). Setting
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0` satisfies
that floor when `uv` builds `main` without release tags. If `uv` already cached
an earlier git build of `vllm-torchtpu-core`, run
`uv cache clean vllm-torchtpu-core` first. To build `vllm` from source in one
step instead, pass `VLLM_TARGET_DEVICE=empty` and `--group vllm` without
`vllm-torchtpu==0.33.0`.

## Required `VLLM_PLUGINS` Setting

Set `VLLM_PLUGINS` whenever a `vllm-omni` process runs alongside
`vllm-torchtpu-core` builds that still carry the legacy `torchtpu` omni entry
point:

```bash
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

`vllm-torchtpu-core` (`<= 0.33.x` and current `main`) registers `torchtpu` in
both `vllm.platform_plugins` and `vllm_omni.platform_plugins`.
`vllm-omni-torchtpu` registers `torchtpu_omni` in both groups so one allowlist
selects `OmniTpuPlatform` and `vllm-torchtpu-core`'s `TpuPlatform`. When
`VLLM_PLUGINS` is unset, `register_tpu_platform_alias` returns `None` so plain
`vLLM` processes continue working without extra configuration. On TPU,
`vllm-omni` processes without `VLLM_PLUGINS` still discover two omni platform
plugins and raise `RuntimeError` until core removes its legacy omni entry point.

## Running CPU Checks

`tests_omni/conftest.py` sets the `VLLM_PLUGINS` default automatically before
`pytest` imports `vllm`. On a CPU host without TPU devices, run the packaging
entry-point test and verify full test collection:

```bash
pytest tests_omni/test_packaging.py
pytest tests_omni --collect-only -q
```

Model and worker tests in `tests_omni/` synchronize the TPU accelerator during
fixture setup and require a TPU host to execute.
