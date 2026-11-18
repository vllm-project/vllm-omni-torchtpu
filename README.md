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

Create a Python 3.12 virtual environment with `uv` and install the prebuilt
`vllm-torchtpu` runtime together with the `core-git` and `test` dependency
groups:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.33.0 \
  uv pip install --torch-backend cpu \
  "vllm-torchtpu==0.33.0" \
  -e . --group core-git --group test
```

The `core-git` group installs `vllm-torchtpu-core` from the upstream commit that
matches this extract (`#2065` and `#3304`, newer than `0.33.0`). Setting
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.33.0` stamps a version
that satisfies `vllm-torchtpu-core>=0.33.0` when `uv` builds from `main` without
release tags. To build `vllm` from source instead of installing
`vllm-torchtpu==0.33.0`, replace `"vllm-torchtpu==0.33.0"` with
`VLLM_TARGET_DEVICE=empty` and `--group vllm`.

## Required `VLLM_PLUGINS` Setting

Setting `VLLM_PLUGINS` is **REQUIRED** whenever `vllm-omni-torchtpu` and
`vllm-torchtpu-core` (`<= 0.33.x`) are installed in the same environment:

```bash
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

`vllm-torchtpu-core` (`<= 0.33.x`) registers `torchtpu` in both
`vllm.platform_plugins` and `vllm_omni.platform_plugins`. `vllm-omni-torchtpu`
registers `torchtpu_omni` in both groups so one allowlist selects this
package's omni plugin and `vllm-torchtpu-core`'s `TpuPlatform`. If you leave
`VLLM_PLUGINS` unset, `vLLM` and `vllm-omni` each discover two active platform
plugins and raise `RuntimeError`. To run plain `vLLM` without `vllm-omni` in
the same environment, set `VLLM_PLUGINS=torchtpu,torchtpu_layers`.

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
