# vllm-omni-torchtpu

**This repository is in migration from [vllm-project/vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu). During the migration, please continue to use [vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu).**

`vllm-omni-torchtpu` (`vllm_omni_torchtpu`) is the Google Cloud TPU platform
plugin for [vLLM-Omni](https://github.com/vllm-project/vllm-omni). Maintainers
extracted this repository from
[`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu)
and preserved the commit history with `Upstream-commit` trailers.

This package provides `OmniTpuPlatform`, TPU diffusion attention backends, RoPE
helpers, multimodal autoregressive workers, and runtime model patches. Core TPU
kernels and `TpuPlatform` remain in `vllm-torchtpu-core`. See the
[plugin overview](docs/omni_plugin.md), [installation guide](docs/installation.md),
[testing guide](docs/testing.md), and [roadmap](docs/ROADMAP.md) for details.

## Quickstart (Python 3.12)

Clone the repository, create a Python 3.12 virtual environment with `uv`
(`>= 0.6`), install the prebuilt `vllm-torchtpu==0.33.0` runtime and this
repository, and set the required `VLLM_PLUGINS` allowlist:

```bash
git clone https://github.com/vllm-project/vllm-omni-torchtpu.git
cd vllm-omni-torchtpu
uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend cpu "vllm-torchtpu==0.33.0"
uv pip install --torch-backend cpu -e .
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

The Qwen3-Omni autoregressive worker also needs `vllm-torchtpu-core` changes
newer than `0.33.0`. See the [installation guide](docs/installation.md) for the
`core-git` group, the `test` group, and building `vllm` from source.

## Required `VLLM_PLUGINS` Setting

Set `VLLM_PLUGINS` whenever a `vllm-omni` process runs alongside
`vllm-torchtpu-core` builds that still carry the legacy `torchtpu` omni entry
point:

```bash
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

`vllm-torchtpu-core` (`<= 0.33.x` and current `main`) registers `torchtpu` in
both `vllm.platform_plugins` and `vllm_omni.platform_plugins`.
`vllm-omni-torchtpu` registers `torchtpu_omni` in those groups and in
`vllm_omni.general_plugins` so one allowlist selects `OmniTpuPlatform`, the
pre-engine Omni patches, and `vllm-torchtpu-core`'s `TpuPlatform`. When
`VLLM_PLUGINS` is unset, `register_tpu_platform_alias` returns `None` so plain
`vLLM` processes continue working without extra configuration. On TPU,
`vllm-omni` processes without `VLLM_PLUGINS` still discover two omni platform
plugins and raise `RuntimeError` until core removes its legacy omni entry point.

## Running CPU Checks

`tests/conftest.py` sets the `VLLM_PLUGINS` default automatically before
`pytest` imports `vllm`. On a CPU host without TPU devices, install the `test`
group, run the packaging entry-point test, and verify full test collection:

```bash
uv pip install --torch-backend cpu -e . --group test
pytest tests/test_packaging.py
pytest tests --collect-only -q
```

Model and worker tests in `tests/` synchronize the TPU accelerator during
fixture setup and require a TPU host to execute.
