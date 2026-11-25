# Installation

Maintainers adapted this guide from the omni installation sections of
[`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu).
In `vllm-torchtpu`, users enabled omni support with `--extra omni`. In
`vllm-omni-torchtpu`, `vllm-omni==0.33.0rc1` and the CPU `torchaudio==2.11.0`
pin are base dependencies, so a plain `uv pip install -e .` covers it.

## Python 3.12 Virtual Environment

`vllm-omni-torchtpu` requires Python 3.12 (`>=3.12,<3.13`) because TPU runtime
wheels target `cp312` only. Clone the repository, create a virtual environment
in the repository root with `uv` (`>= 0.6`), install the prebuilt
`vllm-torchtpu==0.33.0` wheel, and then install this repository with the `test`
group:

```bash
git clone https://github.com/vllm-project/vllm-omni-torchtpu.git
cd vllm-omni-torchtpu
uv venv --python 3.12
source .venv/bin/activate

uv pip install --torch-backend cpu "vllm-torchtpu==0.33.0"
uv pip install --torch-backend cpu -e . --group test

export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

The base dependency floor is the released `vllm-torchtpu-core>=0.33.0`, so this
resolves from PyPI without any version override.

## Newer `vllm-torchtpu-core` from Git (`core-git`)

This extract was cut against `vllm-torchtpu-core` changes newer than `0.33.0`
(`#2065` and `#3304`). `TPUARWorker`, the autoregressive worker for Qwen3-Omni
stages, needs `TPUWorker.model_runner_cls` from `#2065` and raises an
`ImportError` that points to `--group core-git` when core lacks it. Until
`vllm-torchtpu-core>=0.34.0` is on PyPI, CI and TPU runs overlay `core-git`:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0 \
  uv pip install --torch-backend cpu \
  -e . --group core-git --group test
```

`vllm-project/vllm-torchtpu` keeps release tags off `main`, so `uv` builds the
pinned commit with a setuptools-scm version below `0.33.0` that fails the
floor. `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0` is
needed only with `--group core-git`. If `uv` already cached an earlier git build
of `vllm-torchtpu-core`, run `uv cache clean vllm-torchtpu-core` first.

If you prefer to build `vllm` from source in one step instead of using the
prebuilt `vllm-torchtpu==0.33.0` wheel, pass `VLLM_TARGET_DEVICE=empty` and
`--group vllm`, which builds the `vllm` tag pinned in `pyproject.toml`:

```bash
VLLM_TARGET_DEVICE=empty \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0 \
  uv pip install --torch-backend cpu \
  -e . --group core-git --group vllm --group test
```

## Required `VLLM_PLUGINS` Coexistence Setting

Every `vllm-omni` process that runs alongside `vllm-torchtpu-core` builds with
the legacy `torchtpu` omni entry point **must** set `VLLM_PLUGINS`:

```bash
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

When `VLLM_PLUGINS` is unset, `register_tpu_platform_alias` returns `None` so
plain `vLLM` processes continue working through `vllm-torchtpu-core`'s
`torchtpu` plugin. On TPU, leaving `VLLM_PLUGINS` unset in a `vllm-omni` process
activates both omni platform plugins and raises `RuntimeError`. See the
[top-level README](../README.md#required-vllm_plugins-setting) for details on
how both plugin loaders filter entry points, and see [testing.md](testing.md)
to verify your installation.
