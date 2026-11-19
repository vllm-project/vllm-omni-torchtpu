# Installation

Maintainers adapted this guide from the omni installation sections of
[`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu).
In `vllm-torchtpu`, users enabled omni support with `--extra omni`. In
`vllm-omni-torchtpu`, `vllm-omni==0.32.0rc1` and the CPU `torchaudio==2.11.0`
pin are base dependencies.

| Upstream `vllm-torchtpu` option | `vllm-omni-torchtpu` equivalent |
| :--- | :--- |
| `--extra omni` (installs `vllm-omni` and CPU `torchaudio`) | `uv pip install -e .` (both packages are base dependencies of `vllm-omni-torchtpu`). |
| In-tree `vllm-torchtpu-core` checkout | `--group core-git` (installs `vllm-torchtpu-core` at `a0275aad4f81a8befa40f8dc7a3fbeebaaff4355` for `#2065` and `#3304`). |
| `--group vllm` (builds `vllm` `v0.33.0` from source) | `--group vllm` with `VLLM_TARGET_DEVICE=empty` (or install the prebuilt `vllm-torchtpu==0.33.0` wheel). |

## Python 3.12 Virtual Environment

`vllm-omni-torchtpu` requires Python 3.12 (`>=3.12,<3.13`) because TPU runtime
wheels target `cp312` only. Create a virtual environment with `uv` (`>= 0.6`),
install the prebuilt `vllm-torchtpu==0.33.0` wheel, and then install this
repository with `core-git` and `test`:

```bash
uv venv --python 3.12 ~/venvs/vllm-omni-torchtpu
source ~/venvs/vllm-omni-torchtpu/bin/activate
git clone https://github.com/vllm-project/vllm-omni-torchtpu.git
cd vllm-omni-torchtpu

uv pip install --torch-backend cpu "vllm-torchtpu==0.33.0"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0 \
  uv pip install --torch-backend cpu \
  -e . --group core-git --group test

export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

`vllm-torchtpu==0.33.0` pins `vllm-torchtpu-core==0.33.0`, so install it in a
separate step before `-e .` (`vllm-torchtpu-core>=0.34.0.dev0`).
`vllm-project/vllm-torchtpu` keeps release tags off `main`, so setting
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0` stamps a
version that satisfies `vllm-torchtpu-core>=0.34.0.dev0`. If `uv` already cached
an earlier git build of `vllm-torchtpu-core`, run
`uv cache clean vllm-torchtpu-core` first.

If you prefer to build `vllm` `v0.33.0` from source in one step instead of using
the prebuilt `vllm-torchtpu==0.33.0` wheel, pass `VLLM_TARGET_DEVICE=empty` and
`--group vllm`:

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
