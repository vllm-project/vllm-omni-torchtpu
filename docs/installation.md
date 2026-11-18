# Installation

Maintainers adapted this guide from the omni installation sections of
[`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu).
In `vllm-torchtpu`, users enabled omni support with `--extra omni`. In
`vllm-omni-torchtpu`, `vllm-omni==0.32.0rc1` and the CPU `torchaudio==2.11.0`
pin are base dependencies.

| Upstream `vllm-torchtpu` option | `vllm-omni-torchtpu` equivalent |
| :--- | :--- |
| `--extra omni` (installs `vllm-omni` and CPU `torchaudio`) | `uv pip install -e .` (both packages are base dependencies of `vllm-omni-torchtpu`). |
| In-tree `vllm-torchtpu-core` checkout | `--group core-git` (installs `vllm-torchtpu-core` at `b4c66abc296bf91770346fe2d9ad88d106a4f9b6` for `#2065` and `#3304`). |
| `--group vllm` (builds `vllm` `v0.33.0` from source) | `--group vllm` with `VLLM_TARGET_DEVICE=empty` (or install the prebuilt `vllm-torchtpu==0.33.0` wheel). |

## Python 3.12 Virtual Environment

`vllm-omni-torchtpu` requires Python 3.12 (`>=3.12,<3.13`) because TPU runtime
wheels target `cp312` only. Create a virtual environment with `uv` and install
the prebuilt `vllm-torchtpu==0.33.0` wheel together with `core-git` and `test`:

```bash
uv venv --python 3.12 ~/venvs/vllm-omni-torchtpu
source ~/venvs/vllm-omni-torchtpu/bin/activate
git clone https://github.com/mbohlool/vllm-omni-torchtpu-private.git
cd vllm-omni-torchtpu-private

SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.33.0 \
  uv pip install --torch-backend cpu \
  "vllm-torchtpu==0.33.0" \
  -e . --group core-git --group test

export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

`vllm-project/vllm-torchtpu` keeps release tags off `main`, so building
`core-git` without `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.33.0`
stamps a `0.1.dev*` version below `vllm-torchtpu-core>=0.33.0`.

If you prefer to build `vllm` `v0.33.0` from source instead of using the
prebuilt `vllm-torchtpu==0.33.0` wheel, pass `VLLM_TARGET_DEVICE=empty` and
`--group vllm`:

```bash
VLLM_TARGET_DEVICE=empty \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.33.0 \
  uv pip install --torch-backend cpu \
  -e . --group core-git --group vllm --group test
```

## Required `VLLM_PLUGINS` Coexistence Setting

Every process that uses `vllm-omni-torchtpu` alongside `vllm-torchtpu-core`
(`<= 0.33.x`) **must** set `VLLM_PLUGINS`:

```bash
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
```

Leaving `VLLM_PLUGINS` unset activates both `torchtpu` (from
`vllm-torchtpu-core`) and `torchtpu_omni` (from `vllm-omni-torchtpu`), which
causes `vLLM` and `vllm-omni` to raise `RuntimeError`. See the
[top-level README](../README.md#required-vllm_plugins-setting) for details on
how both plugin loaders filter entry points, and see [testing.md](testing.md)
to verify your installation.
