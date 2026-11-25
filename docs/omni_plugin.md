# Omni (vLLM-Omni on TPU)

This package is the TPU platform plugin for
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). vLLM-Omni loads it
through the `vllm_omni.platform_plugins` entry point `torchtpu_omni`
(`vllm_omni_torchtpu:register_omni_tpu_platform`), which returns
`vllm_omni_torchtpu.platform.OmniTpuPlatform` when TPU chips are present.
Model code comes from vLLM-Omni; TPU fixes are applied as runtime patches.

## Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_OMNI_COMPILE` | `"0"` | Selects the opt-in DiT `torch.compile` mode for the `torch_tpu` `"tpu"` backend (`dynamic=False`). `"0"` keeps the DiT eager. `"regional"` compiles each repeated transformer block, and `"full"` compiles the whole DiT (case-insensitive). Requires Omni `enforce_eager=False`. Pipelines that define their own `setup_compile()` stay eager. Diffusion workers scope their compiled executables through `torch.compiler.config.cache_key_tag`. |
| `VLLM_OMNI_COMPILE_SPMD_SAFE` | `False` | When `VLLM_OMNI_COMPILE=regional`, wraps each compiled block of an allow-listed DiT (FLUX, FLUX.2, Qwen-Image, or Wan) as one `torch_tpu` SPMD-safe region so sequence-parallel collectives stay inside the compiled program. The plugin ignores this setting for `full` mode and for unlisted models. |

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Contributing

Before you work on a new feature, bug fix, or model in `vllm-omni-torchtpu`:

1. Open an issue in this repository first to propose the change.
2. Wait for a repository maintainer to confirm the scope on the issue before
   starting work (standalone `CODEOWNERS` and `CONTRIBUTING.md` policies are
   pending post-split setup).
3. Link the issue in your pull request and run the local verification commands
   in [testing.md](testing.md).
