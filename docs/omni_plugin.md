# Omni (vLLM-Omni on TPU)

This package is the TPU platform plugin for
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). vLLM-Omni loads it
through the `vllm_omni.platform_plugins` entry point `torchtpu_omni`
(`vllm_omni_torchtpu:register_omni_tpu_platform`), which returns
`vllm_omni_torchtpu.platform.OmniTpuPlatform` when TPU chips are present.
It also registers the `vllm_omni.general_plugins` entry point `torchtpu_omni`
(`vllm_omni_torchtpu:register_omni_general_plugins`), which vLLM-Omni runs in
every process. On TPU, that hook applies the patches that must be in place
before a diffusion engine starts.
Model code comes from vLLM-Omni; TPU fixes are applied as runtime patches.

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
