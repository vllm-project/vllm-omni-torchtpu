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
| `VLLM_OMNI_WAN_VAE_MULTICHIP_PATCHES` | `True` | Enables patches for Wan VAE decode across multiple chips (`vae_patch_parallel_size > 1`). The plugin replaces point-to-point halo exchange in spatial-shard modes with an all-gather. It runs `DistributedVaeExecutor` integer metadata and `MAX` all-reduce collectives on the host `gloo` group and keeps Wan VAE encode untiled on each rank. Set this variable to `0` to keep the upstream code paths. |

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
