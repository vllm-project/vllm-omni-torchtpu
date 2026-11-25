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
| `VLLM_OMNI_RING_KERNEL` | `True` | Enables the Pallas ICI-DMA diffusion ring attention kernel on TPU v7. Set this variable to `0` to fall back to the `ppermute` residual ring attention path. |
| `VLLM_OMNI_RING_FIXED_M` | `True` | Enables hybrid per-head fixed-m softmax in the Pallas ICI-DMA diffusion ring attention kernel. Set this variable to `0` to run online softmax on every head. |

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
