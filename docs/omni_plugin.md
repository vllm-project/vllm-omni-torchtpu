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
| `VLLM_OMNI_LTX2_PER_SAMPLE_TIMESTEP` | `False` | Opt-in. LTX-2 requests without a conditioning mask pass one timestep per sample (`(B,)`) to the transformer instead of one per token, so each block builds its AdaLN tables once per sample. Off by default because the bf16 TPU output differs from the per-token path and no on-device bf16 comparison yet shows that the difference comes from rounding alone. |

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
