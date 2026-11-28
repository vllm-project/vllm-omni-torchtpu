# Omni (vLLM-Omni on TPU)

This package is the TPU platform plugin for
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). vLLM-Omni loads it
through the `vllm_omni.platform_plugins` entry point `torchtpu_omni`
(`vllm_omni_torchtpu:register_omni_tpu_platform`), which returns
`vllm_omni_torchtpu.platform.OmniTpuPlatform` when TPU chips are present.
Model code comes from vLLM-Omni; TPU fixes are applied as runtime patches.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Wan2.2 TPU Optimizations

Optional Wan2.2 performance optimizations live under
`vllm_omni_torchtpu.wan_opt` and are all **disabled by default**. When unset,
they import no extra modules, install no import hooks, and emit no logs.

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_OMNI_WAN_ATTN_FORK` | `0` | Enable the tuned Splash-attention kernel fork for `TpuFlashAttentionImpl`. |
| `VLLM_OMNI_WAN_BQ` | unset | Override Splash query block size (`block_q`) when `VLLM_OMNI_WAN_ATTN_FORK=1`. |
| `VLLM_OMNI_WAN_BKV` | unset | Override Splash key/value block size (`block_kv`) when `VLLM_OMNI_WAN_ATTN_FORK=1`. |
| `VLLM_OMNI_WAN_BKVC` | unset | Override Splash KV compute block size (`block_kv_compute`) when `VLLM_OMNI_WAN_ATTN_FORK=1`. |
| `VLLM_OMNI_WAN_SPLASH_UNIFORM` | `0` | Use the uniform fixed-m Splash kernel with compiler cost hints. |
| `VLLM_OMNI_WAN_SPARSE_ATTN` | `0` | Enable block-sparse Splash attention after warmup denoising steps. |
| `VLLM_OMNI_WAN_SPARSE_KV_BLOCKS` | `22` | Active KV blocks per query block (sink + centered static window) when `VLLM_OMNI_WAN_SPARSE_ATTN=1`. |
| `VLLM_OMNI_WAN_SPARSE_SINK_BLOCKS` | `2` | Leading sink KV blocks always retained when `VLLM_OMNI_WAN_SPARSE_ATTN=1`. |
| `VLLM_OMNI_WAN_SPARSE_WARMUP_STEPS` | `20` | Initial dense denoising steps before block-sparse attention activates. |
| `VLLM_OMNI_WAN_ULYSSES_CHUNKS` | `1` | Split Ulysses attention heads into `G` chunks (`G > 1`) in a compiled 5-D graph. |
| `VLLM_OMNI_WAN_FP8_LINEAR` | `0` | Enable per-channel FP8 W8A8 quantization for Wan2.2 DiT block linears. |
| `VLLM_OMNI_WAN_FP8_SKIP_HIGH_NOISE` | `0` | Keep the high-noise expert (`denoise_step_idx == 0` at init) in BF16. |
| `VLLM_OMNI_WAN_FP8_SKIP_EDGE_BLOCKS` | `0` | Keep the first and last DiT blocks in BF16 when FP8 linear is enabled. |
| `VLLM_OMNI_WAN_FP8_CROSS_KV` | `1` | Quantize cross-attention `attn2.to_k` and `attn2.to_v` when FP8 linear is enabled. |
| `VLLM_OMNI_WAN_FP8_EXCLUDE` | `""` | Comma-separated block-relative suffixes/prefixes (e.g. `attn2,ffn.net_2`) to keep in BF16. |
| `VLLM_OMNI_WAN_VAE_PP_FIX` | `0` | Replace `ReduceOp.MAX` with `all_gather_single` and sync stages in `DistributedVaeExecutor`. |
| `VLLM_OMNI_WAN_FAST_OUTPUT` | `0` | Enable direct-to-SHM `uint16`/borrowed IPC packing and fast `finalize_diffusion_media`. |

Recommended TPU v7x configuration (S4):

```bash
export VLLM_OMNI_WAN_ATTN_FORK=1
export VLLM_OMNI_WAN_BQ=4736
export VLLM_OMNI_WAN_BKV=2048
export VLLM_OMNI_WAN_SPLASH_UNIFORM=1
export VLLM_OMNI_WAN_SPARSE_ATTN=1
export VLLM_OMNI_WAN_ULYSSES_CHUNKS=2
export VLLM_OMNI_WAN_FP8_LINEAR=1
export VLLM_OMNI_WAN_FP8_SKIP_HIGH_NOISE=1
export VLLM_OMNI_WAN_FP8_CROSS_KV=0
export VLLM_OMNI_WAN_FP8_SKIP_EDGE_BLOCKS=1
export VLLM_OMNI_WAN_VAE_PP_FIX=1
export VLLM_OMNI_WAN_FAST_OUTPUT=1
```

## Contributing

Before you work on a new feature, bug fix, or model in `vllm-omni-torchtpu`:

1. Open an issue in this repository first to propose the change.
2. Wait for a repository maintainer to confirm the scope on the issue before
   starting work (standalone `CODEOWNERS` and `CONTRIBUTING.md` policies are
   pending post-split setup).
3. Link the issue in your pull request and run the local verification commands
   in [testing.md](testing.md).
