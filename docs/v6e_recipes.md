# TPU v6e Serving Recipes for Diffusion Models

`vllm-omni-torchtpu` registers `vllm_omni_torchtpu.platform.OmniTpuPlatform` for
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) through the
`vllm_omni.platform_plugins` entry point `torchtpu_omni`
(`vllm_omni_torchtpu:register_omni_tpu_platform`), so FLUX and Qwen-Image
pipelines run on TPU v6e through the offline `Omni` API and `vllm serve --omni`.
Follow [installation.md](installation.md) for a local Python 3.12 install, or
build [`docker/Dockerfile`](../docker/Dockerfile) from the repository root
(which sets
`ENV VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models`).
The numbers below were measured on native single-host `v6e-1`, `v6e-4`, and
`v6e-8` VMs with `torch-tpu 0.1.1+release.2026.9.22`, `libtpu 0.0.50`, and
`vllm-omni 0.29.0rc1`.

| Model Checkpoint | Recommended v6e Slice | Parallelism Layout |
| :--- | :--- | :--- |
| `black-forest-labs/FLUX.2-klein-4B` | `v6e-1` (1 chip) | world size 1 (or Ulysses `SP=4` on `v6e-4`) |
| `black-forest-labs/FLUX.1-schnell` | `v6e-4` (4 chips) | `TP=4` |
| `black-forest-labs/FLUX.1-dev` | `v6e-4` or `v6e-8` | `TP=4` or `TP=8` |
| `Qwen/Qwen-Image` | `v6e-8` (8 chips) | `TP=8` with `vae_use_tiling=True` (experimental: almost no HBM headroom per chip) |
| `Qwen/Qwen-Image-Edit` | not supported on `v6e` | OOM while loading weights at `TP=8` (see [Limitations on v6e](#limitations-on-v6e)) |

## Container Launch

For current `vllm-omni-torchtpu` builds, build `docker/Dockerfile` from the
repository root (see [installation.md](installation.md)). The historical
measurements below used the upstream CI `-omni` image with the Hugging Face
cache mounted from `/mnt/hf` (`FLUX.1-dev` is gated, so `HF_TOKEN` must grant
access to that checkpoint):

```bash
IMG=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu:a7ce631c23e80a41a3294aa41ef8ed914ca11e95-8039d3cbfd55be2aa172d544042c53b4e966369d-omni

sudo docker run -it --rm --privileged --net=host --shm-size=64g \
  -v /mnt/hf:/mnt/hf \
  -e HF_HOME=/mnt/hf \
  -e HF_TOKEN="${HF_TOKEN}" \
  "${IMG}" bash
```

Set these environment variables inside the container for all runs below, and
leave `TORCH_TPU_TOPOLOGY` unset so the runtime derives it from the world size:

```bash
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
export DIFFUSION_ATTENTION_BACKEND=TPU_SDPA
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=1800
unset TORCH_TPU_TOPOLOGY
```

> [!WARNING]
> **Historical workarounds in the pinned upstream `-omni` image**: The legacy
> `a7ce631c23...-omni` image above predates [`docker/Dockerfile`](../docker/Dockerfile)
> and needed three in-container workarounds (all fixed in current
> `vllm-omni-torchtpu` images):
>
> - `cv2` failed to import (missing `libGL`) in spawned multi-rank workers;
>   resolved by `apt-get install -y libgl1 libglib2.0-0`
>   ([vllm-torchtpu#1939](https://github.com/vllm-project/vllm-torchtpu/pull/1939)).
> - `RoutedExpertsLists` was missing from `vllm.v1.outputs` in that vLLM build
>   and `AsyncOmni.release_kv_cache_memory` was abstract
>   ([vllm-torchtpu#1945](https://github.com/vllm-project/vllm-torchtpu/pull/1945)).
> - `vllm serve --omni` imported `torchaudio`, resolved by installing the CPU
>   wheel `torchaudio==2.11.0` from `https://download.pytorch.org/whl/cpu`
>   ([vllm-torchtpu#1944](https://github.com/vllm-project/vllm-torchtpu/pull/1944)).

## Offline Inference (`Omni` API)

Run `FLUX.2-klein-4B` on `v6e-1`. For `FLUX.1-*`, pass `tensor_parallel_size=4`
(`v6e-4`) or `tensor_parallel_size=8` (`v6e-8`), and set `guidance_scale=0.0`
for `FLUX.1-schnell` or `guidance_scale=3.5` for `FLUX.1-dev`. For Ulysses
sequence parallelism, pass `ulysses_degree=4` instead:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == "__main__":
    omni = Omni(
        "black-forest-labs/FLUX.2-klein-4B",
        init_timeout=1200,  # timeouts used for the measured runs
        stage_init_timeout=1200,
    )
    params = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
    )
    outputs = omni.generate("A glowing crystal monolith in a misty desert at night", params)
    outputs[0].images[0].save("flux2_klein.png")
    omni.shutdown()
```

## Online Serving (`vllm serve --omni`)

Serve `FLUX.1-dev` with `TP=4` on `v6e-4`. For `FLUX.2-klein-4B` on `v6e-1`,
omit `--tensor-parallel-size`:

```bash
vllm serve black-forest-labs/FLUX.1-dev --omni \
  --tensor-parallel-size 4 \
  --init-timeout 1200 \
  --stage-init-timeout 1200 \
  --port 8000
```

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "A glowing crystal monolith in a misty desert at night",
        "size": "1024x1024",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "seed": 42,
        "response_format": "b64_json"
      }' \
  | python3 -c 'import base64, json, sys; open("flux1_dev.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

## Measured Performance on v6e

Warm latency is the second request at the same seed. Peak HBM is the maximum
across the active chips (`32 GB` per chip). The first request includes XLA
compilation.

| Model | Slice and Layout | Mode | Resolution | Steps (guidance) | Warm Latency | Warm s/step | Peak HBM per Chip | First Request |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 | Offline | 512x512 | 4 (1.0) | 0.28 s | 0.071 | 15.58 GiB | 47.3 s |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 | Offline | 1024x1024 | 4 (1.0) | 0.70 s | 0.175 | 15.86 GiB | 183.2 s |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 ¹ | `vllm serve` | 512x512 | 4 (default) | 0.30 s | 0.076 | 16.32 GiB | 37.6 s |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 ¹ | `vllm serve` | 1024x1024 | 4 (default) | 0.78 s | 0.195 | 16.26 GiB | 175.8 s |
| `FLUX.2-klein-4B` | `v6e-4`, Ulysses `SP=4` | Offline | 1024x1024 | 4 (1.0) | 0.49 s | 0.123 | 19.17 GiB | 10.61 s |
| `FLUX.1-schnell` | `v6e-4`, `TP=4` | Offline | 512x512 | 4 (0.0) | 0.49 s | 0.122 | 17.65 GiB | 1.02 s |
| `FLUX.1-dev` | `v6e-4`, `TP=4` | Offline | 512x512 | 10 (3.5) | 1.06 s | 0.106 | 18.95 GiB | 1.09 s |
| `FLUX.1-dev` | `v6e-4`, `TP=4` | Offline | 1024x1024 | 28 (3.5) | 3.60 s | 0.129 | 22.55 GiB | 131.11 s |
| `FLUX.1-dev` | `v6e-4`, `TP=4` | `vllm serve` | 1024x1024 | 28 (3.5) | 3.70 s | 0.131 | 23.90 GiB | 4.50 s ² |
| `FLUX.1-dev` | `v6e-8`, `TP=8` | Offline | 1024x1024 | 28 (3.5) | 3.08 s | 0.110 | 21.02 GiB | 115.8 s |
| `Qwen-Image` | `v6e-8`, `TP=8` ³ | Offline | 512x512 | 20 (true CFG 4.0) | 8.60 s | 0.43 | 30.86 GiB | 40.8 s |
| `Qwen-Image` | `v6e-8`, `TP=8` ³ | Offline | 1024x1024 | 20 (true CFG 4.0) | 10.61 s | 0.53 | 31.11 GiB | 43.0 s |

¹ Measured on a native `v6e-1` VM with the
[vllm-torchtpu#1944](https://github.com/vllm-project/vllm-torchtpu/pull/1944)
and
[vllm-torchtpu#1945](https://github.com/vllm-project/vllm-torchtpu/pull/1945)
fixes applied in the container, without passing `guidance_scale` (pipeline
default). Latency is end-to-end HTTP (`POST /v1/images/generations`), including
PNG encoding and base64 serialization in the response. Warm latency is the
faster of two warm requests at the same seed, and server startup to ready took
`17.8 s`.

² Ran after the offline `1024x1024` `TP=4` run in the same container, so this
value is not a cold-compile measurement.

³ Measured with `vae_use_tiling=True` and `true_cfg_scale=4.0` (Qwen-Image uses
true classifier-free guidance, so each step runs two transformer passes). Peak
HBM is within `0.15 GiB` of the `31.24 GiB` usable per chip; engine
initialization took `129.6 s` (cold) and `65.6 s` (warm Hugging Face cache).
Running without VAE tiling, at larger resolutions, or with batch size above 1
was not tested and is expected to run out of HBM.

`FLUX.1-schnell` with `TP=4` on `v6e-4` also matches the CPU `diffusers` golden
image within the [`tests/models/flux/test_flux_pr.py`](../tests/models/flux/test_flux_pr.py)
thresholds (`MAE 0.0182`, `SSIM 0.9649`).

## Qwen-Image and Qwen-Image-Edit

`Qwen/Qwen-Image` runs on `v6e-8` with `TP=8`, but leaves almost no HBM
headroom: every chip replicates the Qwen2.5-VL text encoder and the
transformer's `img_mod` and `txt_mod` modulation projections are not
tensor-parallel sharded, so peak HBM reaches `31.11 GiB` of the `31.24 GiB`
usable per chip at `1024x1024`. Treat this layout as experimental and send a
warm-up request before serving traffic. Only the offline configuration below was
measured (`vllm serve` was not benchmarked for Qwen-Image on `v6e`):

```python
omni = Omni(
    "Qwen/Qwen-Image",
    tensor_parallel_size=8,
    vae_use_tiling=True,
    init_timeout=1200,
    stage_init_timeout=1200,
)
params = OmniDiffusionSamplingParams(
    height=1024, width=1024, num_inference_steps=20, true_cfg_scale=4.0, seed=42
)
```

`Qwen/Qwen-Image-Edit` does **not** fit on `v6e-8` at `TP=8`: weight loading
fails with `RuntimeBufferAllocationFailure`. The HSDP path (`use_hsdp=True`)
fails inside PyTorch FSDP2 initialization (`variable.set_data(tensor)` with
incompatible tensor types in
`torch.distributed.fsdp._fully_shard._fsdp_init._move_states_to_device`), and
CPU offload is not supported on TPU, so `v6e` has no supported configuration for
`Qwen-Image-Edit` today.

## Limitations on v6e

- **`FLUX.1-*` does not fit on `v6e-1`**: `FLUX.1-schnell` runs out of HBM while
  loading weights on a single chip (`30.66 GiB` peak, `RESOURCE_EXHAUSTED`), and
  `FLUX.1-dev` uses the same transformer size. Use `TP=4` on `v6e-4` or `TP=8`
  on `v6e-8`.
- **2-chip world sizes are not supported**: world size 2 (for example
  `--tensor-parallel-size 2`) fails with
  `RuntimeError: No TPU topology found for count: 2`, and forcing
  `TORCH_TPU_TOPOLOGY=1,2,1` fails in `libtpu` with `START_SESSION failed`. Use
  a world size of 1, 4, or 8
  ([vllm-torchtpu#1943](https://github.com/vllm-project/vllm-torchtpu/pull/1943)).
- **Qwen-Image-Edit does not fit on `v6e-8`**: `TP=8` runs out of HBM while
  loading weights (`RuntimeBufferAllocationFailure`). `Qwen-Image` fits only
  with `vae_use_tiling=True` and batch size 1.
- **HSDP is not available on TPU yet**: `use_hsdp=True` fails in PyTorch FSDP2
  initialization (`variable.set_data(tensor)` incompatible tensor type), so it
  cannot shard the replicated Qwen-Image layers.
- **First-request compilation cost**: the first request compiles the pipeline
  and can take minutes (up to `183.2 s` measured above). Send a warm-up request
  before serving traffic, and set `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=1800` with
  `1200 s` init timeouts.
