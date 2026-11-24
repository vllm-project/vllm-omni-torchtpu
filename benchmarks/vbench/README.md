# VBench

This directory runs [VBench](https://github.com/Vchitect/VBench), the
video-generation benchmark from the
[VBench paper](https://arxiv.org/pdf/2311.17982), on any text-to-video model
that `vllm-omni` serves on TPU. Use it to check that an optimization keeps
generation quality, or to compare models.

The workflow has two stages, connected by a GCS prefix:

1. `generate_videos.py` generates one video per prompt and seed on a TPU host
   and uploads them to `gs://<bucket>/<run-name>/videos/`.
2. `run_gpu_eval.sh` scores those videos with VBench on a GPU host, prints the
   score table and uploads the results to
   `gs://<bucket>/<run-name>/vbench_results/`.

VBench's evaluator models run several times faster on a GPU than on CPUs,
which is why scoring runs on a separate GPU host.

## Files

- `generate_videos.py`: TPU video generation for any `vllm-omni` video model.
- `run_gpu_eval.sh`: VBench setup, evaluation and result upload on a GPU host.
- `gpu_eval_utils.py`: helpers for `run_gpu_eval.sh` (VBench compatibility
  patches, video layout, score table).
- `VBench_full_info_sub110.json`: the default prompts, a 110-prompt subset of
  VBench that still covers all 16 dimensions with 10 prompts each. This is
  enough to catch a quality regression.

The complete VBench prompt set (946 prompts) is not kept in this repository.
`--prompts-json full` downloads VBench's `VBench_full_info.json` from the VBench
repository, pinned to a commit and checksum, and caches it under
`~/.cache/vllm_omni_torchtpu/vbench/` (or `$XDG_CACHE_HOME`).

## Generate Videos on TPU

Install `vllm-omni-torchtpu` as described in
[installation.md](../../docs/installation.md), then run from the repository
root on the TPU host. The model is the only model-specific argument that is
always needed:

```bash
python benchmarks/vbench/generate_videos.py \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --run-name wan21-baseline \
  --gcs-bucket <bucket> \
  --fps 16
```

Generation settings that are not passed (`--height`, `--width`, `--num-frames`,
`--num-inference-steps`, `--guidance-scale`, `--negative-prompt`) are left to
the pipeline's own defaults. `--fps` is required unless `vllm-omni` declares
video defaults for the model, because the MP4 has to be encoded at the rate
the model generated for.

Everything else that is specific to a model or a TPU layout goes through two
repeatable `KEY=VALUE` flags, so a new model needs no change to the script:

- `--sampling-param`: fields of `OmniDiffusionSamplingParams` are set directly;
  any other key lands in `extra_args`, where pipelines read their own knobs.
- `--engine-arg`: keyword arguments of the `Omni` engine, such as the parallel
  layout.

The script itself is the same on every TPU generation, but the layout is not.
The number of workers, which is the product of the parallel sizes, must be one
that `torch_tpu` has a topology for: 1, 4 or 8 on v6e and 2, 4 or 8 on v7x
(as of `torch-tpu` 0.2.0). Any other count stops at start-up with
`No TPU topology found for count`. Prefer a layout that uses every chip of the
VM: on a v6e-8, 8 workers start in under two minutes, while 4 fail to start a
TPU session.

For example, LTX-2 on 8 TPU devices (2-way CFG parallel, 4-way Ulysses), over
the full prompt set with five seeds per prompt:

```bash
python benchmarks/vbench/generate_videos.py \
  --model Lightricks/LTX-2 \
  --run-name ltx2-baseline \
  --gcs-bucket <bucket> \
  --prompts-json full \
  --seeds 12345,12346,12347,12348,12349 \
  --height 512 --width 768 --num-frames 121 --fps 24 \
  --num-inference-steps 40 \
  --sampling-param max_sequence_length=1024 \
  --engine-arg cfg_parallel_size=2 \
  --engine-arg ulysses_degree=4
```

Other options:

- `--seeds`: one video per prompt for each seed (default `12345`). VBench
  scores at most five videos per prompt.
- `--prompt-indices 0,10,20-22`: only these prompts, for a quick smoke test.
- `--shard-id` / `--num-shards`: split the prompts over several TPU hosts.
  Start the same command on each host with a different `--shard-id`.
- `--output-dir`: local output directory (default
  `vbench_outputs/<run-name>`). Without `--gcs-bucket`, videos stay there.

Videos are named `video_<seed>_<prompt index>.mp4`. A rerun skips videos that
already exist locally or in GCS, so an interrupted run resumes where it
stopped. `generation_summary.json` records the settings and the generation
latency (first request, and median over the warm ones).

Uploads use Application Default Credentials. TPU VMs are often created with
read-only storage access, in which case the VM's service account cannot write
to the bucket and the run stops right away with a permission error. Then
either give the VM read-write storage access, run
`gcloud auth application-default login` on the VM to upload as yourself, or
leave out `--gcs-bucket` and copy `<output-dir>/videos` to
`gs://<bucket>/<run-name>/videos/` from another machine.

## Score Videos on GPU

From a machine with this repository and `gcloud`, run the evaluation on a GPU
VM over SSH:

```bash
bash benchmarks/vbench/run_gpu_eval.sh --ssh \
  GPU_NAME=<gpu-vm> GPU_ZONE=<zone> GPU_PROJECT=<project> \
  GCS_BUCKET=<bucket> RUN_NAME=wan21-baseline
```

Or directly on the GPU host:

```bash
bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=<bucket> RUN_NAME=wan21-baseline
```

The first run builds a virtual environment and a patched VBench checkout under
`WORK_DIR` (default `$HOME/vbench_evaluation`); later runs reuse them. With
several GPUs the evaluation runs one process per GPU.

Pass the same prompt file that generation used: for `--prompts-json full`
that is `BENCHMARK_JSON=VBench_full_info.json`. Common options:

- `DIMENSIONS`: space-separated VBench dimensions (default: all in the JSON).
- `BASELINE_RUN_NAME`: an earlier run evaluated in the same `WORK_DIR`. The
  score table then adds the baseline, the difference, and the per-prompt
  paired difference with its 95% confidence interval.
- `ALLOW_MISSING=true`: score an incomplete set of videos, for smoke tests.
  By default a missing video is an error, because scores from different sets
  of prompts are not comparable.
- `VIDEO_DIR`: a local directory of videos instead of the GCS download.
- `UPLOAD_RESULTS=false`: keep the results on the GPU host only.
- `TUNNEL_THROUGH_IAP=true` / `INTERNAL_IP=true`: how `--ssh` reaches the VM.

In SSH mode, results are copied back and uploaded with the caller's
credentials, so the GPU VM's service account only needs to read the bucket.

## Reading the Scores

`run_gpu_eval.sh` prints a Markdown table (also saved as `summary.md`) with
each dimension's raw score and its normalized score. When all 16 dimensions
were evaluated, it adds the Quality, Semantic and Total scores, computed like
the VBench leaderboard: scores are normalized with the ranges in VBench's
`scripts/constant.py`, `dynamic_degree` weighs half, and Total is
`(4 * Quality + Semantic) / 5`.

Scores are only comparable between runs that share the prompt file, the seeds
and the generation settings. To check an optimization, generate both runs with
identical arguments and evaluate the second with `BASELINE_RUN_NAME` set to
the first.

Also score both runs on the same number of GPUs. With more than one process,
VBench computes the `appearance_style` score from the last frame of each video
instead of from all frames, so the same videos score slightly differently on
one GPU and on several. The paired difference in the table uses per-video
scores and is not affected.
