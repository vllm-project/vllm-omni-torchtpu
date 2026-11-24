# Benchmarks

Runnable evaluation workflows for models served by `vllm-omni-torchtpu`. Each
subdirectory owns one benchmark suite, with its scripts, metadata and README.

- [`vbench/`](vbench/README.md): [VBench](https://github.com/Vchitect/VBench)
  quality scores for any text-to-video model. Videos are generated on TPU and
  scored on GPU:

  ```bash
  python benchmarks/vbench/generate_videos.py --model <model> --run-name <run> --gcs-bucket <bucket> --fps <fps>
  bash benchmarks/vbench/run_gpu_eval.sh --ssh GPU_NAME=<gpu-vm> GCS_BUCKET=<bucket> RUN_NAME=<run>
  ```
