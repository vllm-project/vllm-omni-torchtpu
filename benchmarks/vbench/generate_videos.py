#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate VBench videos with any vLLM-Omni text-to-video model on TPU.

Nothing in this script is specific to one model. The checkpoint is ``--model``,
generation settings left unset fall back to the pipeline's own defaults, and
model-specific knobs go through ``--sampling-param`` and ``--engine-arg``.

Videos are written as ``<output-dir>/videos/video_<seed>_<prompt index>.mp4``
and, with ``--gcs-bucket``, uploaded to ``gs://<bucket>/<run-name>/videos/``,
which is where ``run_gpu_eval.sh`` reads them from. Reruns skip videos that
already exist, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

VBENCH_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_JSON = VBENCH_DIR / "VBench_full_info_sub110.json"
# The complete VBench prompt set is VBench's file, so it is not kept in this
# repository. `--prompts-json full` downloads it from the VBench repository into
# a cache directory outside the repository. The commit and checksum are pinned:
# video names carry the prompt's index in this file, so the file must not
# change between generation and evaluation.
FULL_PROMPTS_NAME = "VBench_full_info.json"
FULL_PROMPTS_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/"
    "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/VBench_full_info.json"
)
FULL_PROMPTS_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
DEFAULT_VLLM_PLUGINS = "torchtpu_omni,torchtpu_layers,vllm_omni_register_models"
# VBench scores at most five videos per prompt.
MAX_SEEDS = 5
# Generation settings a model may declare defaults for, and their CLI flags.
GENERATION_SETTINGS = (
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "guidance_scale",
)
MAX_PENDING_ENCODES = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="Hugging Face model ID or local checkpoint path."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of this run; also the GCS prefix that run_gpu_eval.sh reads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output directory (default: ./vbench_outputs/<run-name>).",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=None,
        help="Upload videos and metadata to gs://<bucket>/<run-name>/.",
    )
    parser.add_argument(
        "--prompts-json",
        default=str(DEFAULT_PROMPTS_JSON),
        help="VBench metadata JSON holding the prompts, or 'full' for the complete "
        "946-prompt VBench set, downloaded from the VBench repository on first use "
        "(default: the bundled 110-prompt subset).",
    )
    parser.add_argument(
        "--seeds",
        default="12345",
        help=f"Comma-separated seeds, one video per prompt each (max {MAX_SEEDS}).",
    )
    parser.add_argument(
        "--prompt-indices",
        default=None,
        help="Only these prompt indices, e.g. '0,10,20-22' (default: all).",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0, help="Index of this shard (default: 0)."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the prompts over this many hosts; each runs one --shard-id.",
    )

    generation = parser.add_argument_group(
        "generation settings (default: the model's own)"
    )
    generation.add_argument("--height", type=int, default=None)
    generation.add_argument("--width", type=int, default=None)
    generation.add_argument("--num-frames", type=int, default=None)
    generation.add_argument("--num-inference-steps", type=int, default=None)
    generation.add_argument("--guidance-scale", type=float, default=None)
    generation.add_argument("--negative-prompt", default=None)
    generation.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate to generate and encode at. Required unless vllm-omni "
        "declares video defaults for the model.",
    )
    generation.add_argument(
        "--sampling-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra sampling parameter (JSON value). OmniDiffusionSamplingParams "
        "fields are set directly; any other key goes into extra_args. Repeatable.",
    )
    generation.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Omni() engine argument (JSON value), e.g. ulysses_degree=4, "
        "cfg_parallel_size=2 or tensor_parallel_size=2. Repeatable.",
    )
    return parser.parse_args(argv)


def parse_key_values(items: list[str]) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` items; values are JSON, falling back to plain strings."""
    parsed: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def parse_seeds(spec: str) -> list[int]:
    seeds = list(dict.fromkeys(int(tok) for tok in spec.split(",") if tok.strip()))
    if not 1 <= len(seeds) <= MAX_SEEDS:
        raise ValueError(
            f"Expected 1 to {MAX_SEEDS} distinct seeds, got {len(seeds)}: VBench "
            f"scores at most {MAX_SEEDS} videos per prompt"
        )
    return seeds


def parse_indices(spec: str, total: int) -> list[int]:
    """Parse comma-separated indices and inclusive ``start-end`` ranges."""
    indices: list[int] = []
    for tok in spec.split(","):
        if not tok.strip():
            continue
        start, _, end = tok.partition("-")
        indices.extend(range(int(start), int(end or start) + 1))
    indices = list(dict.fromkeys(indices))
    out_of_range = [i for i in indices if not 0 <= i < total]
    if out_of_range:
        raise ValueError(f"Prompt indices {out_of_range} are outside [0, {total})")
    return indices


def fetch_file(url: str, sha256: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, refusing content that does not match ``sha256``."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{url} has sha256 {digest}, expected {sha256}")
    tmp_path = dest.with_name(f".{dest.name}.tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, dest)


def full_prompts_path() -> Path:
    """Where the downloaded full prompt set is kept: under ``$XDG_CACHE_HOME``."""
    cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache_home) / "vllm_omni_torchtpu" / "vbench" / FULL_PROMPTS_NAME


def resolve_prompts_json(value: str) -> Path:
    """Return the prompt file for ``--prompts-json``, downloading the full set if asked.

    A cached copy is used only if it still matches the pinned checksum.
    """
    if value != "full":
        return Path(value)
    path = full_prompts_path()
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() == FULL_PROMPTS_SHA256:
            return path
        print(f"[vbench] {path} does not match the pinned checksum", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vbench] Downloading {FULL_PROMPTS_URL} to {path}", flush=True)
    fetch_file(FULL_PROMPTS_URL, FULL_PROMPTS_SHA256, path)
    return path


def load_prompts(path: Path) -> list[str]:
    """Read the ``prompt_en`` of every entry in a VBench metadata JSON, in order."""
    return [entry["prompt_en"] for entry in json.loads(path.read_text("utf-8"))]


def select_tasks(
    num_prompts: int,
    seeds: list[int],
    prompt_indices: str | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[tuple[int, int]]:
    """Return this shard's ``(prompt index, seed)`` pairs, one seed at a time.

    Going through all prompts for one seed before starting the next means an
    interrupted run still covers every prompt with as many seeds as it finished.
    """
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"--shard-id {shard_id} is outside [0, {num_shards})")
    indices = (
        parse_indices(prompt_indices, num_prompts)
        if prompt_indices
        else list(range(num_prompts))
    )
    indices = [idx for idx in indices if idx % num_shards == shard_id]
    return [(idx, seed) for seed in seeds for idx in indices]


def video_filename(seed: int, idx: int) -> str:
    return f"video_{seed}_{idx}.mp4"


def to_uint8_video(
    frames: Any, tensor_range: str = "negative_one_to_one"
) -> np.ndarray:
    """Convert a pipeline's video output to a uint8 ``[F, H, W, 3]`` RGB array.

    Pipelines return a batched or unbatched array or tensor, or a list of
    frames (arrays, tensors or PIL images). Integer data is taken as 0-255.
    Float arrays are taken as [0, 1], which is what diffusers' video processor
    returns. Float tensors follow ``tensor_range``, the output range vllm-omni
    declares for the model (``vllm_omni.model_extras.get_output_tensor_range``).
    """
    import torch

    is_tensor = isinstance(frames, torch.Tensor) or (
        isinstance(frames, (list, tuple))
        and len(frames) > 0
        and isinstance(frames[0], torch.Tensor)
    )
    if isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            raise ValueError("The pipeline returned no video frames")
        if is_tensor:
            frames = torch.stack(list(frames)) if len(frames) > 1 else frames[0]
        elif len(frames) == 1:
            frames = frames[0]
        else:
            frames = np.stack([np.asarray(frame) for frame in frames])
    if is_tensor:
        frames = frames.detach().cpu()
        if frames.is_floating_point():
            frames = frames.float()
            if tensor_range == "negative_one_to_one":
                frames = frames * 0.5 + 0.5
        frames = frames.numpy()
    video = np.asarray(frames)

    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected one video per request, got {video.shape[0]}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected a 4D or 5D video, got shape {video.shape}")
    if video.shape[-1] not in (3, 4):
        if video.shape[0] in (3, 4):  # [C, F, H, W]
            video = video.transpose(1, 2, 3, 0)
        elif video.shape[1] in (3, 4):  # [F, C, H, W]
            video = video.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Cannot find the channel axis in shape {video.shape}")
    video = video[..., :3]

    if not np.issubdtype(video.dtype, np.integer):
        if not np.isfinite(video).all():
            raise ValueError("The generated video contains NaN or Inf values")
        video = np.round(np.clip(video, 0.0, 1.0) * 255.0)
    return np.ascontiguousarray(video, dtype=np.uint8)


def extract_video_and_audio(
    outputs: list[Any], tensor_range: str = "negative_one_to_one"
) -> tuple[np.ndarray, np.ndarray | None, int | None]:
    """Return ``(uint8 video, float32 audio or None, audio sample rate or None)``.

    Audio is kept when the model generates it (for example LTX-2). VBench does
    not score it, but the saved MP4 then matches what the model produced.
    """
    if len(outputs) != 1:
        raise ValueError(f"Expected one output per request, got {len(outputs)}")
    output = outputs[0]
    if getattr(output, "error", None):
        raise RuntimeError(f"Generation failed: {output.error}")
    frames: Any = output.images
    multimodal = dict(getattr(output, "multimodal_output", None) or {})
    audio = multimodal.get("audio")
    sample_rate = multimodal.get("audio_sample_rate")

    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio", audio)
        sample_rate = frames.get("audio_sample_rate", sample_rate)
        frames = frames.get("frames", frames.get("video"))
    if frames is None:
        raise ValueError("The pipeline returned no video frames")
    video = to_uint8_video(frames, tensor_range)

    if audio is None or sample_rate is None:
        return video, None, None
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio = np.squeeze(np.asarray(audio, dtype=np.float32))
    return video, audio, int(sample_rate)


def encode_mp4(
    video: np.ndarray,
    audio: np.ndarray | None,
    sample_rate: int | None,
    fps: float,
    path: Path,
) -> None:
    """Encode to H.264 (plus AAC when there is audio) and move into place."""
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    data = mux_video_audio_bytes(
        video, audio, fps=fps, audio_sample_rate=sample_rate or 44100
    )
    # Write under a name the resume check ignores, so a killed run never
    # leaves a truncated video that a rerun would skip.
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


class GcsRunFolder:
    """``gs://<bucket>/<run-name>/``, which is where run_gpu_eval.sh reads a run.

    Uses the google-cloud-storage client with Application Default Credentials:
    the VM's service account, or the user's own credentials after
    ``gcloud auth application-default login``.
    """

    def __init__(self, bucket: str, run_name: str) -> None:
        from google.cloud import storage

        bucket = bucket.removeprefix("gs://").strip("/")
        self.url = f"gs://{bucket}/{run_name}"
        self._run_name = run_name
        self._bucket = storage.Client().bucket(bucket)

    def video_names(self) -> set[str]:
        prefix = f"{self._run_name}/videos/"
        return {
            blob.name.removeprefix(prefix)
            for blob in self._bucket.list_blobs(prefix=prefix)
        }

    def upload(self, path: Path, name: str | None = None) -> None:
        """Upload ``path`` as ``<run-name>/<name>`` (default: its file name)."""
        blob = self._bucket.blob(f"{self._run_name}/{name or path.name}")
        blob.upload_from_filename(str(path))


def resolve_generation_settings(
    args: argparse.Namespace, model_defaults: dict[str, Any]
) -> dict[str, Any]:
    """Merge CLI flags over the defaults vllm-omni declares for the model.

    Settings that neither names are left out, so the pipeline's ``forward()``
    picks its own default. The frame rate is the exception: the MP4 has to be
    encoded at the rate the model generated for, so it must be known here.
    """
    settings: dict[str, Any] = {}
    for name in (*GENERATION_SETTINGS, "fps"):
        value = getattr(args, name)
        if value is None:
            value = model_defaults.get(name)
        if value is not None:
            settings[name] = value
    if "fps" not in settings:
        raise ValueError(
            "vllm-omni declares no default frame rate for this model; pass --fps "
            "(for example 16 for Wan, 24 for LTX-2)"
        )
    return settings


def build_sampling_kwargs(
    settings: dict[str, Any], overrides: dict[str, Any], field_names: set[str]
) -> dict[str, Any]:
    """Build ``OmniDiffusionSamplingParams`` keyword arguments.

    ``overrides`` come from ``--sampling-param``: names that are sampling
    parameter fields are set directly and the rest go into ``extra_args``,
    which is how pipelines receive their model-specific knobs.
    """
    kwargs: dict[str, Any] = {
        # A CPU generator makes a seed produce the same noise on any accelerator.
        "generator_device": "cpu",
        **settings,
        # Pipelines disagree on which of the two they read (Wan: fps, LTX-2:
        # frame_rate), so set both.
        "frame_rate": settings["fps"],
    }
    extra_args: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in field_names:
            kwargs[key] = value
        else:
            extra_args[key] = value
    kwargs["extra_args"] = {**kwargs.get("extra_args", {}), **extra_args}
    return kwargs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Select this package's omni and vLLM plugins over vllm-torchtpu-core's
    # `torchtpu`, as tests/conftest.py does. Must be set before vllm is imported.
    os.environ.setdefault("VLLM_PLUGINS", DEFAULT_VLLM_PLUGINS)

    seeds = parse_seeds(args.seeds)
    prompts_json = resolve_prompts_json(args.prompts_json)
    prompts = load_prompts(prompts_json)
    tasks = select_tasks(
        len(prompts), seeds, args.prompt_indices, args.shard_id, args.num_shards
    )
    sampling_overrides = parse_key_values(args.sampling_param)
    engine_args = parse_key_values(args.engine_arg)

    output_dir = args.output_dir or Path("vbench_outputs") / args.run_name
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    gcs = GcsRunFolder(args.gcs_bucket, args.run_name) if args.gcs_bucket else None
    uploaded: set[str] = set()
    if gcs:
        # run_gpu_eval.sh picks the prompts up from here when it runs somewhere
        # without this repository. Uploading them first also fails within
        # seconds, not after the model has loaded, if the bucket is not writable.
        gcs.upload(prompts_json)
        uploaded = gcs.video_names()

    pending: list[tuple[int, int]] = []
    for idx, seed in tasks:
        name = video_filename(seed, idx)
        if name in uploaded:
            continue
        if not (video_dir / name).is_file():
            pending.append((idx, seed))
        elif gcs:  # Generated by an earlier run that died before uploading.
            gcs.upload(video_dir / name, f"videos/{name}")
    print(
        f"[vbench] {len(tasks)} videos in this shard "
        f"({len(tasks) // len(seeds)} prompts x {len(seeds)} seeds), "
        f"{len(tasks) - len(pending)} already done, {len(pending)} to generate",
        flush=True,
    )

    if pending:
        stats = generate(
            args,
            prompts,
            pending,
            video_dir,
            gcs,
            sampling_overrides,
            engine_args,
        )
        summary = {
            "model": args.model,
            "run_name": args.run_name,
            "prompts_json": prompts_json.name,
            "seeds": seeds,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "videos_in_shard": len(tasks),
            "engine_args": engine_args,
            **stats,
        }
        # Only written by runs that generated something, so that rerunning a
        # finished run does not replace its latency numbers with an empty file.
        shard = (
            f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
        )
        summary_path = output_dir / f"generation_summary{shard}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
        if gcs:
            gcs.upload(summary_path)
    print(f"[vbench] Done. Videos: {gcs.url if gcs else output_dir}/videos", flush=True)
    return 0


def generate(
    args: argparse.Namespace,
    prompts: list[str],
    pending: list[tuple[int, int]],
    video_dir: Path,
    gcs: GcsRunFolder | None,
    sampling_overrides: dict[str, Any],
    engine_args: dict[str, Any],
) -> dict[str, Any]:
    """Start the engine, generate ``pending`` and return run statistics."""
    from vllm_omni.diffusion.data import resolve_model_class_name
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import (
        get_model_class_name,
        get_output_tensor_range,
        get_video_generation_defaults,
    )

    # Resolve settings before the engine starts: a missing --fps should fail in
    # seconds, not after loading and compiling the model.
    model_class_name = engine_args.get("model_class_name") or resolve_model_class_name(
        args.model
    )
    declared = get_video_generation_defaults(model_class_name)
    model_defaults = declared.cli_defaults() if declared is not None else {}
    settings = resolve_generation_settings(args, model_defaults)
    fps = settings["fps"]
    negative_prompt = args.negative_prompt
    if negative_prompt is None and declared is not None:
        negative_prompt = declared.default_negative_prompt
    sampling_kwargs = build_sampling_kwargs(
        settings,
        sampling_overrides,
        {field.name for field in dataclasses.fields(OmniDiffusionSamplingParams)},
    )
    print(
        f"[vbench] model={args.model} class={model_class_name} "
        f"sampling={json.dumps(sampling_kwargs)} engine={json.dumps(engine_args)}",
        flush=True,
    )

    start = time.perf_counter()
    omni = Omni(
        model=args.model,
        **{"init_timeout": 3600, "stage_init_timeout": 3600, **engine_args},
    )
    load_s = time.perf_counter() - start
    print(f"[vbench] Engine ready in {load_s:.0f}s", flush=True)
    tensor_range = get_output_tensor_range(
        get_model_class_name(omni) or model_class_name
    )

    def save(video: np.ndarray, audio: Any, sample_rate: int | None, name: str) -> None:
        encode_mp4(video, audio, sample_rate, fps, video_dir / name)
        if gcs:
            gcs.upload(video_dir / name, f"videos/{name}")

    gen_times: list[float] = []
    video_shape: tuple[int, ...] = ()
    # Encoding and uploading run beside the next generation. The queue is
    # bounded so raw frames of unsaved videos cannot pile up in memory.
    saves: collections.deque[concurrent.futures.Future[None]] = collections.deque()
    try:
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
            open(
                video_dir.parent / "generation_log.jsonl", "a", encoding="utf-8"
            ) as log,
        ):
            for done, (idx, seed) in enumerate(pending, start=1):
                while saves and (saves[0].done() or len(saves) >= MAX_PENDING_ENCODES):
                    saves.popleft().result()
                request: dict[str, Any] = {
                    "prompt": prompts[idx],
                    "modalities": ["video"],
                }
                if negative_prompt is not None:
                    request["negative_prompt"] = negative_prompt
                start = time.perf_counter()
                outputs = list(
                    omni.generate(
                        request,
                        # A fresh copy per request: pipelines may write to it.
                        OmniDiffusionSamplingParams(
                            **copy.deepcopy(sampling_kwargs), seed=seed
                        ),
                    )
                )
                video, audio, sample_rate = extract_video_and_audio(
                    outputs, tensor_range
                )
                gen_s = time.perf_counter() - start
                gen_times.append(gen_s)
                video_shape = video.shape
                name = video_filename(seed, idx)
                saves.append(pool.submit(save, video, audio, sample_rate, name))
                log.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "seed": seed,
                            "gen_s": round(gen_s, 3),
                            "file": name,
                        }
                    )
                    + "\n"
                )
                log.flush()
                # The first request includes compilation, so leave it out of the ETA.
                warm = gen_times[1:] or gen_times
                eta_min = statistics.median(warm) * (len(pending) - done) / 60
                print(
                    f"[vbench] {done}/{len(pending)} idx={idx} seed={seed} "
                    f"shape={video.shape} gen={gen_s:.1f}s eta={eta_min:.0f}min",
                    flush=True,
                )
            while saves:
                saves.popleft().result()
    finally:
        omni.close()

    warm = gen_times[1:]
    return {
        "model_class_name": model_class_name,
        "sampling_params": sampling_kwargs,
        "negative_prompt": negative_prompt,
        "video_shape": list(video_shape),
        "videos_generated": len(gen_times),
        "engine_load_s": round(load_s, 1),
        "first_gen_s": round(gen_times[0], 3),
        "warm_gen_s_median": round(statistics.median(warm), 3) if warm else None,
        "warm_gen_s_mean": round(statistics.mean(warm), 3) if warm else None,
    }


def stop_workers_and_exit(status: int) -> None:
    """End the process after a failure, taking the engine's workers with it.

    When engine start-up or a request fails, vllm-omni leaves worker processes
    (which hold TPU chips) and non-daemon threads behind. A normal exit then
    never returns, so the failed run would hang after printing its error.
    """
    import psutil

    children = psutil.Process().children(recursive=True)
    for child in children:
        with contextlib.suppress(psutil.Error):
            child.terminate()
    _, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        with contextlib.suppress(psutil.Error):
            child.kill()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        stop_workers_and_exit(1)
