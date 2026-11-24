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
"""Helpers for run_gpu_eval.sh. Standard library only: runs before the venv exists."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
from pathlib import Path

# generate_videos.py writes video_<seed>_<prompt index>.mp4. Only the two
# trailing numbers matter, so MaxDiffusion's <name>_<seed>_<index>.mp4 works too.
VIDEO_NAME = re.compile(r"_(\d+)_(\d+)\.mp4$")
MAX_VIDEOS_PER_PROMPT = 5

# Score normalization and weighting, copied from VBench's scripts/constant.py.
NORMALIZE_RANGE = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.364),
    "overall_consistency": (0.0, 0.364),
}
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
SEMANTIC_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)
DIMENSION_WEIGHT = {"dynamic_degree": 0.5}  # every other dimension weighs 1
QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1


def safe_prompt(prompt: str) -> str:
    """File-name-safe form of a prompt. Must match the patch in patch_vbench()."""
    return re.sub(r"[^\w\s-]", "_", prompt).strip()[:120]


def extract_dimensions(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.json_file).read_text("utf-8"))
    dimensions = dict.fromkeys(
        dimension for item in data for dimension in item.get("dimension", [])
    )
    print(" ".join(dimensions))


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(
                f"Patch target {old!r} not found in {path}. Upstream may have changed."
            )
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def patch_vbench(args: argparse.Namespace) -> None:
    """Patch a VBench checkout so it installs and runs on current GPU images."""
    root = Path(args.vbench_dir)
    # VBench pins an old torch; any recent CUDA build works.
    patch_file(
        root / "setup.py",
        [("def check_torch_version():", "def check_torch_version():\n    return")],
    )
    patch_file(
        root / "vbench/distributed.py",
        [
            (
                "backend = 'gloo' if os.name == 'nt' else 'nccl'",
                "backend = 'gloo' if (os.name == 'nt' or not "
                "torch.cuda.is_available()) else 'nccl'",
            ),
            (
                "torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))",
                "if torch.cuda.is_available():\n        "
                "torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))",
            ),
        ],
    )
    # One process per GPU under torchrun instead of every rank on cuda:0. And
    # each process named the results after the time it started, so a process
    # starting a second later looked for a file that did not exist: take the
    # time from the launcher (run_gpu_eval.sh sets VBENCH_RUN_TIMESTAMP).
    patch_file(
        root / "evaluate.py",
        [
            (
                "    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')\n",
                "    current_time = os.environ.get('VBENCH_RUN_TIMESTAMP') or "
                "datetime.now().strftime('%Y-%m-%d-%H:%M:%S')\n",
            ),
            (
                'device = torch.device("cuda")',
                "assert torch.cuda.is_available(), "
                '"CUDA is not available, but is required for VBench evaluation."\n'
                "    device = torch.device("
                "f\"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}\")",
            ),
        ],
    )
    # VBench looks videos up as "<prompt>-<i>.mp4". Prompts contain characters
    # that are awkward in file names and two prompts appear twice, so look for
    # "<safe prompt>_<prompt index>-<i>.mp4" first (see prepare_videos).
    patch_file(
        root / "vbench/__init__.py",
        [
            ("import os\n", "import os\nimport re\n"),
            (
                (
                    "            for prompt_dict in full_info_list:\n"
                    "                # if the prompt belongs to any dimension we want"
                    " to evaluate\n"
                    "                if set(dimension_list) & "
                    'set(prompt_dict["dimension"]): \n'
                    "                    prompt = prompt_dict['prompt_en']\n"
                    "                    prompt_dict['video_list'] = []\n"
                    "                    for i in range(5): # video index for the same"
                    " prompt\n"
                    "                        intended_video_name = "
                    "f'{prompt}{special_str}-{str(i)}{postfix}'"
                ),
                (
                    "            for idx, prompt_dict in enumerate(full_info_list):\n"
                    "                # if the prompt belongs to any dimension we want"
                    " to evaluate\n"
                    "                if set(dimension_list) & "
                    'set(prompt_dict["dimension"]): \n'
                    "                    prompt = prompt_dict['prompt_en']\n"
                    "                    safe_prompt = re.sub(r'[^\\w\\s-]', '_', "
                    "prompt).strip()[:120]\n"
                    "                    prompt_dict['video_list'] = []\n"
                    "                    for i in range(5): # video index for the same"
                    " prompt\n"
                    "                        intended_video_name = "
                    "f'{safe_prompt}_{idx}-{str(i)}{postfix}'\n"
                    "                        if intended_video_name not in "
                    "video_names:\n"
                    "                            intended_video_name = "
                    "f'{prompt}{special_str}-{str(i)}{postfix}'"
                ),
            ),
        ],
    )
    # With several processes, every one of them writes this metadata file while
    # the others may already be reading it. Write it once and make all wait.
    patch_file(
        root / "vbench/__init__.py",
        [
            (
                "        save_json(cur_full_info_list, cur_full_info_path)\n"
                "        print0(f'Evaluation meta data saved to "
                "{cur_full_info_path}')\n",
                "        if get_rank() == 0:\n"
                "            save_json(cur_full_info_list, cur_full_info_path)\n"
                "            print0(f'Evaluation meta data saved to "
                "{cur_full_info_path}')\n"
                "        import torch.distributed as dist\n"
                "        if dist.is_available() and dist.is_initialized():\n"
                "            dist.barrier()\n",
            ),
        ],
    )
    # `color` only counts videos in which the object was detected. A process
    # whose videos all lack a detection divided by zero. The per-process value
    # is not used for the final score, which is computed over all processes.
    patch_file(
        root / "vbench/color.py",
        [
            (
                "    success_rate = success_frame_count_all / video_count\n",
                "    success_rate = (success_frame_count_all / video_count) "
                "if video_count > 0 else 0.0\n",
            ),
            (
                "        all_results = success_frame_count / frame_count\n",
                "        all_results = (success_frame_count / frame_count) "
                "if frame_count > 0 else 0.0\n",
            ),
        ],
    )
    # pkg_resources is gone from current setuptools; packaging is what ViCLIP
    # (temporal_style, overall_consistency) wanted from it.
    patch_file(
        root / "vbench/third_party/ViCLIP/viclip_text.py",
        [("from pkg_resources import packaging\n", "import packaging.version\n")],
    )


def prepare_videos(args: argparse.Namespace) -> None:
    """Link generated videos under the names the patched VBench looks for.

    ``<name>_<seed>_<prompt index>.mp4`` becomes
    ``<safe prompt>_<position>-<i>.mp4``, where ``i`` is the position of the
    seed among all seeds found, in ascending order.

    Also writes ``eval_json``, the metadata to evaluate with: the entries of
    ``json_file`` that have at least one video. VBench hands every entry to
    its worker processes, and a process that gets entries without videos
    crashes, so entries without videos must not reach it. ``position`` is the
    index within ``eval_json``, which is the original prompt index when no
    video is missing.
    """
    bench_data = json.loads(Path(args.json_file).read_text("utf-8"))

    videos: dict[int, dict[int, Path]] = {}
    for path in sorted(Path(args.download_dir).glob("*.mp4")):
        match = VIDEO_NAME.search(path.name)
        if match:
            videos.setdefault(int(match.group(2)), {})[int(match.group(1))] = path
    seeds = sorted({seed for by_seed in videos.values() for seed in by_seed})
    print(f"Found videos for {len(videos)} prompts and seeds {seeds}.")
    if not seeds:
        raise ValueError(f"No <name>_<seed>_<index>.mp4 videos in {args.download_dir}")
    if len(seeds) > MAX_VIDEOS_PER_PROMPT:
        raise ValueError(
            f"Found {len(seeds)} seeds, but VBench scores at most "
            f"{MAX_VIDEOS_PER_PROMPT} videos per prompt."
        )

    missing = [
        f"prompt {idx} seed {seed}"
        for idx in range(len(bench_data))
        for seed in seeds
        if seed not in videos.get(idx, {})
    ]
    if missing and not args.allow_missing:
        raise ValueError(
            f"{len(missing)} videos are missing (first: {', '.join(missing[:5])}). "
            "Scores from an incomplete set are not comparable; finish generation, "
            "or pass ALLOW_MISSING=true to score what is there."
        )

    shutil.rmtree(args.vbench_dir, ignore_errors=True)
    os.makedirs(args.vbench_dir)
    eval_data = []
    count = 0
    for idx, item in enumerate(bench_data):
        if idx not in videos:
            continue
        for i, seed in enumerate(seeds):
            source = videos[idx].get(seed)
            if source is None:
                continue
            target = Path(args.vbench_dir) / (
                f"{safe_prompt(item['prompt_en'])}_{len(eval_data)}-{i}.mp4"
            )
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, target)
            count += 1
        eval_data.append(item)
    Path(args.eval_json).write_text(json.dumps(eval_data, indent=2), "utf-8")
    print(f"Prepared {count} VBench videos ({len(missing)} missing).")


def max_processes(args: argparse.Namespace) -> None:
    """Print how many VBench processes the given dimensions can be split over.

    VBench splits the prompts of a dimension over its processes, and a process
    without any prompt crashes. So the limit is the number of prompts of the
    smallest dimension.
    """
    data = json.loads(Path(args.json_file).read_text("utf-8"))
    counts = [
        sum(dimension in item.get("dimension", []) for item in data)
        for dimension in args.dimensions
    ]
    empty = [d for d, count in zip(args.dimensions, counts) if count == 0]
    if empty:
        raise ValueError(f"No videos for dimensions: {', '.join(empty)}")
    print(min(counts))


def load_results(results_dir: str) -> dict[str, tuple[float, dict[str, float]]]:
    """Read the newest ``*_eval_results.json`` of a VBench results directory.

    Returns, per dimension, the raw score and the score of each video keyed by
    file name.
    """

    def unit_scale(dimension: str, score: float) -> float:
        # VBench reports imaging_quality per video on a 0-100 scale (and, in
        # older versions, the dimension score too). Every other score is 0-1.
        return score / 100 if dimension == "imaging_quality" and score > 1 else score

    files = sorted(Path(results_dir).glob("*_eval_results.json"))
    if not files:
        raise FileNotFoundError(f"No *_eval_results.json in {results_dir}")
    results: dict[str, tuple[float, dict[str, float]]] = {}
    for dimension, (score, entries) in json.loads(files[-1].read_text("utf-8")).items():
        per_video = {
            Path(entry["video_path"]).name: unit_scale(
                dimension, float(entry["video_results"])
            )
            for entry in entries
            if isinstance(entry.get("video_results"), (bool, int, float))
        }
        results[dimension] = (unit_scale(dimension, float(score)), per_video)
    return results


def normalize(dimension: str, score: float) -> float:
    low, high = NORMALIZE_RANGE[dimension]
    return (score - low) / (high - low)


def group_score(scores: dict[str, float], dimensions: tuple[str, ...]) -> float | None:
    """Weighted mean of the normalized scores of a group, as VBench computes it."""
    if any(dimension not in scores for dimension in dimensions):
        return None
    weights = [DIMENSION_WEIGHT.get(dimension, 1.0) for dimension in dimensions]
    total = sum(
        normalize(dimension, scores[dimension]) * weight
        for dimension, weight in zip(dimensions, weights)
    )
    return total / sum(weights)


def total_scores(scores: dict[str, float]) -> dict[str, float | None]:
    quality = group_score(scores, QUALITY_DIMENSIONS)
    semantic = group_score(scores, SEMANTIC_DIMENSIONS)
    total = None
    if quality is not None and semantic is not None:
        total = (quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT) / (
            QUALITY_WEIGHT + SEMANTIC_WEIGHT
        )
    return {"Quality": quality, "Semantic": semantic, "Total": total}


def paired_delta(
    videos: dict[str, float], baseline: dict[str, float]
) -> tuple[float, float] | None:
    """Mean per-prompt difference to the baseline and its 95% interval half-width.

    Videos of one prompt (one per seed) are averaged first, so the interval
    reflects variation across prompts rather than across seeds.
    """

    def by_prompt(per_video: dict[str, float]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for name, score in per_video.items():
            grouped.setdefault(name.rsplit("-", 1)[0], []).append(score)
        return {prompt: statistics.mean(values) for prompt, values in grouped.items()}

    ours, theirs = by_prompt(videos), by_prompt(baseline)
    deltas = [ours[prompt] - theirs[prompt] for prompt in sorted(ours.keys() & theirs)]
    if len(deltas) < 2:
        return None
    return statistics.mean(deltas), 1.96 * statistics.stdev(deltas) / math.sqrt(
        len(deltas)
    )


def summarize(args: argparse.Namespace) -> None:
    """Print a Markdown score table, optionally against a baseline run."""
    results = load_results(args.results_dir)
    baseline = load_results(args.baseline) if args.baseline else None
    scores = {dimension: score for dimension, (score, _) in results.items()}

    header = ["Dimension", "Group", "Videos", "Score", "Normalized"]
    if baseline:
        header += ["Baseline", "Delta", "Paired delta (95% CI)"]
    rows = []
    for dimension in (*QUALITY_DIMENSIONS, *SEMANTIC_DIMENSIONS):
        if dimension not in results:
            continue
        score, per_video = results[dimension]
        row = [
            dimension,
            "Quality" if dimension in QUALITY_DIMENSIONS else "Semantic",
            str(len(per_video)),
            f"{score:.4f}",
            f"{normalize(dimension, score):.2%}",
        ]
        if baseline:
            if dimension in baseline:
                base_score, base_videos = baseline[dimension]
                paired = paired_delta(per_video, base_videos)
                row += [
                    f"{base_score:.4f}",
                    f"{score - base_score:+.4f}",
                    f"{paired[0]:+.4f} ± {paired[1]:.4f}" if paired else "n/a",
                ]
            else:
                row += ["n/a", "n/a", "n/a"]
        rows.append(row)

    base_totals = (
        total_scores({dimension: score for dimension, (score, _) in baseline.items()})
        if baseline
        else {}
    )
    for name, value in total_scores(scores).items():
        if value is None:
            continue
        row = [f"**{name}**", "", "", "", f"**{value:.2%}**"]
        if baseline:
            base_value = base_totals.get(name)
            row += (
                [f"{base_value:.2%}", f"{(value - base_value) * 100:+.2f} pp", ""]
                if base_value is not None
                else ["n/a", "n/a", ""]
            )
        rows.append(row)

    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" :--- "] + [" ---: "] * (len(header) - 1)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    missing = [d for d in NORMALIZE_RANGE if d not in results]
    if missing:
        print(f"\nNot evaluated, so no total score: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    dimensions = subparsers.add_parser("dimensions")
    dimensions.add_argument("json_file")
    dimensions.set_defaults(func=extract_dimensions)

    patch = subparsers.add_parser("patch-vbench")
    patch.add_argument("vbench_dir")
    patch.set_defaults(func=patch_vbench)

    prepare = subparsers.add_parser("prepare-videos")
    prepare.add_argument("json_file")
    prepare.add_argument("download_dir")
    prepare.add_argument("vbench_dir")
    prepare.add_argument("eval_json")
    prepare.add_argument("--allow-missing", action="store_true")
    prepare.set_defaults(func=prepare_videos)

    processes = subparsers.add_parser("max-processes")
    processes.add_argument("json_file")
    processes.add_argument("dimensions", nargs="+")
    processes.set_defaults(func=max_processes)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("results_dir")
    summary.add_argument("--baseline", help="Results directory of a run to compare to.")
    summary.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
