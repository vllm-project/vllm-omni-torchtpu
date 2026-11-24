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
"""CPU tests for the VBench benchmark helpers in benchmarks/vbench.

They cover the parts that run without a TPU or a GPU: choosing which videos to
generate, turning pipeline outputs into uint8 frames, laying videos out for
VBench and turning VBench results into scores.
"""

import argparse
import collections
import hashlib
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from benchmarks.vbench import generate_videos, gpu_eval_utils

VBENCH_DIR = Path(generate_videos.__file__).parent


def test_bundled_subset_covers_every_dimension_evenly() -> None:
    subset = json.loads(generate_videos.DEFAULT_PROMPTS_JSON.read_text("utf-8"))
    assert len(subset) == 110
    counts = collections.Counter(d for entry in subset for d in entry["dimension"])
    assert set(counts) == set(gpu_eval_utils.NORMALIZE_RANGE)
    assert set(counts.values()) == {10}
    assert generate_videos.load_prompts(generate_videos.DEFAULT_PROMPTS_JSON) == [
        entry["prompt_en"] for entry in subset
    ]


def test_full_prompt_set_is_downloaded_to_the_cache_and_checksummed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "upstream.json"
    upstream.write_text('[{"prompt_en": "a dog", "dimension": ["scene"]}]')
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(generate_videos, "FULL_PROMPTS_URL", upstream.as_uri())
    cached = tmp_path / "cache/vllm_omni_torchtpu/vbench/VBench_full_info.json"
    assert generate_videos.full_prompts_path() == cached

    # Content other than the pinned one is refused and leaves no file behind.
    monkeypatch.setattr(generate_videos, "FULL_PROMPTS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="expected 0000"):
        generate_videos.resolve_prompts_json("full")
    assert not cached.exists()

    digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
    monkeypatch.setattr(generate_videos, "FULL_PROMPTS_SHA256", digest)
    assert generate_videos.resolve_prompts_json("full") == cached
    assert generate_videos.load_prompts(cached) == ["a dog"]

    # A valid cached copy is used without another download...
    upstream.unlink()
    assert generate_videos.resolve_prompts_json("full") == cached
    # ...but a modified one is downloaded again.
    cached.write_text("[]")
    with pytest.raises(urllib.error.URLError):
        generate_videos.resolve_prompts_json("full")
    # Anything else is a path, and nothing is written next to the scripts.
    assert generate_videos.resolve_prompts_json("my.json") == Path("my.json")
    assert not (VBENCH_DIR / "VBench_full_info.json").exists()


def test_parse_key_values() -> None:
    assert generate_videos.parse_key_values(
        ["ulysses_degree=4", "flow_shift=3.0", "sample_solver=unipc", "tiling=true"]
    ) == {
        "ulysses_degree": 4,
        "flow_shift": 3.0,
        "sample_solver": "unipc",
        "tiling": True,
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        generate_videos.parse_key_values(["ulysses_degree"])


def test_parse_seeds_rejects_more_than_vbench_scores() -> None:
    assert generate_videos.parse_seeds("3, 1,3") == [3, 1]
    with pytest.raises(ValueError, match="at most 5"):
        generate_videos.parse_seeds("1,2,3,4,5,6")
    with pytest.raises(ValueError):
        generate_videos.parse_seeds("")


def test_parse_indices() -> None:
    assert generate_videos.parse_indices("0,10,20-22,10", 30) == [0, 10, 20, 21, 22]
    with pytest.raises(ValueError, match="outside"):
        generate_videos.parse_indices("29-30", 30)


def test_select_tasks_goes_seed_by_seed() -> None:
    assert generate_videos.select_tasks(3, [7, 8]) == [
        (0, 7),
        (1, 7),
        (2, 7),
        (0, 8),
        (1, 8),
        (2, 8),
    ]
    assert generate_videos.select_tasks(10, [7], prompt_indices="2-4") == [
        (2, 7),
        (3, 7),
        (4, 7),
    ]


def test_select_tasks_shards_cover_every_video_once() -> None:
    shards = [
        generate_videos.select_tasks(10, [7, 8], shard_id=shard, num_shards=3)
        for shard in range(3)
    ]
    assert sorted(task for shard in shards for task in shard) == sorted(
        generate_videos.select_tasks(10, [7, 8])
    )
    with pytest.raises(ValueError, match="shard-id"):
        generate_videos.select_tasks(10, [7], shard_id=3, num_shards=3)


# One 2-frame, 4x6 RGB video in each layout that vllm-omni pipelines return.
_VIDEO_U8 = np.random.default_rng(0).integers(0, 256, (2, 4, 6, 3), dtype=np.uint8)
_VIDEO_UNIT = _VIDEO_U8.astype(np.float32) / 255.0


@pytest.mark.parametrize(
    "frames,tensor_range",
    [
        pytest.param(_VIDEO_U8[None], "negative_one_to_one", id="ltx2-uint8-batch"),
        pytest.param(_VIDEO_UNIT[None], "negative_one_to_one", id="wan-float-batch"),
        pytest.param([_VIDEO_UNIT[None]], "negative_one_to_one", id="list-of-batch"),
        pytest.param(
            torch.from_numpy(_VIDEO_UNIT * 2 - 1).permute(3, 0, 1, 2)[None],
            "negative_one_to_one",
            id="tensor-bcfhw-signed",
        ),
        pytest.param(
            torch.from_numpy(_VIDEO_UNIT).permute(0, 3, 1, 2),
            "zero_to_one",
            id="tensor-fchw-unit",
        ),
        pytest.param(
            [torch.from_numpy(frame) for frame in _VIDEO_UNIT],
            "zero_to_one",
            id="list-of-frame-tensors",
        ),
        pytest.param(
            [Image.fromarray(frame) for frame in _VIDEO_U8],
            "negative_one_to_one",
            id="list-of-pil",
        ),
        pytest.param(
            np.concatenate([_VIDEO_U8, _VIDEO_U8[..., :1]], axis=-1),
            "negative_one_to_one",
            id="rgba",
        ),
    ],
)
def test_to_uint8_video(frames: object, tensor_range: str) -> None:
    video = generate_videos.to_uint8_video(frames, tensor_range)
    assert video.dtype == np.uint8
    np.testing.assert_array_equal(video, _VIDEO_U8)


def test_to_uint8_video_rejects_bad_output() -> None:
    with pytest.raises(ValueError, match="NaN"):
        generate_videos.to_uint8_video(np.full((2, 4, 6, 3), np.nan, np.float32))
    with pytest.raises(ValueError, match="one video per request"):
        generate_videos.to_uint8_video(np.zeros((2, 2, 4, 6, 3), np.uint8))
    with pytest.raises(ValueError, match="no video frames"):
        generate_videos.to_uint8_video([])


def test_extract_video_and_audio() -> None:
    audio = torch.zeros(1, 2, 480, dtype=torch.bfloat16)
    with_audio = SimpleNamespace(
        images=[_VIDEO_U8[None]],
        multimodal_output={"audio": audio, "audio_sample_rate": 24000},
        error=None,
    )
    video, waveform, sample_rate = generate_videos.extract_video_and_audio([with_audio])
    np.testing.assert_array_equal(video, _VIDEO_U8)
    assert waveform is not None
    assert waveform.shape == (2, 480)
    assert waveform.dtype == np.float32
    assert sample_rate == 24000

    silent = SimpleNamespace(images=[_VIDEO_UNIT[None]], multimodal_output={})
    video, waveform, sample_rate = generate_videos.extract_video_and_audio([silent])
    np.testing.assert_array_equal(video, _VIDEO_U8)
    assert waveform is None
    assert sample_rate is None

    failed = SimpleNamespace(images=[], multimodal_output={}, error="out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        generate_videos.extract_video_and_audio([failed])


def _generation_args(**overrides: object) -> argparse.Namespace:
    settings = dict.fromkeys((*generate_videos.GENERATION_SETTINGS, "fps"))
    return argparse.Namespace(**{**settings, **overrides})


def test_resolve_generation_settings() -> None:
    model_defaults = {"height": 480, "width": 832, "fps": 16.0, "flow_shift": 3.0}
    # Flags win over model defaults; unset settings are left to the pipeline.
    assert generate_videos.resolve_generation_settings(
        _generation_args(height=720, num_frames=81), model_defaults
    ) == {"height": 720, "width": 832, "num_frames": 81, "fps": 16.0}
    with pytest.raises(ValueError, match="--fps"):
        generate_videos.resolve_generation_settings(_generation_args(height=720), {})


def test_build_sampling_kwargs_routes_unknown_keys_to_extra_args() -> None:
    kwargs = generate_videos.build_sampling_kwargs(
        {"height": 480, "fps": 16.0},
        {"max_sequence_length": 512, "sample_solver": "unipc", "frame_rate": 24.0},
        {"height", "fps", "frame_rate", "max_sequence_length", "extra_args"},
    )
    assert kwargs == {
        "generator_device": "cpu",
        "height": 480,
        "fps": 16.0,
        "frame_rate": 24.0,
        "max_sequence_length": 512,
        "extra_args": {"sample_solver": "unipc"},
    }


def _write_benchmark(tmp_path: Path, prompts: list[str]) -> Path:
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps([{"prompt_en": p, "dimension": ["scene"]} for p in prompts])
    )
    return path


def _prepare(tmp_path: Path, bench: Path, allow_missing: bool = False) -> list[str]:
    gpu_eval_utils.prepare_videos(
        argparse.Namespace(
            json_file=str(bench),
            download_dir=str(tmp_path / "downloaded"),
            vbench_dir=str(tmp_path / "vbench"),
            eval_json=str(tmp_path / "eval.json"),
            allow_missing=allow_missing,
        )
    )
    return sorted(path.name for path in (tmp_path / "vbench").iterdir())


def test_prepare_videos_orders_seeds_and_keeps_duplicate_prompts(
    tmp_path: Path,
) -> None:
    # VBench lists a few prompts twice; the prompt index keeps them apart.
    bench = _write_benchmark(tmp_path, ["a dog, running", "a cat", "a cat"])
    (tmp_path / "downloaded").mkdir()
    for seed in (12346, 12345):
        for idx in range(3):
            name = generate_videos.video_filename(seed, idx)
            (tmp_path / "downloaded" / name).write_bytes(f"{seed}-{idx}".encode())

    assert _prepare(tmp_path, bench) == [
        "a cat_1-0.mp4",
        "a cat_1-1.mp4",
        "a cat_2-0.mp4",
        "a cat_2-1.mp4",
        "a dog_ running_0-0.mp4",
        "a dog_ running_0-1.mp4",
    ]
    assert (tmp_path / "vbench" / "a cat_2-0.mp4").read_bytes() == b"12345-2"
    assert (tmp_path / "vbench" / "a cat_2-1.mp4").read_bytes() == b"12346-2"
    # Nothing is missing, so VBench evaluates the metadata unchanged.
    assert json.loads((tmp_path / "eval.json").read_text()) == json.loads(
        bench.read_text()
    )


def test_prepare_videos_rejects_incomplete_set(tmp_path: Path) -> None:
    bench = _write_benchmark(tmp_path, ["a dog", "a cat"])
    (tmp_path / "downloaded").mkdir()
    (tmp_path / "downloaded" / "video_12345_1.mp4").write_bytes(b"x")

    with pytest.raises(ValueError, match="1 videos are missing.*prompt 0 seed 12345"):
        _prepare(tmp_path, bench)
    # Prompts without videos are dropped from what VBench evaluates, and the
    # link is named after the position in that shorter list.
    assert _prepare(tmp_path, bench, allow_missing=True) == ["a cat_0-0.mp4"]
    assert json.loads((tmp_path / "eval.json").read_text()) == [
        {"prompt_en": "a cat", "dimension": ["scene"]}
    ]


def test_max_processes_is_the_smallest_dimension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {"prompt_en": "a", "dimension": ["scene", "color"]},
                {"prompt_en": "b", "dimension": ["scene"]},
                {"prompt_en": "c", "dimension": ["scene"]},
            ]
        )
    )
    args = argparse.Namespace(json_file=str(path), dimensions=["scene", "color"])
    gpu_eval_utils.max_processes(args)
    assert capsys.readouterr().out.strip() == "1"
    args.dimensions = ["scene"]
    gpu_eval_utils.max_processes(args)
    assert capsys.readouterr().out.strip() == "3"
    args.dimensions = ["scene", "human_action"]
    with pytest.raises(ValueError, match="No videos for dimensions: human_action"):
        gpu_eval_utils.max_processes(args)


_VBENCH_SOURCES = {
    "setup.py": "def check_torch_version():\n    raise RuntimeError\n",
    "vbench/distributed.py": (
        "backend = 'gloo' if os.name == 'nt' else 'nccl'\n"
        "    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))\n"
    ),
    "evaluate.py": (
        "    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')\n"
        '    device = torch.device("cuda")\n'
    ),
    "vbench/__init__.py": (
        "import os\n"
        "            for prompt_dict in full_info_list:\n"
        "                # if the prompt belongs to any dimension we want to evaluate\n"
        '                if set(dimension_list) & set(prompt_dict["dimension"]): \n'
        "                    prompt = prompt_dict['prompt_en']\n"
        "                    prompt_dict['video_list'] = []\n"
        "                    for i in range(5): # video index for the same prompt\n"
        "                        intended_video_name = "
        "f'{prompt}{special_str}-{str(i)}{postfix}'\n"
        "        save_json(cur_full_info_list, cur_full_info_path)\n"
        "        print0(f'Evaluation meta data saved to {cur_full_info_path}')\n"
    ),
    "vbench/color.py": (
        "    success_rate = success_frame_count_all / video_count\n"
        "        all_results = success_frame_count / frame_count\n"
    ),
    "vbench/third_party/ViCLIP/viclip_text.py": "from pkg_resources import packaging\n",
}


def test_patch_vbench_is_idempotent_and_detects_upstream_drift(tmp_path: Path) -> None:
    for name, text in _VBENCH_SOURCES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    args = argparse.Namespace(vbench_dir=str(tmp_path))

    gpu_eval_utils.patch_vbench(args)
    patched = {name: (tmp_path / name).read_text() for name in _VBENCH_SOURCES}
    assert "f'{safe_prompt}_{idx}-{str(i)}{postfix}'" in patched["vbench/__init__.py"]
    assert "cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" in patched["evaluate.py"]
    assert "os.environ.get('VBENCH_RUN_TIMESTAMP') or" in patched["evaluate.py"]
    assert "if get_rank() == 0:" in patched["vbench/__init__.py"]
    assert "if video_count > 0 else 0.0" in patched["vbench/color.py"]
    assert "pkg_resources" not in patched["vbench/third_party/ViCLIP/viclip_text.py"]
    gpu_eval_utils.patch_vbench(args)
    assert patched == {name: (tmp_path / name).read_text() for name in _VBENCH_SOURCES}

    (tmp_path / "evaluate.py").write_text('    device = torch.device("gpu")\n')
    with pytest.raises(RuntimeError, match="Upstream may have changed"):
        gpu_eval_utils.patch_vbench(args)


def _write_results(results_dir: Path, scores: dict[str, float]) -> None:
    """Write a VBench results file with two prompts and two videos per prompt."""
    results_dir.mkdir()
    results = {}
    for dimension, score in scores.items():
        per_video_scale = 100 if dimension == "imaging_quality" else 1
        entries = [
            {
                "video_path": f"/videos/{prompt}_{idx}-{i}.mp4",
                "video_results": (score + offset) * per_video_scale,
            }
            for idx, (prompt, offset) in enumerate([("a dog", -0.01), ("a cat", 0.01)])
            for i in range(2)
        ]
        results[dimension] = [score, entries]
    (results_dir / "results_2026-01-01-00:00:00_eval_results.json").write_text(
        json.dumps(results)
    )


def test_total_scores_follow_vbench_weighting() -> None:
    # Every dimension at the top of its range normalizes to 1.
    best = {name: high for name, (_, high) in gpu_eval_utils.NORMALIZE_RANGE.items()}
    assert gpu_eval_utils.total_scores(best) == pytest.approx(
        {"Quality": 1.0, "Semantic": 1.0, "Total": 1.0}
    )
    # dynamic_degree weighs half as much as the six other quality dimensions,
    # and quality weighs four times as much as semantic in the total.
    scores = {**best, "dynamic_degree": 0.0, "color": 0.0}
    assert gpu_eval_utils.total_scores(scores) == pytest.approx(
        {
            "Quality": 6 / 6.5,
            "Semantic": 8 / 9,
            "Total": (4 * 6 / 6.5 + 8 / 9) / 5,
        }
    )
    partial = {"subject_consistency": 0.9}
    assert gpu_eval_utils.total_scores(partial) == {
        "Quality": None,
        "Semantic": None,
        "Total": None,
    }


def test_summarize_against_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scores = dict.fromkeys(gpu_eval_utils.NORMALIZE_RANGE, 0.5)
    _write_results(tmp_path / "base", scores)
    _write_results(tmp_path / "new", {**scores, "imaging_quality": 0.6})

    gpu_eval_utils.summarize(
        argparse.Namespace(
            results_dir=str(tmp_path / "new"), baseline=str(tmp_path / "base")
        )
    )
    table = capsys.readouterr().out.splitlines()
    assert table[0].startswith("| Dimension | Group | Videos | Score | Normalized |")
    # imaging_quality per-video results are on a 0-100 scale; the others on 0-1.
    imaging = next(line for line in table if line.startswith("| imaging_quality"))
    assert imaging == (
        "| imaging_quality | Quality | 4 | 0.6000 | 60.00% | 0.5000 | +0.1000 "
        "| +0.1000 ± 0.0000 |"
    )
    assert sum(line.startswith("| **") for line in table) == 3
    assert not any("Not evaluated" in line for line in table)
