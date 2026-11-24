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
"""Unit tests for src/vllm_omni_torchtpu/patches/ltx2.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.hooks import sequence_parallel
from vllm_omni.diffusion.models.ltx2 import (
    ltx2_conditioning,
    ltx2_guidance,
    ltx2_runtime,
    ltx2_transformer,
)

from vllm_omni_torchtpu.patches import ltx2 as ltx2_patches

# Capture the upstream method at collection, before any test installs patches.
_UPSTREAM_DENOISE_TIMESTEP_KWARGS = ltx2_runtime.LTXRuntime._denoise_timestep_kwargs


class _DummyLTX2Pipeline(
    ltx2_conditioning.LTXI2VConditioningMixin,
    ltx2_runtime.LTXRuntime,
):
    def __init__(self) -> None:
        self.guidance_executor = ltx2_guidance.LTXGuidanceExecutor()
        self.unified_text_image_entry = True


@pytest.fixture
def upstream_denoise_timestep_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset LTXRuntime._denoise_timestep_kwargs to the unpatched upstream method."""
    monkeypatch.setattr(
        ltx2_runtime.LTXRuntime,
        "_denoise_timestep_kwargs",
        _UPSTREAM_DENOISE_TIMESTEP_KWARGS,
    )


@pytest.mark.parametrize(
    ("is_per_token", "expected_sharded"),
    [
        pytest.param(False, False, id="1d_per_batch_unsplit"),
        pytest.param(True, True, id="2d_per_token_sharded"),
    ],
)
def test_ltx2_sp_plan_timestep_sharding(
    monkeypatch: pytest.MonkeyPatch,
    is_per_token: bool,
    expected_sharded: bool,
) -> None:
    """1-D (B,) timesteps bypass SP sharding; 2-D (B, T) I2V timesteps are sharded."""
    batch_size = 2
    video_tokens = 16
    ulysses_degree = 2
    shards: list[tuple[tuple[int, ...], int]] = []

    def recording_sp_shard(x: torch.Tensor, dim: int, **kwargs: object) -> torch.Tensor:
        del kwargs
        shards.append((tuple(x.shape), dim))
        return x.narrow(dim, 0, x.size(dim) // ulysses_degree)

    monkeypatch.setattr(sequence_parallel, "sp_shard", recording_sp_shard)
    plan = ltx2_transformer.LTX2VideoTransformer3DModel._build_sp_plan("split")[""]
    hook = sequence_parallel.SequenceParallelSplitHook(
        plan,
        sequence_parallel.SequenceParallelConfig(ulysses_degree=ulysses_degree),
    )

    per_batch = torch.tensor([900.0, 450.0])
    assert per_batch.shape == (batch_size,)
    if not is_per_token:
        out = hook._prepare_sp_input(per_batch, plan["timestep"])
        assert out is per_batch
        assert shards == []
    else:
        per_token = per_batch.reshape(batch_size, 1).expand(batch_size, video_tokens)
        per_token = per_token.contiguous()
        out = hook._prepare_sp_input(per_token, plan["timestep"])
        assert expected_sharded
        assert out.shape == (batch_size, video_tokens // ulysses_degree)
        assert shards == [((batch_size, video_tokens), 1)]


def test_ltx2_upstream_expands_unconditioned_timestep(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Confirm upstream vLLM-Omni expands unconditioned timesteps to (B, T)."""
    del upstream_denoise_timestep_kwargs
    batch_size = 2
    video_tokens = 16
    audio_tokens = 4
    pipe = _DummyLTX2Pipeline()
    ts = torch.full((batch_size,), 900.0)
    uncond_ctx = SimpleNamespace(
        conditioning_mask=None, conditioning_mask_for_model=None
    )

    kw_upstream = pipe._denoise_timestep_kwargs(
        ts,
        None,
        uncond_ctx,
        video_token_count=video_tokens,
        audio_token_count=audio_tokens,
    )
    assert kw_upstream["timestep"].shape == (batch_size, video_tokens)
    assert kw_upstream["audio_timestep"].shape == (batch_size, audio_tokens)


def test_ltx2_per_batch_timestep_patch_is_idempotent(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Installing the per-batch timestep patch twice leaves the same wrapper."""
    del upstream_denoise_timestep_kwargs
    ltx2_patches._patch_ltx2_per_batch_timestep()
    patched = ltx2_runtime.LTXRuntime._denoise_timestep_kwargs
    assert getattr(patched, ltx2_patches._LTX2_PER_BATCH_TIMESTEP_PATCHED, False)

    ltx2_patches._patch_ltx2_per_batch_timestep()
    assert ltx2_runtime.LTXRuntime._denoise_timestep_kwargs is patched


def test_ltx2_per_batch_timestep_patch_keeps_t2v_1d(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Unconditioned T2V keeps 1-D (B,) video and audio timesteps after patching."""
    del upstream_denoise_timestep_kwargs
    batch_size = 2
    video_tokens = 16
    audio_tokens = 4
    pipe = _DummyLTX2Pipeline()
    ts = torch.full((batch_size,), 900.0)
    uncond_ctx = SimpleNamespace(
        conditioning_mask=None, conditioning_mask_for_model=None
    )

    ltx2_patches._patch_ltx2_per_batch_timestep()
    kw_uncond = pipe._denoise_timestep_kwargs(
        ts,
        None,
        uncond_ctx,
        video_token_count=video_tokens,
        audio_token_count=audio_tokens,
    )
    assert kw_uncond["timestep"].shape == (batch_size,)
    assert kw_uncond["audio_timestep"].shape == (batch_size,)


def test_ltx2_per_batch_timestep_patch_preserves_i2v_conditioning_mask(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Conditioned I2V still overrides video timestep with the 2-D (B, T) mask."""
    del upstream_denoise_timestep_kwargs
    batch_size = 2
    video_tokens = 16
    audio_tokens = 4
    cond_tokens = 4
    pipe = _DummyLTX2Pipeline()
    ts = torch.full((batch_size,), 900.0)
    mask = torch.zeros(batch_size, video_tokens)
    mask[:, :cond_tokens] = 1.0
    cond_ctx = SimpleNamespace(conditioning_mask=mask, conditioning_mask_for_model=mask)

    ltx2_patches._patch_ltx2_per_batch_timestep()
    kw_cond = pipe._denoise_timestep_kwargs(
        ts,
        None,
        cond_ctx,
        video_token_count=video_tokens,
        audio_token_count=audio_tokens,
    )
    assert kw_cond["timestep"].shape == (batch_size, video_tokens)
    assert torch.all(kw_cond["timestep"][:, :cond_tokens] == 0.0)
    assert torch.all(kw_cond["timestep"][:, cond_tokens:] == 900.0)
    assert kw_cond["audio_timestep"].shape == (batch_size,)
