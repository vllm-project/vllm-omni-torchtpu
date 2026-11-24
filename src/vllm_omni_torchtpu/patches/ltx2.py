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
"""TPU runtime patches for vLLM-Omni LTX-2 diffusion models."""

from __future__ import annotations

from typing import Any

import torch
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LTX2_PER_BATCH_TIMESTEP_PATCHED = "_tpu_ltx2_per_batch_timestep_patched"


def apply_ltx2_patches() -> None:
    """Apply LTX-2-specific diffusion runtime patches."""
    _patch_ltx2_per_batch_timestep()


def _patch_ltx2_per_batch_timestep() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once vLLM-Omni's LTXRuntime._denoise_timestep_kwargs
    # stops expanding unconditioned timesteps for sequence parallelism.
    #
    # Problem: In vllm-omni 0.32.0rc1
    #   (vllm_omni/diffusion/models/ltx2/ltx2_runtime.py::
    #   LTXRuntime._denoise_timestep_kwargs), the method discards forward_ctx
    #   and denoise_ctx and calls guidance_executor.timestep_kwargs with
    #   expand_for_sequence_parallel=True. That expands each sample's scalar
    #   timestep to (B, T_video) and (B, T_audio), causing time_embed and every
    #   transformer block to project and materialize per-token AdaLN modulation
    #   tables even though all tokens share the same timestep.
    # Fix: Replace LTXRuntime._denoise_timestep_kwargs (copied from vllm-omni
    #   0.32.0rc1, since wrapping cannot override the hardcoded keyword
    #   argument) to pass expand_for_sequence_parallel=False. Unconditioned
    #   timesteps stay 1-D (B,), keeping temb at (B, 1, C) so it broadcasts
    #   into each block's norm modulation.
    # Scope:
    #   - LTX2VideoTransformer3DModel._build_sp_plan declares
    #     SequenceParallelInput(split_dim=1, expected_dims=2, split_output=False)
    #     on timestep, so 1-D (B,) tensors skip sequence sharding and stay
    #     replicated across SP ranks. They take the dimension-mismatch path of
    #     SequenceParallelSplitHook._prepare_sp_input
    #     (vllm_omni/diffusion/hooks/sequence_parallel.py), which logs
    #     "Expected tensor with 2 dims, got 1. Skipping split." through
    #     warning_once: under SP every rank prints it once for T2V, and it is
    #     expected.
    #   - Conditioned I2V requests with a conditioning_mask still override
    #     timestep with the 2-D (B, T_video) masked tensor in
    #     LTXI2VConditioningMixin._denoise_timestep_kwargs (which is sharded
    #     by the SP plan), while keeping audio_timestep 1-D (B,).
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime

    if getattr(
        LTXRuntime._denoise_timestep_kwargs,
        _LTX2_PER_BATCH_TIMESTEP_PATCHED,
        False,
    ):
        return

    def _per_batch_denoise_timestep_kwargs(
        self: Any,
        ts: torch.Tensor,
        forward_ctx: Any,
        denoise_ctx: Any,
        *,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        del forward_ctx, denoise_ctx
        return self.guidance_executor.timestep_kwargs(
            ts,
            video_token_count=video_token_count,
            audio_token_count=audio_token_count,
            expand_for_sequence_parallel=False,
        )

    setattr(
        _per_batch_denoise_timestep_kwargs,
        _LTX2_PER_BATCH_TIMESTEP_PATCHED,
        True,
    )
    LTXRuntime._denoise_timestep_kwargs = _per_batch_denoise_timestep_kwargs
    logger.info("Applied TPU patch: per-batch LTX-2 unconditioned timestep.")
