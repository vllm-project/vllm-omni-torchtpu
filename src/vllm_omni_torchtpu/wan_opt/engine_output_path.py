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
"""W-F2 engine-side hook: fast bit-exact ``finalize_diffusion_media``.

Runs from ``apply_omni_tpu_patches()`` in every process (including the engine).
Top-level imports are restricted to the stdlib, ``vllm_torchtpu.envs``, and
``wan_opt._common`` so platform initialization cannot trigger circular imports
or touch the TPU runtime.

``VLLM_OMNI_WAN_FAST_OUTPUT=1`` replaces ``VideoProcessor.postprocess_video``
inside ``finalize_diffusion_media`` for ``NORMALIZED_FLOAT`` ``BCTHW`` video
with ``output_type="np"``: it performs the exact same ``dtype`` denormalize
(``mul(0.5).add_(0.5).clamp_(0, 1)``), fuses ``BCTHW -> BTHWC`` + ``float32``
conversion, and avoids ``np.stack``'s extra copy for single-item or batched
arrays.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any

from vllm_torchtpu.envs import env_bool

from vllm_omni_torchtpu.wan_opt._common import when_imported

ENV = "VLLM_OMNI_WAN_FAST_OUTPUT"
_ENGINE_OUTPUT_PATCHED = "_tpu_wan_engine_output_patched"


def _patch_postprocess_media(media_mod: ModuleType) -> None:
    orig_finalize = media_mod.finalize_diffusion_media
    if getattr(orig_finalize, _ENGINE_OUTPUT_PATCHED, False):
        return

    from vllm_omni.diffusion.media import (
        VideoTensorEncoding,
        VideoTensorLayout,
        VideoValueRange,
    )

    @functools.wraps(orig_finalize)
    def _fast_finalize_diffusion_media(
        media: Any,
        *,
        sampling_params: Any,
    ) -> dict[str, object]:
        media.validate()
        if not media.prepared_for_transport:
            raise ValueError(
                "Diffusion media reached the engine before transport preparation"
            )
        video = media.video
        output_type = getattr(sampling_params, "output_type", None) or "np"
        if (
            video.constraints.pending_float_consumers
            or output_type != "np"
            or video.spec.encoding is not VideoTensorEncoding.NORMALIZED_FLOAT
            or video.spec.layout is not VideoTensorLayout.BCTHW
        ):
            return orig_finalize(media, sampling_params=sampling_params)

        tensor = video.tensor
        if video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE:
            denorm = tensor.mul(0.5)
            denorm.add_(0.5)
            denorm.clamp_(0.0, 1.0)
            tensor = denorm
        frames = (
            tensor.permute(0, 2, 3, 4, 1).float().contiguous().detach().cpu().numpy()
        )
        return {
            "payload": {"video": frames},
            "metadata": {},
        }

    setattr(_fast_finalize_diffusion_media, _ENGINE_OUTPUT_PATCHED, True)
    media_mod._tpu_wan_orig_finalize_diffusion_media = orig_finalize
    media_mod.finalize_diffusion_media = _fast_finalize_diffusion_media


def _sync_diffusion_engine_binding(engine_mod: ModuleType) -> None:
    if env_bool(ENV, False)():
        from vllm_omni.diffusion.postprocess import media as media_mod

        if hasattr(engine_mod, "finalize_diffusion_media"):
            engine_mod.finalize_diffusion_media = media_mod.finalize_diffusion_media


def apply_engine() -> bool:
    """Register post-import engine hooks when ``VLLM_OMNI_WAN_FAST_OUTPUT`` is set.

    Returns:
        True if the hooks were registered, False when disabled.
    """
    if not env_bool(ENV, False)():
        return False

    when_imported("vllm_omni.diffusion.postprocess.media", _patch_postprocess_media)
    when_imported(
        "vllm_omni.diffusion.diffusion_engine", _sync_diffusion_engine_binding
    )
    return True
