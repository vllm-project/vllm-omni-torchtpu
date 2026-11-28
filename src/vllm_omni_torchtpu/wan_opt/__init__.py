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
"""Opt-in Wan2.2 TPU optimizations, each gated by its own environment variable.

All submodules are lazily imported from an explicit registry only when their
corresponding ``VLLM_OMNI_WAN_*`` environment gate is enabled. When all gates
are off (the default), ``apply_wan_opt_engine_patches()`` and
``apply_wan_opt_patches()`` perform zero submodule imports, install zero
``sys.meta_path`` hooks, and emit zero log lines.

``apply_wan_opt_patches()`` runs at the end of
``patches.apply_omni_model_specific_patches()`` (in every diffusion worker
process, after the stock Wan patches), so a submodule may wrap methods those
patches installed (e.g. the fused ``WanSelfAttention.forward``).

Engine / parent-process patches live in ``engine_*`` submodules and run from
``apply_wan_opt_engine_patches()`` via ``patches.apply_omni_tpu_patches()`` at
``OmniTpuPlatform`` construction in *every* process (engine and workers).
Because that runs while vLLM-Omni may still be importing and in the engine
process (which must never touch the TPU runtime), ``engine_*`` modules may
import only the stdlib, ``vllm_torchtpu.envs``, and ``wan_opt._common`` at top
level, registering patches via ``wan_opt._common.when_imported()``.
"""

import importlib
from collections.abc import Callable
from functools import cache as run_once

from vllm_torchtpu.envs import env_bool, env_int
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Registry entries: (submodule_name, env_predicate, apply_fn_name).
# Checked BEFORE importing the submodule so default-OFF incurs zero import cost.
_ENGINE_PATCHES: tuple[tuple[str, Callable[[], bool], str], ...] = (
    (
        "engine_output_path",
        env_bool("VLLM_OMNI_WAN_FAST_OUTPUT", False),
        "apply_engine",
    ),
)

_WORKER_PATCHES: tuple[tuple[str, Callable[[], bool], str], ...] = (
    ("attn_kernel", env_bool("VLLM_OMNI_WAN_ATTN_FORK", False), "apply"),
    ("fp8_linear", env_bool("VLLM_OMNI_WAN_FP8_LINEAR", False), "apply"),
    ("output_path", env_bool("VLLM_OMNI_WAN_FAST_OUTPUT", False), "apply"),
    (
        "ulysses_pipe",
        lambda: env_int("VLLM_OMNI_WAN_ULYSSES_CHUNKS", 1)() > 1,
        "apply",
    ),
    ("vae_parallel", env_bool("VLLM_OMNI_WAN_VAE_PP_FIX", False), "apply"),
)


@run_once
def apply_wan_opt_engine_patches() -> list[str]:
    """Register opt-in engine-side Wan hooks whose env gates are enabled.

    Returns:
        Names of the ``engine_*`` submodules whose hook was registered.
    """
    applied: list[str] = []
    for name, is_enabled, fn_name in _ENGINE_PATCHES:
        if not is_enabled():
            continue
        module = importlib.import_module(f"{__name__}.{name}")
        hook = getattr(module, fn_name, None)
        if hook is not None and hook():
            applied.append(name)
    if applied:
        logger.info("wan_opt: engine-side hooks registered = %s", applied)
    return applied


@run_once
def apply_wan_opt_patches() -> list[str]:
    """Import and apply opt-in Wan submodules whose env gates are enabled.

    Returns:
        Names of the submodules whose ``apply()`` returned True.
    """
    applied: list[str] = []
    for name, is_enabled, fn_name in _WORKER_PATCHES:
        if not is_enabled():
            continue
        module = importlib.import_module(f"{__name__}.{name}")
        apply_fn = getattr(module, fn_name, None)
        if apply_fn is not None and apply_fn():
            applied.append(name)
            logger.info("wan_opt: applied %s", name)
    if applied:
        logger.info("wan_opt: active optimizations = %s", applied)
    return applied
