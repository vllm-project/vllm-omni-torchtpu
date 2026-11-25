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
"""vLLM-Omni platform plugin support for Google Cloud TPU."""

from vllm_torchtpu.tpu_info import get_num_chips


def register_omni_tpu_platform() -> str | None:
    """Plugin entry point for vllm_omni.platform_plugins."""
    if get_num_chips() <= 0:
        return None
    return "vllm_omni_torchtpu.platform.OmniTpuPlatform"


def register_omni_general_plugins() -> None:
    """Plugin entry point for vllm_omni.general_plugins.

    vLLM-Omni runs its general plugins in every process after imports and
    before it builds a diffusion engine or worker. Installing the warmup-sync
    patch from OmniTpuPlatform.__init__ would hit a circular import of
    DiffusionEngine on a cold StageDiffusionProc start (diffusion_engine.py
    resolves current_omni_platform before class DiffusionEngine is defined), so
    it runs from this general-plugin hook instead. Returns early unless
    register_omni_tpu_platform() finds TPU chips (get_num_chips() > 0); safe
    to call more than once.
    """
    if register_omni_tpu_platform() is None:
        return
    from vllm_omni_torchtpu.patches import apply_omni_general_plugin_patches

    apply_omni_general_plugin_patches()


def register_tpu_platform_alias() -> str | None:
    """Alias core's vLLM platform plugin only when selected via VLLM_PLUGINS.

    When ``VLLM_PLUGINS`` is unset (or already includes ``torchtpu``),
    ``vllm-torchtpu-core``'s ``torchtpu`` entry point activates ``TpuPlatform``
    on its own, so returning ``None`` keeps plain vLLM processes from seeing
    two activated out-of-tree platform plugins.
    """
    import vllm.envs as envs

    if envs.VLLM_PLUGINS is None or "torchtpu" in envs.VLLM_PLUGINS:
        return None
    from vllm_torchtpu.platforms import register_tpu_platform

    return register_tpu_platform()


__all__ = [
    "register_omni_general_plugins",
    "register_omni_tpu_platform",
    "register_tpu_platform_alias",
]
