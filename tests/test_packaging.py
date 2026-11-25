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
"""CPU unit tests for vllm-omni-torchtpu distribution metadata and entry points."""

from importlib.metadata import distribution, entry_points
from types import SimpleNamespace

import pytest
import vllm.envs as vllm_envs

from vllm_omni_torchtpu import register_tpu_platform_alias
from vllm_omni_torchtpu.worker import tpu_ar_worker
from vllm_omni_torchtpu.worker.tpu_ar_worker import _require_core_model_runner_cls

EXPECTED_ENTRY_POINTS = {
    ("vllm_omni.platform_plugins", "torchtpu_omni"): (
        "vllm_omni_torchtpu:register_omni_tpu_platform"
    ),
    ("vllm_omni.general_plugins", "torchtpu_omni"): (
        "vllm_omni_torchtpu:register_omni_general_plugins"
    ),
    ("vllm.platform_plugins", "torchtpu_omni"): (
        "vllm_omni_torchtpu:register_tpu_platform_alias"
    ),
}


@pytest.fixture(autouse=True)
def _sync_tpu() -> None:
    """Packaging metadata tests run on CPU without a TPU accelerator."""


def test_distribution_entry_points() -> None:
    dist = distribution("vllm-omni-torchtpu")
    actual = {(ep.group, ep.name): ep.value for ep in dist.entry_points}
    assert actual == EXPECTED_ENTRY_POINTS

    for (group, name), target in EXPECTED_ENTRY_POINTS.items():
        matches = [ep.value for ep in entry_points(group=group, name=name)]
        assert matches == [target]


def test_register_tpu_platform_alias_respects_vllm_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    assert register_tpu_platform_alias() is None

    monkeypatch.setenv("VLLM_PLUGINS", "torchtpu,torchtpu_layers")
    assert register_tpu_platform_alias() is None

    monkeypatch.setenv(
        "VLLM_PLUGINS",
        "torchtpu_omni,torchtpu_layers,vllm_omni_register_models",
    )
    assert register_tpu_platform_alias() == (
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform"
    )


def test_require_core_model_runner_cls_rejects_old_core() -> None:
    class _LegacyTPUWorker:
        pass

    with pytest.raises(ImportError, match="model_runner_cls"):
        _require_core_model_runner_cls(_LegacyTPUWorker)

    _require_core_model_runner_cls(SimpleNamespace(model_runner_cls=object))


def test_tpu_ar_worker_init_rejects_old_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module imports on core 0.33.0; building the AR worker fails."""

    class _LegacyTPUWorker:
        pass

    monkeypatch.setattr(tpu_ar_worker, "TPUWorker", _LegacyTPUWorker)
    with pytest.raises(ImportError, match="core-git"):
        tpu_ar_worker.TPUARWorker()
