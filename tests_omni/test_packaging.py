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

import pytest

EXPECTED_ENTRY_POINTS = {
    ("vllm_omni.platform_plugins", "torchtpu_omni"): (
        "vllm_omni_torchtpu:register_omni_tpu_platform"
    ),
    ("vllm.platform_plugins", "torchtpu_omni"): (
        "vllm_torchtpu.platforms:register_tpu_platform"
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
