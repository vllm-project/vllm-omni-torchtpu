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
"""Isolation tests proving ``wan_opt`` has zero overhead when all gates are off."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vllm_omni_torchtpu import wan_opt
from vllm_omni_torchtpu.wan_opt import _common


def _clear_wan_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("VLLM_OMNI_WAN_"):
            monkeypatch.delenv(key, raising=False)


def test_default_off_no_submodules_no_hooks_no_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _clear_wan_env(monkeypatch)
    wan_opt.apply_wan_opt_engine_patches.cache_clear()
    wan_opt.apply_wan_opt_patches.cache_clear()

    # Record pre-existing wan_opt submodules (only _common may be loaded by this file).
    before_submods = {
        name
        for name in sys.modules
        if name.startswith("vllm_omni_torchtpu.wan_opt.")
        and name != "vllm_omni_torchtpu.wan_opt._common"
    }
    before_meta_path = list(sys.meta_path)

    with caplog.at_level(logging.DEBUG, logger="vllm_omni_torchtpu.wan_opt"):
        assert wan_opt.apply_wan_opt_engine_patches() == []
        assert wan_opt.apply_wan_opt_patches() == []

    after_submods = {
        name
        for name in sys.modules
        if name.startswith("vllm_omni_torchtpu.wan_opt.")
        and name != "vllm_omni_torchtpu.wan_opt._common"
    }
    assert after_submods == before_submods
    assert list(sys.meta_path) == before_meta_path
    assert _common._FINDER not in sys.meta_path
    assert caplog.records == []


def test_default_off_fresh_interpreter_zero_heavy_imports() -> None:
    """In a clean interpreter, default-OFF must import zero heavy backends."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_OMNI_WAN_")}
    script = """
import json
import sys
from vllm_omni_torchtpu import wan_opt

before_modules = set(sys.modules)
meta_before = list(sys.meta_path)
eng = wan_opt.apply_wan_opt_engine_patches()
wrk = wan_opt.apply_wan_opt_patches()
assert eng == [], eng
assert wrk == [], wrk
assert list(sys.meta_path) == meta_before

added_modules = set(sys.modules) - before_modules
loaded_wan_submods = sorted(
    m for m in sys.modules if m.startswith("vllm_omni_torchtpu.wan_opt.")
)
heavy_present = sorted(
    m
    for m in (
        "torch_tpu",
        "diffusers",
        "vllm_torchtpu.layers.adapter.linear_common",
    )
    if m in sys.modules
)
heavy_added = sorted(
    m
    for m in added_modules
    if m.startswith(("jax", "torch_tpu", "diffusers", "vllm_omni.diffusion"))
)
print(
    json.dumps(
        {
            "submods": loaded_wan_submods,
            "heavy_present": heavy_present,
            "heavy_added": heavy_added,
        }
    )
)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["submods"] == []
    assert result["heavy_present"] == []
    assert result["heavy_added"] == []


def test_default_off_leaves_upstream_hooks_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion import ipc
    from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
        DistributedVaeExecutor,
    )
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanSelfAttention,
        WanTransformer3DModel,
    )
    from vllm_omni.diffusion.postprocess import media as media_mod

    from vllm_omni_torchtpu.attention import TpuFlashAttentionImpl

    _clear_wan_env(monkeypatch)
    wan_opt.apply_wan_opt_engine_patches.cache_clear()
    wan_opt.apply_wan_opt_patches.cache_clear()

    snap = {
        "ipc_pack": ipc._pack_tensor_if_large,
        "media_finalize": media_mod.finalize_diffusion_media,
        "wan_model_fwd": WanTransformer3DModel.forward,
        "wan_self_fwd": WanSelfAttention.forward,
        "flash_fwd": TpuFlashAttentionImpl.forward,
        "vae_exec": DistributedVaeExecutor.execute,
    }
    assert wan_opt.apply_wan_opt_engine_patches() == []
    assert wan_opt.apply_wan_opt_patches() == []
    assert snap == {
        "ipc_pack": ipc._pack_tensor_if_large,
        "media_finalize": media_mod.finalize_diffusion_media,
        "wan_model_fwd": WanTransformer3DModel.forward,
        "wan_self_fwd": WanSelfAttention.forward,
        "flash_fwd": TpuFlashAttentionImpl.forward,
        "vae_exec": DistributedVaeExecutor.execute,
    }


def test_invalid_env_gate_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_wan_env(monkeypatch)
    wan_opt.apply_wan_opt_patches.cache_clear()
    monkeypatch.setenv("VLLM_OMNI_WAN_FP8_LINEAR", "treu")
    with pytest.raises(ValueError, match="Invalid boolean value"):
        wan_opt.apply_wan_opt_patches()

    wan_opt.apply_wan_opt_patches.cache_clear()
    monkeypatch.delenv("VLLM_OMNI_WAN_FP8_LINEAR")
    monkeypatch.setenv("VLLM_OMNI_WAN_ULYSSES_CHUNKS", "abc")
    with pytest.raises(ValueError, match="must be an integer"):
        wan_opt.apply_wan_opt_patches()
    wan_opt.apply_wan_opt_patches.cache_clear()


def test_when_imported_unregisters_finder_when_pending_empties(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod_name = "_wan_opt_dummy_lazy_mod"
    mod_file = tmp_path / f"{mod_name}.py"
    mod_file.write_text("VALUE = 42\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    _common._PENDING.clear()
    if _common._FINDER in sys.meta_path:
        sys.meta_path.remove(_common._FINDER)

    seen: list[types.ModuleType] = []
    try:
        _common.when_imported(mod_name, seen.append)
        assert _common._FINDER in sys.meta_path
        imported = __import__(mod_name)
        assert seen == [imported]
        assert imported.VALUE == 42
        assert _common._FINDER not in sys.meta_path
        assert _common._PENDING == {}
    finally:
        sys.modules.pop(mod_name, None)
        _common._PENDING.clear()
        if _common._FINDER in sys.meta_path:
            sys.meta_path.remove(_common._FINDER)
