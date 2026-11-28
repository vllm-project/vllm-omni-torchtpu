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
"""CPU unit tests for ``wan_opt.vae_parallel``."""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm_omni.diffusion.distributed.autoencoders import (
    distributed_vae_executor as dve,
)

from vllm_omni_torchtpu.wan_opt import vae_parallel


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _restore_executor_methods() -> None:
    cls = dve.DistributedVaeExecutor
    for orig_attr, target_attr in (
        (vae_parallel._ORIG_SYNC_ERRORS_ATTR, "_sync_errors"),
        (
            vae_parallel._ORIG_COMPUTE_SHAPE_ATTR,
            "_compute_global_padding_shape",
        ),
        (vae_parallel._ORIG_UNPACK_TILES_ATTR, "_unpack_tiles"),
        (vae_parallel._ORIG_EXECUTE_ATTR, "execute"),
    ):
        if hasattr(cls, orig_attr):
            setattr(cls, target_attr, getattr(cls, orig_attr))
            delattr(cls, orig_attr)
    if hasattr(cls, vae_parallel._PATCH_ATTR):
        delattr(cls, vae_parallel._PATCH_ATTR)


def test_vae_parallel_gate_and_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _restore_executor_methods()
    orig_sync = dve.DistributedVaeExecutor._sync_errors
    orig_shape = dve.DistributedVaeExecutor._compute_global_padding_shape
    orig_exec = dve.DistributedVaeExecutor.execute

    mock_info = MagicMock()
    mock_warn = MagicMock()
    monkeypatch.setattr(vae_parallel.logger, "info", mock_info)
    monkeypatch.setattr(vae_parallel.logger, "warning", mock_warn)

    try:
        monkeypatch.delenv(vae_parallel._ENV_FLAG, raising=False)
        assert vae_parallel.apply() is False
        assert dve.DistributedVaeExecutor._sync_errors is orig_sync
        assert dve.DistributedVaeExecutor._compute_global_padding_shape is orig_shape
        assert dve.DistributedVaeExecutor.execute is orig_exec

        monkeypatch.setenv(vae_parallel._ENV_FLAG, "1")
        assert vae_parallel.apply() is True
        assert mock_info.call_count == 1
        assert mock_warn.call_count == 0
        patched_sync = dve.DistributedVaeExecutor._sync_errors
        patched_shape = dve.DistributedVaeExecutor._compute_global_padding_shape
        patched_exec = dve.DistributedVaeExecutor.execute
        assert patched_sync is not orig_sync
        assert patched_shape is not orig_shape
        assert patched_exec is not orig_exec

        assert vae_parallel.apply() is True
        assert mock_info.call_count == 1
        assert dve.DistributedVaeExecutor._sync_errors is patched_sync
        assert dve.DistributedVaeExecutor._compute_global_padding_shape is patched_shape
        assert dve.DistributedVaeExecutor.execute is patched_exec
    finally:
        _restore_executor_methods()


def _gloo_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ[vae_parallel._ENV_FLAG] = "1"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        dev_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        orig_get_world_group = dve.get_world_group
        dve.get_world_group = lambda: SimpleNamespace(
            device_group=dev_group,
            cpu_group=cpu_group,
        )
        try:
            assert vae_parallel.apply() is True
            executor = dve.DistributedVaeExecutor()
            executor.set_parallel_size(world_size, mode="tile")

            # 1. Execute parity with uneven tile shapes and empty ranks (5 / 8).
            tile_specs = [
                (0, (0, 0), (1, 3, 4, 8, 8)),
                (1, (0, 1), (1, 3, 4, 8, 5)),
                (2, (1, 0), (1, 3, 4, 6, 8)),
                (3, (1, 1), (1, 3, 4, 6, 5)),
                (4, (1, 2), (1, 3, 4, 6, 4)),
            ]
            torch.manual_seed(42)
            full_tiles = {
                coord: torch.randn(shape, dtype=torch.float32)
                for _, coord, shape in tile_specs
            }

            def split_fn(_z: torch.Tensor):
                tasks = [
                    dve.TileTask(
                        tile_id=tid,
                        grid_coord=coord,
                        tensor=full_tiles[coord],
                        workload=shape[-2] * shape[-1],
                    )
                    for tid, coord, shape in tile_specs
                ]
                grid = dve.GridSpec(
                    split_dims=(3, 4),
                    grid_shape=(2, 3),
                    output_dtype=torch.float32,
                )
                return tasks, grid

            exec_calls: list[int] = []

            def exec_fn(task: dve.TileTask) -> torch.Tensor:
                exec_calls.append(task.tile_id)
                return task.tensor * 2.0 + 1.0

            def merge_fn(
                coord_map: dict[tuple[int, ...], torch.Tensor],
                _grid: dve.GridSpec,
            ) -> torch.Tensor:
                parts = [coord_map[coord].flatten() for _, coord, _ in tile_specs]
                return torch.cat(parts, dim=0)

            op = dve.DistributedOperator(
                split=split_fn,
                exec=exec_fn,
                merge=merge_fn,
            )
            out = executor.execute(
                torch.zeros((1, 3, 4, 14, 17)),
                op,
                broadcast_result=False,
            )
            # Empty ranks (ranks 5-7) must not speculatively run operator.exec.
            expected_calls = 1 if rank < len(tile_specs) else 0
            assert len(exec_calls) == expected_calls

            if rank == 0:
                expected = torch.cat(
                    [(full_tiles[c] * 2.0 + 1.0).flatten() for _, c, _ in tile_specs],
                    dim=0,
                )
                torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
            else:
                assert out.numel() == 0

            # 2. _sync_errors precedence: RuntimeError (2) on rank 1 beats OOM (1).
            expected_msg = "boom" if rank == 1 else "another rank failed"
            with (
                pytest.raises(RuntimeError, match=expected_msg),
                executor._sync_errors("test"),
            ):
                if rank == 0:
                    raise torch.OutOfMemoryError("oom")
                if rank == 1:
                    raise ValueError("boom")
        finally:
            dve.get_world_group = orig_get_world_group
            _restore_executor_methods()
    finally:
        dist.destroy_process_group()


def test_gloo_execute_parity_and_error_precedence() -> None:
    port = _find_free_port()
    mp.spawn(
        _gloo_worker,
        args=(8, port),
        nprocs=8,
        join=True,
    )
