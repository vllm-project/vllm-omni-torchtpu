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

"""TPU compatibility fix for distributed tile VAE decoding.

``vllm_omni``'s ``DistributedVaeExecutor`` uses ``dist.all_reduce(...,
op=dist.ReduceOp.MAX)`` in both ``_sync_errors`` and
``_compute_global_padding_shape``. On ``torch_tpu``, XLA lowering currently
supports only ``ReduceOp.SUM`` for device ``all_reduce``, causing
``vae_parallel_mode="tile"`` to fail compilation. In addition, running
``execute`` asynchronously without device synchronization across collectives
and rank-0-only merge causes PJRT collective wave hangs on startup dummy runs.

When ``VLLM_OMNI_WAN_VAE_PP_FIX=1`` is set, this module idempotently patches
``DistributedVaeExecutor`` to:
- Replace ``ReduceOp.MAX`` with ``dist.all_gather_single`` + local ``.max()``.
- Synchronize the TPU stream around collective boundaries in ``execute``
  (including rank-0 merge completion inside ``_sync_errors("tile merge")``).
- Batch-read tile metadata via ``.tolist()`` in ``_unpack_tiles`` instead of
  per-element scalar device-to-host syncs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.envs import env_bool
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)

_ENV_FLAG = "VLLM_OMNI_WAN_VAE_PP_FIX"

_PATCH_ATTR = "_wan_opt_vae_pp_patched"
_ORIG_SYNC_ERRORS_ATTR = "_wan_opt_orig_sync_errors"
_ORIG_COMPUTE_SHAPE_ATTR = "_wan_opt_orig_compute_global_padding_shape"
_ORIG_EXECUTE_ATTR = "_wan_opt_orig_execute"
_ORIG_UNPACK_TILES_ATTR = "_wan_opt_orig_unpack_tiles"


def _sync_device(device: torch.device) -> None:
    import torch

    if device.type != "cpu" and hasattr(torch, "accelerator"):
        torch.accelerator.synchronize()


@contextmanager
def _sync_errors_all_gather(self: Any, stage: str) -> Iterator[None]:
    """Agrees on local-stage failures without ``ReduceOp.MAX`` collectives.

    Args:
      self: ``DistributedVaeExecutor`` instance.
      stage: Human-readable stage name included in raised error messages.

    Yields:
      Control to the wrapped stage block on every rank before error reduction.
    """
    import torch
    import torch.distributed as dist

    status = 0
    message = ""
    try:
        yield
    except torch.OutOfMemoryError as error:
        status = 1
        message = str(error)
    except Exception as error:
        # Matches upstream DistributedVaeExecutor._sync_errors: every rank must
        # catch local failures and participate in the collective below so peer
        # ranks do not deadlock waiting on all_gather_single. Non-OOM errors
        # take precedence over OOM on another rank.
        status = 2
        message = f"{type(error).__name__}: {error}"

    self._error_status.fill_(status)
    # Allocate only on the CPU error buffer so OOM reporting stays off-device.
    gathered = torch.empty(
        (self.world_size,),
        dtype=self._error_status.dtype,
        device=self._error_status.device,
    )
    dist.all_gather_single(
        gathered,
        self._error_status.reshape(1),
        group=self._error_group,
    )
    global_status = int(gathered.max().item())
    if global_status == 2:
        detail = message if status == 2 else "another rank failed"
        raise RuntimeError(f"Distributed VAE {stage} failed: {detail}")
    if global_status == 1:
        detail = message if status == 1 else "another rank ran out of memory"
        raise torch.OutOfMemoryError(f"Distributed VAE {stage} failed: {detail}")


def _compute_global_padding_shape_all_gather(
    self: Any,
    local_results: list[tuple[int, torch.Tensor]],
    output_ndim: int,
    device: torch.device,
) -> list[int]:
    """Computes global tile padding shape via ``all_gather_single`` + max.

    Args:
      self: ``DistributedVaeExecutor`` instance.
      local_results: List of ``(tile_id, tile_tensor)`` pairs decoded locally.
      output_ndim: Rank of the decoded output tensor.
      device: Execution device for the collective tensor.

    Returns:
      Maximum ``[tile_count, *tile_dims]`` across all ranks.
    """
    import torch
    import torch.distributed as dist

    with self._sync_errors("shape preparation"):
        local_tile_max_dims = [0] * output_ndim
        for _, tile_tensor in local_results:
            for dim_idx, dim_size in enumerate(tile_tensor.shape):
                local_tile_max_dims[dim_idx] = max(
                    local_tile_max_dims[dim_idx], dim_size
                )
        local_shape_stat = torch.tensor(
            [len(local_results), *local_tile_max_dims],
            dtype=torch.int64,
            device=device,
        )

    gathered = torch.empty(
        (self.world_size, local_shape_stat.numel()),
        dtype=local_shape_stat.dtype,
        device=device,
    )
    dist.all_gather_single(
        gathered,
        local_shape_stat.unsqueeze(0),
        group=self.group,
    )
    return gathered.max(dim=0).values.tolist()


def _unpack_tiles_batched(
    self: Any,
    meta_gather: list[torch.Tensor],
    tile_gather: list[torch.Tensor],
    grid_spec: Any,
    tid_coord_map: dict[int, tuple[int, ...]],
) -> dict[tuple[int, ...], torch.Tensor]:
    """Unpacks gathered tiles after converting per-rank metadata once.

    Args:
      self: ``DistributedVaeExecutor`` instance.
      meta_gather: Per-rank tile metadata tensors from ``gather_tensors``.
      tile_gather: Per-rank padded tile tensors from ``gather_tensors``.
      grid_spec: Grid specification describing ``split_dims``.
      tid_coord_map: Mapping from ``tile_id`` to grid coordinate tuple.

    Returns:
      Mapping from grid coordinate tuple to unpadded tile tensor.
    """
    coord_tensor_map: dict[tuple[int, ...], torch.Tensor] = {}
    for r in range(self.world_size):
        meta_rows = meta_gather[r].tolist()
        tiles_src = tile_gather[r]
        for idx, row in enumerate(meta_rows):
            tid = int(row[0])
            if tid < 0:
                continue
            slices_list = [slice(None)] * tiles_src[idx].ndim
            for i, dim in enumerate(grid_spec.split_dims):
                slices_list[dim] = slice(0, int(row[i + 1]))
            coord_tensor_map[tid_coord_map[tid]] = tiles_src[idx][tuple(slices_list)]
    return coord_tensor_map


def _execute_patched(
    self: Any,
    z: torch.Tensor,
    operator: Any,
    broadcast_result: bool = True,
) -> torch.Tensor:
    """Executes distributed tiles with device syncs around TPU collectives.

    Args:
      self: ``DistributedVaeExecutor`` instance.
      z: Input latent tensor.
      operator: ``DistributedOperator`` with ``split``, ``exec``, ``merge``.
      broadcast_result: Whether rank 0 broadcasts the merged result to all ranks.

    Returns:
      Merged output tensor (on rank 0, or all ranks if ``broadcast_result``).
    """
    import torch

    pp_size = min(self.parallel_size, self.world_size)
    dev = z.device
    _sync_device(dev)

    with self._sync_errors("tile decode"):
        tiletask_list, grid_spec = operator.split(z)
        tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}
        assigned = self._balance_tasks(tiletask_list, pp_size)
        local_tasks = assigned[self.rank] if self.rank < pp_size else []
        local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]
        _sync_device(dev)

    global_padding_shape = self._compute_global_padding_shape(
        local_results, z.ndim, dev
    )

    output_dtype = (
        grid_spec.output_dtype if grid_spec.output_dtype is not None else z.dtype
    )
    with self._sync_errors("tile packing"):
        local_tile_tensor, local_meta_tensor = self._pack_local_tiles(
            local_results, global_padding_shape, grid_spec, dev, output_dtype
        )
        _sync_device(dev)

    meta_gather = self.gather_tensors(local_meta_tensor)
    tile_gather = self.gather_tensors(local_tile_tensor)
    _sync_device(dev)

    with self._sync_errors("tile merge"):
        if self.rank != 0:
            result = torch.empty(0, device=dev)
        else:
            coord_tensor_map = self._unpack_tiles(
                meta_gather, tile_gather, grid_spec, tid_coord_map
            )
            result = operator.merge(coord_tensor_map, grid_spec)
        _sync_device(dev)

    if broadcast_result:
        result = self._sync_final_result(result, z.ndim, dev, output_dtype)
        _sync_device(dev)

    return result


def apply() -> bool:
    """Installs the TPU tile-VAE collective fix when enabled by env.

    Returns:
      ``True`` if ``DistributedVaeExecutor`` is patched, ``False`` if disabled.
    """
    if not env_bool(_ENV_FLAG, default=False)():
        return False

    from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
        DistributedVaeExecutor,
    )

    if not getattr(DistributedVaeExecutor, _PATCH_ATTR, False):
        setattr(
            DistributedVaeExecutor,
            _ORIG_SYNC_ERRORS_ATTR,
            DistributedVaeExecutor._sync_errors,
        )
        setattr(
            DistributedVaeExecutor,
            _ORIG_COMPUTE_SHAPE_ATTR,
            DistributedVaeExecutor._compute_global_padding_shape,
        )
        setattr(
            DistributedVaeExecutor,
            _ORIG_UNPACK_TILES_ATTR,
            DistributedVaeExecutor._unpack_tiles,
        )
        setattr(
            DistributedVaeExecutor,
            _ORIG_EXECUTE_ATTR,
            DistributedVaeExecutor.execute,
        )
        DistributedVaeExecutor._sync_errors = _sync_errors_all_gather
        DistributedVaeExecutor._compute_global_padding_shape = (
            _compute_global_padding_shape_all_gather
        )
        DistributedVaeExecutor._unpack_tiles = _unpack_tiles_batched
        DistributedVaeExecutor.execute = _execute_patched

        setattr(DistributedVaeExecutor, _PATCH_ATTR, True)
        logger.info(
            "Applied TPU patch: wan_opt DistributedVaeExecutor all_gather "
            "collective sync."
        )
    return True
