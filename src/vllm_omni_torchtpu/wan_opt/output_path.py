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
"""W-F2: optimize post-pipeline D2H and SHM IPC output transport.

``VLLM_OMNI_WAN_FAST_OUTPUT=1`` replaces the worker-side SHM tensor packer
so large tensors are copied directly from device into a POSIX shared-memory
buffer via ``host_tensor.copy_(src)``, with ``bfloat16`` transferred as raw
``uint16`` bits (``torch_dtype_view=True``) and ``borrow_on_unpack=True``.
The engine's existing ``ipc._tensor_from_shm`` natively borrows that SHM
mapping and views it back to ``bfloat16`` without copying or widening to
``float32``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm_torchtpu.envs import env_bool
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_OMNI_WAN_FAST_OUTPUT"
_OUTPUT_PATH_PATCHED = "_tpu_wan_output_path_patched"


def _fast_pack_tensor_if_large(
    val: torch.Tensor,
    d2h_stream: torch.Stream | None = None,
    created: list[dict[str, Any]] | None = None,
    *,
    enable_registered_shm: bool = False,
) -> torch.Tensor | dict[str, Any]:
    from vllm_omni.diffusion import ipc

    view_bytes = val.nelement() * val.element_size()
    try:
        storage_bytes = val.untyped_storage().nbytes()
    except (RuntimeError, NotImplementedError):
        storage_bytes = view_bytes
    if max(view_bytes, storage_bytes) <= ipc._SHM_TENSOR_THRESHOLD:
        return val.detach().cpu() if val.device.type != "cpu" else val

    original_dtype = val.dtype
    is_bf16 = original_dtype == torch.bfloat16
    if is_bf16:
        numpy_dtype = np.dtype(np.uint16)
    else:
        try:
            numpy_dtype = torch.empty((), dtype=original_dtype).numpy().dtype
        except TypeError:
            orig_pack = getattr(ipc, "_tpu_wan_orig_pack_tensor", None)
            if orig_pack is not None:
                return orig_pack(
                    val,
                    d2h_stream,
                    created,
                    enable_registered_shm=enable_registered_shm,
                )
            raise

    src = val.detach()
    if not src.is_contiguous():
        src = src.contiguous()

    shm, shm_array, handle = ipc._new_shm_array(tuple(src.shape), numpy_dtype)
    host_tensor = None
    succeeded = False
    try:
        host_tensor = torch.from_numpy(shm_array)
        if is_bf16:
            host_tensor = host_tensor.view(torch.bfloat16)
        host_tensor.copy_(src)
        handle.update(
            {
                "__tensor_shm__": True,
                "torch_dtype": str(original_dtype),
                "borrow_on_unpack": True,
                **({"torch_dtype_view": True} if is_bf16 else {}),
            }
        )
        succeeded = True
    finally:
        del host_tensor, shm_array
        shm.close()
        if not succeeded:
            shm.unlink()

    if created is not None:
        created.append(handle)
    return handle


def apply() -> bool:
    """Install W-F2 fast direct-to-SHM output packing when enabled by env.

    Returns:
        True if the patch was installed, False if disabled or already applied.
    """
    if not env_bool(ENV, False)():
        return False

    from vllm_omni.diffusion import ipc

    if getattr(ipc._pack_tensor_if_large, _OUTPUT_PATH_PATCHED, False):
        return False

    ipc._tpu_wan_orig_pack_tensor = ipc._pack_tensor_if_large
    setattr(_fast_pack_tensor_if_large, _OUTPUT_PATH_PATCHED, True)
    ipc._pack_tensor_if_large = _fast_pack_tensor_if_large
    logger.info("Applied TPU patch: direct-to-SHM uint16/borrowed output packing.")
    return True
