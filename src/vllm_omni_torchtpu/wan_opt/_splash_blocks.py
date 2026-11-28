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

"""Block-size resolution with explicit overrides for Wan2.2 splash attention.

Reuses ``TpuSpec``, ``V7_SPEC``, ``autotune_block_sizes``, ``detect_spec``, and
``vmem_footprint_bytes`` from
``vllm_torchtpu.kernels.diffusion_attention.block_sizes`` and adds
``resolve_block_sizes()`` so ``BQ``, ``BKV``, and ``BKVC`` overrides can exceed
the default 0.6 VMEM fixed-m cap up to the 63 MiB physical VMEM limit.
"""

from __future__ import annotations

import functools

from vllm_torchtpu.kernels.diffusion_attention.block_sizes import (
    V7_SPEC,
    TpuSpec,
    autotune_block_sizes,
    detect_spec,
    vmem_footprint_bytes,
)
from vllm_torchtpu.logger import init_logger

__all__ = [
    "TpuSpec",
    "V7_SPEC",
    "autotune_block_sizes",
    "detect_spec",
    "resolve_block_sizes",
    "vmem_footprint_bytes",
]

_logger = init_logger(__name__)

_VMEM_RESERVE_BYTES = 2**20
_MAX_VMEM_FOOTPRINT_BYTES = 63 * 2**20


def _validate_block_sizes(
    bq: int,
    bkv: int,
    bkvc: int,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    spec: TpuSpec,
) -> str | None:
    """Returns an error reason if ``(bq, bkv, bkvc)`` violates kernel constraints.

    Args:
      bq: Clamped query block size.
      bkv: Clamped KV block size.
      bkvc: Clamped KV compute sub-block size.
      q_seq_len: Query sequence length.
      kv_seq_len: Key/value sequence length.
      head_dim: Per-head feature dimension.
      spec: Hardware specification for lane alignment and VMEM budget.

    Returns:
      Human-readable error string if invalid, or ``None`` when valid.
    """
    if bq <= 0 or bkv <= 0 or bkvc <= 0:
        return f"non-positive block size ({bq=}, {bkv=}, {bkvc=})"
    if bkv % bkvc != 0:
        return f"bkv ({bkv}) must be divisible by bkvc ({bkvc})"
    if bq < q_seq_len and bq % spec.num_lanes != 0:
        return f"bq ({bq}) must be a multiple of {spec.num_lanes} when bq < {q_seq_len}"
    if bkv < kv_seq_len and bkv % 16 != 0:
        return f"bkv ({bkv}) must be a multiple of 16 when bkv < {kv_seq_len}"
    if bkvc < kv_seq_len and bkvc % 16 != 0:
        return f"bkvc ({bkvc}) must be a multiple of 16 when bkvc < {kv_seq_len}"
    vmem_hard_limit = min(
        _MAX_VMEM_FOOTPRINT_BYTES, spec.vmem_bytes - _VMEM_RESERVE_BYTES
    )
    footprint = vmem_footprint_bytes(bq, bkvc, head_dim, bkv)
    if footprint > vmem_hard_limit:
        return (
            f"vmem_footprint_bytes({bq}, {bkvc}, {head_dim}, {bkv})={footprint} "
            f"exceeds limit {vmem_hard_limit}"
        )
    return None


@functools.cache
def resolve_block_sizes(
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int = 128,
    *,
    bytes_per_element: int = 2,
    spec: TpuSpec | None = None,
    kernel_mode: str = "online",
    override_bq: int | None = None,
    override_bkv: int | None = None,
    override_bkvc: int | None = None,
) -> dict[str, int]:
    """Resolves clamped block sizes, honoring valid explicit overrides.

    Explicit ``override_bq`` / ``override_bkv`` / ``override_bkvc`` values may
    exceed the autotuner's 0.6 VMEM fixed-m heuristic (e.g. ``BQ=4736`` on v7x)
    as long as they pass alignment and the 63 MiB VMEM footprint ceiling.
    Invalid overrides log a one-time warning and fall back to
    ``autotune_block_sizes``.

    Args:
      q_seq_len: Query sequence length.
      kv_seq_len: Key/value sequence length.
      head_dim: Per-head feature dimension.
      bytes_per_element: Activation element size in bytes.
      spec: Optional hardware spec (auto-detected when omitted).
      kernel_mode: Autotuner kernel mode (``"online"`` or ``"fixedm"``).
      override_bq: Optional explicit query block size override.
      override_bkv: Optional explicit KV block size override.
      override_bkvc: Optional explicit KV compute sub-block override.

    Returns:
      Mapping with ``"block_q"``, ``"block_kv"``, and ``"block_kv_compute"``.
    """
    eff_spec = spec if spec is not None else detect_spec()
    autotuned = autotune_block_sizes(
        q_seq_len,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
        spec=eff_spec,
        kv_seq_len=kv_seq_len,
        kernel_mode=kernel_mode,
    )
    auto_bq = min(autotuned["block_q"], q_seq_len)
    auto_bkv = min(autotuned["block_kv"], kv_seq_len)
    auto_bkvc = min(autotuned["block_kv_compute"], kv_seq_len)

    if override_bq is None and override_bkv is None and override_bkvc is None:
        return {
            "block_q": auto_bq,
            "block_kv": auto_bkv,
            "block_kv_compute": auto_bkvc,
        }

    req_bq = override_bq if override_bq is not None else autotuned["block_q"]
    req_bkv = override_bkv if override_bkv is not None else autotuned["block_kv"]
    req_bkvc = override_bkvc if override_bkvc is not None else req_bkv

    bq = min(req_bq, q_seq_len)
    bkv = min(req_bkv, kv_seq_len)
    bkvc = min(req_bkvc, kv_seq_len)

    reason = _validate_block_sizes(
        bq, bkv, bkvc, q_seq_len, kv_seq_len, head_dim, eff_spec
    )
    if reason is not None:
        _logger.warning_once(
            "wan_opt.attn_kernel: invalid block-size override "
            "(bq=%s, bkv=%s, bkvc=%s for q=%d, kv=%d): %s; "
            "falling back to autotuned (%d, %d, %d).",
            override_bq,
            override_bkv,
            override_bkvc,
            q_seq_len,
            kv_seq_len,
            reason,
            auto_bq,
            auto_bkv,
            auto_bkvc,
        )
        return {
            "block_q": auto_bq,
            "block_kv": auto_bkv,
            "block_kv_compute": auto_bkvc,
        }

    return {
        "block_q": bq,
        "block_kv": bkv,
        "block_kv_compute": bkvc,
    }
