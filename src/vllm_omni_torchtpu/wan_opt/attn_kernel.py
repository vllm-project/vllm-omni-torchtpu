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

"""Splash-attention kernel fork for Wan2.2 TPU optimization.

Gated by ``VLLM_OMNI_WAN_ATTN_FORK=1``. Replaces ``TpuFlashAttentionImpl.forward``
to invoke ``pallas::wan_flash_attention_fork`` while preserving the upstream
GQA validation and SDPA fallback checks.

Optional knobs (all off by default; only active when
``VLLM_OMNI_WAN_ATTN_FORK=1``):
- ``VLLM_OMNI_WAN_BQ`` / ``VLLM_OMNI_WAN_BKV`` / ``VLLM_OMNI_WAN_BKVC``:
  explicit block-size overrides validated up to 63 MiB VMEM.
- ``VLLM_OMNI_WAN_SPLASH_UNIFORM=1``: uniform fixed-m fast path with Mosaic
  compiler hints and ``CostEstimate``.
- ``VLLM_OMNI_WAN_SPARSE_ATTN=1``: block-sparse sink+window attention after
  ``VLLM_OMNI_WAN_SPARSE_WARMUP_STEPS`` (default 20), keeping
  ``VLLM_OMNI_WAN_SPARSE_KV_BLOCKS`` (default 22) blocks including
  ``VLLM_OMNI_WAN_SPARSE_SINK_BLOCKS`` (default 2) leading sink blocks.
"""

import dataclasses
import functools
from collections.abc import Callable
from functools import cache as run_once
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.envs import (
    env_bool,
    env_nonnegative_int,
    env_positive_int_or_none,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)

ENV = "VLLM_OMNI_WAN_ATTN_FORK"
ENV_BQ = "VLLM_OMNI_WAN_BQ"
ENV_BKV = "VLLM_OMNI_WAN_BKV"
ENV_BKVC = "VLLM_OMNI_WAN_BKVC"
ENV_SPLASH_UNIFORM = "VLLM_OMNI_WAN_SPLASH_UNIFORM"
ENV_SPARSE_ATTN = "VLLM_OMNI_WAN_SPARSE_ATTN"
ENV_SPARSE_KV_BLOCKS = "VLLM_OMNI_WAN_SPARSE_KV_BLOCKS"
ENV_SPARSE_SINK_BLOCKS = "VLLM_OMNI_WAN_SPARSE_SINK_BLOCKS"
ENV_SPARSE_WARMUP_STEPS = "VLLM_OMNI_WAN_SPARSE_WARMUP_STEPS"

_DEFAULT_SPARSE_KV_BLOCKS = 22
_DEFAULT_SPARSE_SINK_BLOCKS = 2
_DEFAULT_SPARSE_WARMUP_STEPS = 20

_ATTN_FORK_PATCHED = "_tpu_wan_attn_fork_patched"


@dataclasses.dataclass(frozen=True)
class CompiledFlashSpec:
    """Resolved flash-attention custom op and static tiling for compiled graphs.

    Attributes:
      op: 7-arg PyTorch custom op ``op(q, k, v, scale, bq, bkv, bkvc)``.
      block_q: Resolved query block size.
      block_kv: Resolved KV block size.
      block_kv_compute: Resolved KV compute sub-block size.
      scale: Softmax scale factor.
      cache_key: Static key identifying ``(id(op), bq, bkv, bkvc, scale)``.
    """

    op: Callable[..., Any]
    block_q: int
    block_kv: int
    block_kv_compute: int
    scale: float
    cache_key: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class _ForkConfig:
    bq: int | None = None
    bkv: int | None = None
    bkvc: int | None = None
    uniform_fixed_m: bool = False
    sparse_attn: bool = False
    sparse_kv_blocks: int = _DEFAULT_SPARSE_KV_BLOCKS
    sparse_sink_blocks: int = _DEFAULT_SPARSE_SINK_BLOCKS
    sparse_warmup_steps: int = _DEFAULT_SPARSE_WARMUP_STEPS


_ACTIVE_CONFIG: _ForkConfig | None = None


def _config_from_env() -> _ForkConfig:
    return _ForkConfig(
        bq=env_positive_int_or_none(ENV_BQ)(),
        bkv=env_positive_int_or_none(ENV_BKV)(),
        bkvc=env_positive_int_or_none(ENV_BKVC)(),
        uniform_fixed_m=env_bool(ENV_SPLASH_UNIFORM, default=False)(),
        sparse_attn=env_bool(ENV_SPARSE_ATTN, default=False)(),
        sparse_kv_blocks=env_nonnegative_int(
            ENV_SPARSE_KV_BLOCKS, _DEFAULT_SPARSE_KV_BLOCKS
        )(),
        sparse_sink_blocks=env_nonnegative_int(
            ENV_SPARSE_SINK_BLOCKS, _DEFAULT_SPARSE_SINK_BLOCKS
        )(),
        sparse_warmup_steps=env_nonnegative_int(
            ENV_SPARSE_WARMUP_STEPS, _DEFAULT_SPARSE_WARMUP_STEPS
        )(),
    )


def _current_config() -> _ForkConfig:
    return _ACTIVE_CONFIG if _ACTIVE_CONFIG is not None else _config_from_env()


def _current_denoise_step_idx() -> int | None:
    """Reads the active request's integer denoise step index if available."""
    try:
        from vllm_omni.diffusion.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except ImportError:
        return None
    if not is_forward_context_available():
        return None
    return getattr(get_forward_context(), "denoise_step_idx", None)


def _is_sparse_step(cfg: _ForkConfig) -> bool:
    if not cfg.sparse_attn or cfg.sparse_kv_blocks <= 0:
        return False
    if cfg.sparse_warmup_steps <= 0:
        return True
    step_idx = _current_denoise_step_idx()
    return step_idx is not None and step_idx >= cfg.sparse_warmup_steps


def _kernel_mode_key(cfg: _ForkConfig) -> tuple[bool, int | None, int]:
    sparse_k = cfg.sparse_kv_blocks if _is_sparse_step(cfg) else None
    sparse_sink = cfg.sparse_sink_blocks if sparse_k is not None else 0
    return (cfg.uniform_fixed_m, sparse_k, sparse_sink)


def _op_name_suffix(
    uniform_fixed_m: bool,
    sparse_kv_blocks: int | None = None,
    sparse_sink_blocks: int = 0,
) -> str:
    if not uniform_fixed_m and sparse_kv_blocks is None:
        return ""
    parts = [f"u{int(uniform_fixed_m)}"]
    if sparse_kv_blocks is not None:
        parts.append(f"sk{sparse_kv_blocks}_ss{sparse_sink_blocks}")
    return "_" + "_".join(parts)


@run_once
def _build_fork_op(
    uniform_fixed_m: bool,
    sparse_kv_blocks: int | None,
    sparse_sink_blocks: int,
) -> Callable[..., Any]:
    """Builds a config-keyed torch custom op wrapping the forked Pallas kernel."""
    import jax
    import torch
    from torch_tpu._internal import pallas

    from vllm_omni_torchtpu.wan_opt._splash_fork import flash_attention_bshd_fork

    suffix = _op_name_suffix(
        uniform_fixed_m,
        sparse_kv_blocks,
        sparse_sink_blocks,
    )

    def wan_flash_attention_fork(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        scale: float,
        block_q: int,
        block_kv: int,
        block_kv_compute: int,
    ) -> jax.Array:
        return flash_attention_bshd_fork(
            q,
            k,
            v,
            scale=scale,
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            uniform_fixed_m=uniform_fixed_m,
            sparse_kv_blocks=sparse_kv_blocks,
            sparse_sink_blocks=sparse_sink_blocks,
        )

    op = pallas.jax_op(
        f"pallas::wan_flash_attention_fork{suffix}", wan_flash_attention_fork
    )

    def _fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        block_q: int,
        block_kv: int,
        block_kv_compute: int,
    ) -> torch.Tensor:
        return q.new_empty((*q.shape[:3], v.shape[-1]))

    op.register_fake(_fake)
    return op


def _flash_attention_fork_op() -> Callable[..., Any]:
    """Returns the active torch custom op for the current splash config."""
    return _build_fork_op(*_kernel_mode_key(_current_config()))


@functools.cache
def _resolve_for_call(
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    override_bq: int | None,
    override_bkv: int | None,
    override_bkvc: int | None,
) -> tuple[int, int, int]:
    """Resolves and caches ``(block_q, block_kv, block_kv_compute)``."""
    from vllm_omni_torchtpu.wan_opt._splash_blocks import (
        detect_spec,
        resolve_block_sizes,
    )

    spec = detect_spec()
    fixed_tiles = kv_seq_len > spec.safe_kv_block
    bs = resolve_block_sizes(
        q_seq_len,
        kv_seq_len,
        head_dim,
        bytes_per_element=2,
        spec=spec,
        kernel_mode="fixedm" if fixed_tiles else "online",
        override_bq=override_bq if fixed_tiles else None,
        override_bkv=override_bkv if fixed_tiles else None,
        override_bkvc=override_bkvc if fixed_tiles else None,
    )
    return bs["block_q"], bs["block_kv"], bs["block_kv_compute"]


def resolve_compiled_flash_op(
    seq_len: int,
    head_dim: int,
    scale: float,
) -> CompiledFlashSpec | None:
    """Resolves the active forked flash op and block sizes for compiled graphs.

    Returns ``CompiledFlashSpec`` when ``VLLM_OMNI_WAN_ATTN_FORK`` is applied
    (selecting the dense or sparse op for the current ``denoise_step_idx``),
    or ``None`` when disabled or not yet applied.

    Args:
      seq_len: Self-attention sequence length seen by the local kernel.
      head_dim: Per-head feature dimension.
      scale: Softmax scale factor.

    Returns:
      ``CompiledFlashSpec`` when the attention fork is active, else ``None``.
    """
    if not env_bool(ENV, default=False)():
        return None
    if _ACTIVE_CONFIG is None:
        from vllm_omni_torchtpu.attention import TpuFlashAttentionImpl

        if not getattr(TpuFlashAttentionImpl.forward, _ATTN_FORK_PATCHED, False):
            return None

    cfg = _current_config()
    op = _flash_attention_fork_op()
    bq, bkv, bkvc = _resolve_for_call(
        seq_len, seq_len, head_dim, cfg.bq, cfg.bkv, cfg.bkvc
    )
    eff_scale = float(scale)
    return CompiledFlashSpec(
        op=op,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkvc,
        scale=eff_scale,
        cache_key=(id(op), bq, bkv, bkvc, eff_scale),
    )


def apply() -> bool:
    """Installs the forked splash attention op when ``VLLM_OMNI_WAN_ATTN_FORK=1``.

    Returns:
      ``True`` if the fork patch is active, ``False`` when disabled by env.
    """
    global _ACTIVE_CONFIG

    if not env_bool(ENV, default=False)():
        return False

    cfg = _config_from_env()
    _ACTIVE_CONFIG = cfg
    # Warm dense and (if enabled) sparse custom ops outside compiled regions.
    _build_fork_op(cfg.uniform_fixed_m, None, 0)
    if cfg.sparse_attn and cfg.sparse_kv_blocks > 0:
        _build_fork_op(
            cfg.uniform_fixed_m,
            cfg.sparse_kv_blocks,
            cfg.sparse_sink_blocks,
        )

    from vllm_omni_torchtpu.attention import TpuFlashAttentionImpl, TpuSDPAImpl

    if getattr(TpuFlashAttentionImpl.forward, _ATTN_FORK_PATCHED, False):
        return True

    orig_forward = getattr(
        TpuFlashAttentionImpl,
        "_tpu_wan_attn_orig_forward",
        TpuFlashAttentionImpl.forward,
    )
    TpuFlashAttentionImpl._tpu_wan_attn_orig_forward = orig_forward

    def _forked_flash_forward(
        self: Any,
        query: "torch.Tensor",
        key: "torch.Tensor",
        value: "torch.Tensor",
        attn_metadata: Any | None = None,
    ) -> "torch.Tensor":
        if query.shape[2] % key.shape[2] != 0:
            raise ValueError(
                f"num_heads ({query.shape[2]}) must be a multiple of "
                f"num_kv_heads ({key.shape[2]})"
            )
        if self._needs_sdpa_fallback(query, key, attn_metadata):
            return TpuSDPAImpl.forward(self, query, key, value, attn_metadata)

        active_cfg = _current_config()
        bq, bkv, bkvc = _resolve_for_call(
            query.shape[1],
            key.shape[1],
            query.shape[3],
            active_cfg.bq,
            active_cfg.bkv,
            active_cfg.bkvc,
        )
        return _flash_attention_fork_op()(
            query,
            key,
            value,
            self.softmax_scale,
            bq,
            bkv,
            bkvc,
        )

    setattr(_forked_flash_forward, _ATTN_FORK_PATCHED, True)
    TpuFlashAttentionImpl.forward = _forked_flash_forward
    logger.info(
        "Applied TPU patch: wan_opt splash attention fork "
        "(bq=%s, bkv=%s, bkvc=%s, uniform=%s, "
        "sparse=%s, sparse_kv=%s, sparse_sink=%s, sparse_warmup=%s).",
        cfg.bq,
        cfg.bkv,
        cfg.bkvc,
        cfg.uniform_fixed_m,
        cfg.sparse_attn,
        cfg.sparse_kv_blocks,
        cfg.sparse_sink_blocks,
        cfg.sparse_warmup_steps,
    )
    return True
