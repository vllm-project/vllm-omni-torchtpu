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
"""W-D: Head-chunked, pipelined Ulysses all-to-all for Wan self-attention.

Gated on ``VLLM_OMNI_WAN_ULYSSES_CHUNKS=G`` (``G > 1``; default off). Splits
the ``H`` heads into ``G`` chunks of multiples of the Ulysses degree ``U`` and
compiles the fused ``WanSelfAttention`` forward into a single ``fullgraph=True``
TPU graph via ``torch.compile(backend="tpu", dynamic=False)`` using
``funcol.all_to_all_single`` and a DUS-free 5D ``(G, U, S_l, 1, hl, D)``
pre-permuted layout so XLA's latency-hiding scheduler overlaps chunk ``g+1``'s
forward all-to-alls and chunk ``g``'s reverse all-to-all with flash attention
within one HLO module.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from vllm_torchtpu.envs import env_int
from vllm_torchtpu.logger import init_logger

from vllm_omni_torchtpu import rope as omni_rope

logger = init_logger(__name__)

ENV = "VLLM_OMNI_WAN_ULYSSES_CHUNKS"
_ULYSSES_PIPE_PATCHED = "_tpu_ulysses_pipe_patched"
_FALLBACK_LOGGED_REASONS: set[str] = set()
_UNEVEN_WARNED_KEYS: set[tuple[int, int, int]] = set()
_COMPILED_SELF_ATTN_CACHE: dict[tuple[Any, ...], Callable[..., torch.Tensor]] = {}
_ACTIVE_LOGGED = False


def _get_weight_flipped_attr() -> str:
    """Return ``WEIGHT_FLIPPED_ATTR`` from ``linear_common`` if available."""
    try:
        linear_common = importlib.import_module(
            "vllm_torchtpu.layers.adapter.linear_common"
        )
    except ImportError:
        return "_tpu_weight_flipped"
    return str(getattr(linear_common, "WEIGHT_FLIPPED_ATTR", "_tpu_weight_flipped"))


def _import_linear_common() -> Any:
    """Import ``linear_common`` lazily with an explicit version-compat error."""
    try:
        linear_common = importlib.import_module(
            "vllm_torchtpu.layers.adapter.linear_common"
        )
    except ImportError as exc:
        raise ImportError(
            "wan_opt.ulysses_pipe requires "
            "vllm_torchtpu.layers.adapter.linear_common; upgrade "
            "vllm-torchtpu-core."
        ) from exc
    for attr in ("_get_quantized_matmul_op", "quantized_matmul"):
        if not hasattr(linear_common, attr):
            raise ImportError(
                f"vllm_torchtpu.layers.adapter.linear_common is missing {attr!r}; "
                "upgrade vllm-torchtpu-core."
            )
    return linear_common


def _log_fallback_once(reason: str) -> None:
    if reason not in _FALLBACK_LOGGED_REASONS:
        _FALLBACK_LOGGED_REASONS.add(reason)
        logger.warning(
            "wan_opt.ulysses_pipe falling back to wrapped forward: %s", reason
        )


def _ulysses_head_chunk_ranges(
    num_heads: int, ulysses_shards: int, num_chunks: int
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` head ranges, each a multiple of ``ulysses_shards``."""
    if num_chunks <= 1:
        return [(0, num_heads)]
    if num_heads % ulysses_shards != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by ulysses_shards "
            f"({ulysses_shards})."
        )
    head_groups = num_heads // ulysses_shards
    eff_chunks = min(num_chunks, head_groups)
    if head_groups % eff_chunks != 0:
        key = (num_heads, ulysses_shards, num_chunks)
        if key not in _UNEVEN_WARNED_KEYS:
            _UNEVEN_WARNED_KEYS.add(key)
            logger.warning(
                "wan_opt.ulysses_pipe: head_groups=%d (H=%d / U=%d) is not "
                "divisible by chunks=%d; uneven chunks create a double-sized "
                "tail chunk and an extra kernel shape.",
                head_groups,
                num_heads,
                ulysses_shards,
                num_chunks,
            )
    regular_groups = max(1, head_groups // eff_chunks)
    ranges: list[tuple[int, int]] = []
    start_group = 0
    for chunk_idx in range(eff_chunks):
        end_group = (
            head_groups
            if chunk_idx == eff_chunks - 1
            else min(start_group + regular_groups, head_groups)
        )
        if start_group >= end_group:
            break
        ranges.append((start_group * ulysses_shards, end_group * ulysses_shards))
        start_group = end_group
    return ranges


def _check_eligible(
    attn_mod: Any,
    hidden_states: torch.Tensor,
    rotary_emb: Any,
    attn_metadata: Any,
) -> tuple[str | None, Any]:
    """Return ``(None, sp_group)`` when eligible, else ``(reason, None)``."""
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        get_ulysses_mode,
        is_forward_context_available,
    )

    if not omni_rope.can_fuse_qk_norm_rope(attn_mod, hidden_states, rotary_emb):
        return "can_fuse_qk_norm_rope is False", None
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        return f"requires B=1 3D hidden_states, got {tuple(hidden_states.shape)}", None
    cos, sin = rotary_emb
    seq = hidden_states.shape[1]
    expected_rot = (1, seq, 1, attn_mod.head_dim // 2)
    if (
        not isinstance(cos, torch.Tensor)
        or not isinstance(sin, torch.Tensor)
        or tuple(cos.shape) != expected_rot
        or tuple(sin.shape) != expected_rot
    ):
        cos_shape = getattr(cos, "shape", None)
        sin_shape = getattr(sin, "shape", None)
        return f"unsupported rotary_emb shape {cos_shape}/{sin_shape}", None
    if not is_forward_context_available():
        return "forward context unavailable", None
    if not get_forward_context().sp_active:
        return "sp_active is False", None

    attn_layer = attn_mod.attn
    if getattr(attn_layer, "skip_sequence_parallel", False):
        return "skip_sequence_parallel is True", None
    if getattr(attn_layer, "use_ring", False):
        return "use_ring is True", None
    if getattr(attn_layer, "_has_custom_attention", False):
        return "custom_attention is active", None
    is_paged_kv = getattr(attn_layer, "is_paged_kv_active", None)
    if callable(is_paged_kv) and is_paged_kv():
        return "paged_kv is active", None
    if getattr(attn_layer, "causal", False):
        return "causal is True", None
    if (
        getattr(attn_layer, "scatter_idx", 2) != 2
        or getattr(attn_layer, "gather_idx", 1) != 1
    ):
        return "non-standard scatter/gather indices", None

    ulysses_mode = get_ulysses_mode(default="strict")
    if ulysses_mode != "strict":
        return f"ulysses_mode={ulysses_mode!r} (requires 'strict')", None

    try:
        sp_group = get_sp_group()
    except (AssertionError, RuntimeError):
        return "sp_group unavailable", None

    if sp_group.ulysses_world_size <= 1:
        return f"ulysses_world_size={sp_group.ulysses_world_size} (requires > 1)", None
    if sp_group.ring_world_size != 1:
        return f"ring_world_size={sp_group.ring_world_size} (requires 1)", None
    if sp_group.allgather_world_size != 1:
        ag_size = sp_group.allgather_world_size
        return f"allgather_world_size={ag_size} (requires 1)", None

    if attn_metadata is not None:
        if getattr(attn_metadata, "attn_mask", None) is not None:
            return "attn_mask is not None", None
        if (
            getattr(attn_metadata, "joint_query", None) is not None
            or getattr(attn_metadata, "joint_key", None) is not None
            or getattr(attn_metadata, "joint_value", None) is not None
            or getattr(attn_metadata, "joint_attn_mask", None) is not None
        ):
            return "joint tensors present", None
        extra = getattr(attn_metadata, "extra", None)
        if isinstance(extra, dict) and extra.get("gate_compress") is not None:
            return "gate_compress present", None

    kv_cache_dtype = getattr(attn_layer, "_kv_cache_dtype", None)
    if kv_cache_dtype not in (None, "float"):
        should_quant = getattr(attn_layer, "_should_apply_kv_cache_quant", None)
        if not callable(should_quant) or should_quant():
            return f"kv_cache_dtype={kv_cache_dtype!r} active", None

    if attn_mod.num_heads % sp_group.ulysses_world_size != 0:
        u_size = sp_group.ulysses_world_size
        return (
            f"num_heads={attn_mod.num_heads} not divisible by u_size={u_size}",
            None,
        )

    return None, sp_group


def _is_simple_linear(
    layer: Any,
    *,
    in_features: int,
    out_features: int | None,
    dtype: torch.dtype,
    allow_return_bias: bool = False,
) -> bool:
    """Return True when ``layer`` matches ``F.linear(x, layer.weight, layer.bias)``."""
    weight = getattr(layer, "weight", None)
    quant_method = getattr(layer, "quant_method", None)
    tp_size = getattr(layer, "tp_size", 1)
    return_bias = getattr(layer, "return_bias", False)
    skip_bias_add = getattr(layer, "skip_bias_add", False)
    flipped = getattr(layer, _get_weight_flipped_attr(), False)
    return not (
        not isinstance(weight, torch.Tensor)
        or tp_size != 1
        or (return_bias and not allow_return_bias)
        or skip_bias_add
        or flipped
        or (
            quant_method is not None
            and quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        )
        or weight.ndim != 2
        or (out_features is not None and weight.shape[0] != out_features)
        or weight.shape[1] != in_features
        or weight.dtype != dtype
    )


def _is_fp8_linear(
    layer: Any,
    *,
    in_features: int,
    out_features: int | None,
    allow_return_bias: bool = False,
) -> bool:
    """Return True when ``layer`` follows the per-channel ``_wan_fp8_active`` spec."""
    if not getattr(layer, "_wan_fp8_active", False):
        return False
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "_wan_fp8_scale", None)
    tp_size = getattr(layer, "tp_size", 1)
    return_bias = getattr(layer, "return_bias", False)
    skip_bias_add = getattr(layer, "skip_bias_add", False)
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or tp_size != 1
        or (return_bias and not allow_return_bias)
        or skip_bias_add
        or weight.ndim != 2
        or weight.shape[0] != in_features
        or (out_features is not None and weight.shape[1] != out_features)
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
    ):
        return False
    n_out = weight.shape[1]
    return scale.ndim == 1 and scale.shape == (n_out,)


def _a2a_forward_chunk(
    x: torch.Tensor,
    col0: int,
    col1: int,
    seq: int,
    u_size: int,
    chunk_width: int,
    group: dist.ProcessGroup,
    use_sync: bool,
) -> torch.Tensor:
    from vllm_omni.platforms import current_omni_platform

    send = (
        x[:, col0:col1].reshape(seq, u_size, chunk_width).transpose(0, 1).contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    if use_sync:
        current_omni_platform.synchronize()
    return recv


_FlashFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _resolve_flash_spec(
    attn_mod: Any,
    u_seq: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[tuple[Any, ...], _FlashFn] | None:
    """Resolve a graph-break-free flash-attention callable and static cache key."""
    attn_impl = attn_mod.attn.attention
    if dtype != torch.bfloat16 or head_dim != 128:
        return None
    if getattr(attn_impl, "use_sdpa", False):
        return None

    scale = float(getattr(attn_impl, "softmax_scale", head_dim**-0.5))
    if getattr(attn_impl.forward, "_tpu_wan_attn_fork_patched", False):
        from vllm_omni_torchtpu.wan_opt import attn_kernel

        if not hasattr(attn_kernel, "resolve_compiled_flash_op"):
            return None
        spec = attn_kernel.resolve_compiled_flash_op(u_seq, head_dim, scale)
        if spec is None:
            return None

        def _call_fork(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            return spec.op(
                q,
                k,
                v,
                spec.scale,
                spec.block_q,
                spec.block_kv,
                spec.block_kv_compute,
            )

        return (True, spec.cache_key), _call_fork

    flash_op = getattr(attn_impl, "_op", None)
    if flash_op is None or not callable(flash_op):
        return None

    def _call_stock(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flash_op(q, k, v, scale)

    return (False, u_seq, scale), _call_stock


def _make_pipe_core(
    *,
    seq: int,
    num_heads: int,
    head_dim: int,
    u_size: int,
    g_cnt: int,
    eps: float,
    ulysses_pg: dist.ProcessGroup,
    rope_op: Any,
    call_flash: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    qkv_fp8: bool = False,
    out_fp8: bool = False,
    quantized_matmul_fn: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., torch.Tensor]:
    """Build stateless 5D pre-permuted funcol pipelined attention function."""
    hl = num_heads // (u_size * g_cnt)
    inner_dim = num_heads * head_dim
    cw = hl * head_dim
    splits = [1] * u_size

    def _pipe_fn(
        hs: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
        qkv_w: torch.Tensor,
        qkv_b: torch.Tensor | None,
        nq_w: torch.Tensor,
        nk_w: torch.Tensor,
        out_w: torch.Tensor,
        out_b: torch.Tensor | None,
        qkv_s: torch.Tensor | None = None,
        out_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if qkv_fp8:
            assert qkv_s is not None
            assert quantized_matmul_fn is not None
            qkv = quantized_matmul_fn(hs, qkv_w, qkv_s, w8a16_max_rows=0)
            if qkv_b is not None:
                qkv = qkv + (qkv_b if qkv_b.dtype == qkv.dtype else qkv_b.to(qkv.dtype))
        else:
            qkv = F.linear(hs, qkv_w, qkv_b)
        c_exp = (
            omni_rope._repeat_pairs(cos_t.reshape(1, seq, -1))
            .expand(1, seq, head_dim)
            .reshape(seq, -1)
        )
        s_exp = (
            omni_rope._repeat_pairs(sin_t.reshape(1, seq, -1))
            .expand(1, seq, head_dim)
            .reshape(seq, -1)
        )
        q, k = rope_op(
            qkv.reshape(seq, 3 * inner_dim), nq_w, nk_w, c_exp, s_exp, eps, head_dim
        )
        q_buf = (
            q.view(seq, g_cnt, u_size, 1, hl, head_dim)
            .permute(1, 2, 0, 3, 4, 5)
            .contiguous()
        )
        k_buf = (
            k.view(seq, g_cnt, u_size, 1, hl, head_dim)
            .permute(1, 2, 0, 3, 4, 5)
            .contiguous()
        )
        v_buf = (
            qkv.view(seq, 3, g_cnt, u_size, 1, hl, head_dim)[:, 2]
            .permute(1, 2, 0, 3, 4, 5)
            .contiguous()
        )

        # Prefetch chunk g+1 forward A2As before chunk g flash attention so XLA
        # LHS overlaps ICI collectives with flash within a single HLO module.
        qkv_futs = [
            (
                funcol.all_to_all_single(q_buf[0], splits, splits, ulysses_pg),
                funcol.all_to_all_single(k_buf[0], splits, splits, ulysses_pg),
                funcol.all_to_all_single(v_buf[0], splits, splits, ulysses_pg),
            )
        ]
        o_futs: list[torch.Tensor] = []
        for g in range(g_cnt):
            qg_f, kg_f, vg_f = qkv_futs[g]
            if g + 1 < g_cnt:
                qkv_futs.append(
                    (
                        funcol.all_to_all_single(
                            q_buf[g + 1], splits, splits, ulysses_pg
                        ),
                        funcol.all_to_all_single(
                            k_buf[g + 1], splits, splits, ulysses_pg
                        ),
                        funcol.all_to_all_single(
                            v_buf[g + 1], splits, splits, ulysses_pg
                        ),
                    )
                )
            og = call_flash(
                qg_f.reshape(1, u_size * seq, hl, head_dim),
                kg_f.reshape(1, u_size * seq, hl, head_dim),
                vg_f.reshape(1, u_size * seq, hl, head_dim),
            )
            o_futs.append(
                funcol.all_to_all_single(
                    og.reshape(u_size, seq, 1, hl, head_dim),
                    splits,
                    splits,
                    ulysses_pg,
                )
            )

        o_stack = torch.stack([of.reshape(u_size, seq, cw) for of in o_futs], dim=0)
        out_cat = o_stack.permute(2, 0, 1, 3).reshape(1, seq, inner_dim)
        if out_fp8:
            assert out_s is not None
            assert quantized_matmul_fn is not None
            out = quantized_matmul_fn(out_cat, out_w, out_s, w8a16_max_rows=0)
            if out_b is not None:
                out = out + (out_b if out_b.dtype == out.dtype else out_b.to(out.dtype))
            return out
        return F.linear(out_cat, out_w, out_b)

    return _pipe_fn


def _get_or_compile_pipe_fn(
    attn_mod: Any,
    hidden_states: torch.Tensor,
    sp_group: Any,
    g_cnt: int,
) -> Callable[..., torch.Tensor] | None:
    """Return cached compiled (or in-graph uncompiled) 5D funcol fn if eligible."""
    if torch.is_grad_enabled():
        return None
    if bool(getattr(attn_mod.attn, "use_sync", False)):
        return None

    u_size = sp_group.ulysses_world_size
    num_heads = attn_mod.num_heads
    if (num_heads // u_size) % g_cnt != 0:
        return None

    seq = hidden_states.shape[1]
    head_dim = attn_mod.head_dim
    inner_dim = num_heads * head_dim
    dim = getattr(attn_mod, "dim", inner_dim)
    dtype = hidden_states.dtype

    qkv_fp8 = _is_fp8_linear(
        attn_mod.to_qkv,
        in_features=dim,
        out_features=3 * inner_dim,
        allow_return_bias=True,
    )
    if not qkv_fp8 and not _is_simple_linear(
        attn_mod.to_qkv,
        in_features=dim,
        out_features=3 * inner_dim,
        dtype=dtype,
        allow_return_bias=True,
    ):
        return None
    out_fp8 = _is_fp8_linear(
        attn_mod.to_out,
        in_features=inner_dim,
        out_features=dim,
        allow_return_bias=False,
    )
    if not out_fp8 and not _is_simple_linear(
        attn_mod.to_out,
        in_features=inner_dim,
        out_features=dim,
        dtype=dtype,
        allow_return_bias=False,
    ):
        return None

    flash_spec = _resolve_flash_spec(attn_mod, u_size * seq, head_dim, dtype)
    if flash_spec is None:
        return None
    flash_key, call_flash = flash_spec

    quantized_matmul_fn = (
        _import_linear_common().quantized_matmul if (qkv_fp8 or out_fp8) else None
    )
    ulysses_pg = sp_group.ulysses_group
    eps = float(attn_mod.norm_q.variance_epsilon)
    rope_op = omni_rope._qk_norm_rope_op()

    if torch.compiler.is_compiling() or hidden_states.device.type != "tpu":
        return _make_pipe_core(
            seq=seq,
            num_heads=num_heads,
            head_dim=head_dim,
            u_size=u_size,
            g_cnt=g_cnt,
            eps=eps,
            ulysses_pg=ulysses_pg,
            rope_op=rope_op,
            call_flash=call_flash,
            qkv_fp8=qkv_fp8,
            out_fp8=out_fp8,
            quantized_matmul_fn=quantized_matmul_fn,
        )

    qkv_scale_shape = tuple(attn_mod.to_qkv._wan_fp8_scale.shape) if qkv_fp8 else None
    out_scale_shape = tuple(attn_mod.to_out._wan_fp8_scale.shape) if out_fp8 else None
    cache_key = (
        seq,
        dim,
        num_heads,
        head_dim,
        u_size,
        g_cnt,
        getattr(attn_mod.to_qkv, "bias", None) is not None,
        getattr(attn_mod.to_out, "bias", None) is not None,
        qkv_fp8,
        out_fp8,
        qkv_scale_shape,
        out_scale_shape,
        eps,
        flash_key,
        dtype,
        str(hidden_states.device),
        id(ulysses_pg),
    )
    compiled = _COMPILED_SELF_ATTN_CACHE.get(cache_key)
    if compiled is None:
        if qkv_fp8 or out_fp8:
            # Eagerly build the pallas custom op outside torch.compile so cold-start
            # tracing never runs Lock / pallas.jax_op inside fullgraph=True.
            _import_linear_common()._get_quantized_matmul_op()
        core = _make_pipe_core(
            seq=seq,
            num_heads=num_heads,
            head_dim=head_dim,
            u_size=u_size,
            g_cnt=g_cnt,
            eps=eps,
            ulysses_pg=ulysses_pg,
            rope_op=rope_op,
            call_flash=call_flash,
            qkv_fp8=qkv_fp8,
            out_fp8=out_fp8,
            quantized_matmul_fn=quantized_matmul_fn,
        )
        compiled = torch.compile(core, backend="tpu", dynamic=False, fullgraph=True)
        _COMPILED_SELF_ATTN_CACHE[cache_key] = compiled
    return compiled


def _run_chunked_ulysses_eager(
    attn_mod: Any,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    sp_group: Any,
    num_chunks: int,
) -> torch.Tensor:
    """General eager chunked Ulysses path (supports uneven chunks and CPU/SDPA)."""
    from vllm_omni.platforms import current_omni_platform

    u_size = sp_group.ulysses_world_size
    ulysses_pg = sp_group.ulysses_group
    use_sync = bool(getattr(attn_mod.attn, "use_sync", False))
    chunk_ranges = _ulysses_head_chunk_ranges(attn_mod.num_heads, u_size, num_chunks)

    qkv, _ = attn_mod.to_qkv(hidden_states)
    batch, seq, width = qkv.shape
    head_dim = attn_mod.head_dim
    inner_dim = attn_mod.num_heads * head_dim
    cos, sin = rotary_emb

    def table(t: torch.Tensor) -> torch.Tensor:
        repeated = omni_rope._repeat_pairs(t.reshape(t.shape[0], seq, -1))
        return repeated.expand(batch, seq, head_dim).reshape(batch * seq, -1)

    query, key = omni_rope._qk_norm_rope_op()(
        qkv.reshape(batch * seq, width),
        attn_mod.norm_q.weight,
        attn_mod.norm_k.weight,
        table(cos),
        table(sin),
        attn_mod.norm_q.variance_epsilon,
        head_dim,
    )
    value = qkv[0, :, 2 * inner_dim :]

    attn_impl = attn_mod.attn.attention
    out_chunks: list[torch.Tensor] = []

    for h0, h1 in chunk_ranges:
        hl = (h1 - h0) // u_size
        col0 = h0 * head_dim
        col1 = h1 * head_dim
        chunk_width = hl * head_dim

        q_g = _a2a_forward_chunk(
            query, col0, col1, seq, u_size, chunk_width, ulysses_pg, use_sync
        ).reshape(1, u_size * seq, hl, head_dim)
        k_g = _a2a_forward_chunk(
            key, col0, col1, seq, u_size, chunk_width, ulysses_pg, use_sync
        ).reshape(1, u_size * seq, hl, head_dim)
        v_g = _a2a_forward_chunk(
            value, col0, col1, seq, u_size, chunk_width, ulysses_pg, use_sync
        ).reshape(1, u_size * seq, hl, head_dim)

        o_g = attn_impl.forward(q_g, k_g, v_g, None)

        o_send = o_g.reshape(u_size, seq, chunk_width).contiguous()
        o_local = torch.empty_like(o_send)
        dist.all_to_all_single(o_local, o_send, group=ulysses_pg)
        if use_sync:
            current_omni_platform.synchronize()

        out_chunks.append(o_local.transpose(0, 1).reshape(1, seq, (h1 - h0) * head_dim))

    out_cat = torch.cat(out_chunks, dim=2)
    return attn_mod.dropout(attn_mod.to_out(out_cat))


def _run_chunked_ulysses_self_attention(
    attn_mod: Any,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    sp_group: Any,
    num_chunks: int,
) -> torch.Tensor:
    """Execute WanSelfAttention with head-chunked Ulysses A2A."""
    global _ACTIVE_LOGGED
    u_size = sp_group.ulysses_world_size
    chunk_ranges = _ulysses_head_chunk_ranges(attn_mod.num_heads, u_size, num_chunks)
    pipe_fn = _get_or_compile_pipe_fn(
        attn_mod,
        hidden_states,
        sp_group,
        len(chunk_ranges),
    )
    if not _ACTIVE_LOGGED:
        _ACTIVE_LOGGED = True
        logger.info(
            "wan_opt.ulysses_pipe active: mode=%s, ulysses=%d, chunks=%d, ranges=%s.",
            "compiled_5d" if pipe_fn is not None else "eager",
            u_size,
            len(chunk_ranges),
            tuple(chunk_ranges),
        )

    if pipe_fn is not None:
        cos, sin = rotary_emb
        qkv_fp8 = bool(getattr(attn_mod.to_qkv, "_wan_fp8_active", False))
        out_fp8 = bool(getattr(attn_mod.to_out, "_wan_fp8_active", False))
        extra_args = (
            (
                attn_mod.to_qkv._wan_fp8_scale if qkv_fp8 else None,
                attn_mod.to_out._wan_fp8_scale if out_fp8 else None,
            )
            if (qkv_fp8 or out_fp8)
            else ()
        )
        out = pipe_fn(
            hidden_states,
            cos,
            sin,
            attn_mod.to_qkv.weight,
            getattr(attn_mod.to_qkv, "bias", None),
            attn_mod.norm_q.weight,
            attn_mod.norm_k.weight,
            attn_mod.to_out.weight,
            getattr(attn_mod.to_out, "bias", None),
            *extra_args,
        )
        return attn_mod.dropout(out)

    return _run_chunked_ulysses_eager(
        attn_mod,
        hidden_states,
        rotary_emb,
        sp_group,
        num_chunks,
    )


def apply() -> bool:
    """Install the head-chunked Ulysses self-attention wrapper when gated on.

    Returns:
        True if the wrapper was installed, False when disabled or already active.
    """
    num_chunks = env_int(ENV, 1)()
    if num_chunks <= 1:
        return False

    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanSelfAttention,
    )

    if getattr(WanSelfAttention.forward, _ULYSSES_PIPE_PATCHED, False):
        return False

    wrapped_forward = WanSelfAttention.forward

    def _chunked_ulysses_self_attention_forward(
        self: Any,
        hidden_states: torch.Tensor,
        rotary_emb: Any = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        reason, sp_group = _check_eligible(
            self,
            hidden_states,
            rotary_emb,
            attn_metadata,
        )
        if reason is not None:
            _log_fallback_once(reason)
            return wrapped_forward(self, hidden_states, rotary_emb, attn_metadata)
        return _run_chunked_ulysses_self_attention(
            self,
            hidden_states,
            rotary_emb,
            sp_group,
            num_chunks,
        )

    setattr(_chunked_ulysses_self_attention_forward, _ULYSSES_PIPE_PATCHED, True)
    WanSelfAttention.forward = _chunked_ulysses_self_attention_forward
    logger.info(
        "Applied TPU patch: pipelined head-chunked Ulysses self-attention (chunks=%d).",
        num_chunks,
    )
    return True
