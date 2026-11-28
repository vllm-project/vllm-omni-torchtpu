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

"""Forked Pallas flash-attention kernels for Wan2.2 TPU optimization.

Reuses constants, ``BlockSizes``, and ``fixed_m_operands`` from
``vllm_torchtpu.kernels.diffusion_attention.kernel`` while adding:
- Explicit ``block_q`` / ``block_kv`` / ``block_kv_compute`` overrides via
  ``_splash_blocks.resolve_block_sizes``.
- Uniform fixed-m softmax via ``jax.lax.cond`` with Mosaic compiler hints and
  ``pl.CostEstimate``.
- Static block-sparse sink+window KV attention (``build_sparse_kv_indices``).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from vllm_torchtpu.kernels.diffusion_attention.kernel import (
    DEFAULT_MASK_VALUE,
    FIXED_M_RECENTER,
    FIXED_M_SAFE_BOUND,
    LOG2_E,
    NT_DIM_NUMBERS,
    NUM_SUBLANES,
    BlockSizes,
    fixed_m_operands,
)

from vllm_omni_torchtpu.wan_opt._splash_blocks import (
    detect_spec,
    resolve_block_sizes,
)


def _attention_cost_estimate(
    num_heads: int,
    q_len: int,
    kv_len: int,
    head_dim_qk: int,
    head_dim_v: int,
    bytes_accessed: int,
) -> pl.CostEstimate:
    """Builds a Pallas ``CostEstimate`` for XLA LHS overlap and XProf."""
    scores = num_heads * q_len * kv_len
    flops = 2 * scores * (head_dim_qk + head_dim_v)
    return pl.CostEstimate(
        flops=int(flops),
        transcendentals=int(scores),
        bytes_accessed=int(bytes_accessed),
    )


@functools.cache
def _splash_vmem_limit_bytes() -> int:
    """Returns Mosaic scoped-VMEM ceiling per ``pallas_call``."""
    return detect_spec().vmem_bytes


def _flash_attention_kernel(
    mk_ref: Any,
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    *refs: Any,
    mask_value: float,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    kv_seq_len: int,
    use_fixed_m: bool,
    uniform_fixed_m: bool = False,
    score_scale: float,
) -> None:
    """Hybrid, uniform fixed-m, or online-softmax flash attention over one tile."""
    float32 = jnp.float32
    m_scratch_ref, l_scratch_ref, o_scratch_ref, out_ref = refs
    _, rem = divmod(head_dim_v, NUM_SUBLANES)
    if rem != 0:
        raise NotImplementedError(
            f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}"
        )
    sv_dims = (((0,), (0,)), ((), ()))

    fixed_only = use_fixed_m and uniform_fixed_m
    if uniform_fixed_m and not use_fixed_m:
        raise ValueError("uniform_fixed_m requires use_fixed_m.")

    h = pl.program_id(0)
    j = pl.program_id(2)
    is_fixed = mk_ref[1, h] > 0.5 if use_fixed_m and not fixed_only else False

    @pl.when(j == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        if fixed_only:
            m_scratch_ref[...] = jnp.full_like(
                m_scratch_ref, mk_ref[0, h] - FIXED_M_RECENTER
            )
        elif use_fixed_m:

            @pl.when(is_fixed)
            def init_fixed():
                m_scratch_ref[...] = jnp.full_like(
                    m_scratch_ref, mk_ref[0, h] - FIXED_M_RECENTER
                )

            @pl.when(jnp.logical_not(is_fixed))
            def init_online():
                m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

        else:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

    def _update(slice_k: Any, fixed: bool, valid_len: int | None = None) -> None:
        q = q_ref[...]
        k_chunk = k_ref[slice_k, :]
        v_chunk = v_ref[slice_k, :]
        qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)

        if fixed:
            m_fix = mk_ref[0, h] - FIXED_M_RECENTER
            if valid_len is not None:
                qk = qk * score_scale
                qk_mask = lax.broadcasted_iota(jnp.int32, qk.shape, 0) < valid_len
                qk = jnp.where(qk_mask, qk, mask_value)
                v_mask = lax.broadcasted_iota(jnp.int32, v_chunk.shape, 0) < valid_len
                v_chunk = jnp.where(v_mask, v_chunk, 0)
                s_curr = jnp.exp2(qk - m_fix)
            else:
                s_curr = jnp.exp2(qk * score_scale - m_fix)
            s_cast = s_curr.astype(v_ref.dtype)
            l_curr = s_curr.sum(axis=0, keepdims=True).astype(float32)
            o_curr = lax.dot_general(
                v_chunk, s_cast, sv_dims, preferred_element_type=float32
            )
            o_scratch_ref[:] = o_scratch_ref[:] + o_curr
            l_scratch_ref[...] = l_scratch_ref[...] + l_curr
            return

        qk = qk * score_scale
        if valid_len is not None:
            qk_mask = lax.broadcasted_iota(jnp.int32, qk.shape, 0) < valid_len
            qk = jnp.where(qk_mask, qk, mask_value)
            v_mask = lax.broadcasted_iota(jnp.int32, v_chunk.shape, 0) < valid_len
            v_chunk = jnp.where(v_mask, v_chunk, 0)

        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        m_curr = qk.max(axis=0)[None, :]
        m_next = jnp.maximum(m_prev, m_curr)
        s_curr = jnp.exp2(qk - m_next[0:1])
        s_cast = s_curr.astype(v_ref.dtype)
        l_curr = s_curr.sum(axis=0, keepdims=True)
        alpha = jnp.exp2(m_prev - m_next)
        o_curr = lax.dot_general(
            v_chunk, s_cast, sv_dims, preferred_element_type=float32
        )
        o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

    def _compute_body(fixed: bool):
        def compute_body(kv_compute_index, _):
            _update(pl.ds(kv_compute_index * bkv_compute, bkv_compute), fixed)

        return compute_body

    compute_online = _compute_body(False)

    def _last_compute_body(fixed: bool):
        def last_compute_body(kv_compute_index):
            slice_len = kv_seq_len % bkv_compute
            aligned_len = (
                (slice_len + NUM_SUBLANES - 1) // NUM_SUBLANES
            ) * NUM_SUBLANES
            _update(
                pl.ds(kv_compute_index * bkv_compute, aligned_len),
                fixed,
                valid_len=slice_len if aligned_len != slice_len else None,
            )

        return last_compute_body

    assert bkv % bkv_compute == 0

    def _run_main(body_fn):
        lax.fori_loop(0, bkv // bkv_compute, body_fn, None, unroll=True)

    def _run_last_with(body_fn, last_fn):
        if kv_seq_len % bkv == 0:
            lax.fori_loop(0, bkv // bkv_compute, body_fn, None, unroll=True)
        else:
            remain = kv_seq_len % bkv
            iters = (remain + bkv_compute - 1) // bkv_compute
            if remain % bkv_compute == 0:
                lax.fori_loop(0, iters, body_fn, None, unroll=True)
            else:
                lax.fori_loop(0, iters - 1, body_fn, None, unroll=True)
                last_fn(iters - 1)

    if fixed_only:
        compute_fixed = _compute_body(True)
        last_compute_fixed = _last_compute_body(True)

        @pl.when(j != grid_width - 1)
        def body_uniform_fixed():
            _run_main(compute_fixed)

        @pl.when(j == grid_width - 1)
        def last_body_uniform_fixed():
            _run_last_with(compute_fixed, last_compute_fixed)

    elif use_fixed_m:
        compute_fixed = _compute_body(True)
        last_compute_online = _last_compute_body(False)

        @pl.when((j != grid_width - 1) & is_fixed)
        def body_fixed():
            _run_main(compute_fixed)

        @pl.when((j != grid_width - 1) & jnp.logical_not(is_fixed))
        def body_online():
            _run_main(compute_online)

        @pl.when(j == grid_width - 1)
        def last_body():
            _run_last_with(compute_online, last_compute_online)

    else:
        last_compute_online = _last_compute_body(False)

        @pl.when(j != grid_width - 1)
        def body():
            _run_main(compute_online)

        @pl.when(j == grid_width - 1)
        def last_body():
            _run_last_with(compute_online, last_compute_online)

    @pl.when(j == grid_width - 1)
    def end():
        l_sum = l_scratch_ref[...]
        l_inv = 1.0 / l_sum[0:1]
        out_ref[...] = (o_scratch_ref[...] * l_inv).astype(out_ref.dtype)


def build_sparse_kv_indices(
    grid_height: int,
    grid_width: int,
    block_q: int,
    block_kv: int,
    k_active: int,
    sink_blocks: int = 2,
) -> jax.Array:
    """Builds static ``(grid_height, k_active)`` sorted KV block indices.

    Each Q-block ``i`` attends to ``sink`` initial KV blocks ``[0 .. sink-1]``
    plus a contiguous local window of ``k_active - sink`` blocks centered on
    Q-block ``i``'s token midpoint, clamped inside ``[sink, grid_width)``.

    Args:
      grid_height: Number of Q blocks along the query sequence.
      grid_width: Total number of KV blocks along the KV sequence.
      block_q: Query block size in tokens.
      block_kv: KV block size in tokens.
      k_active: Number of active KV blocks per Q block.
      sink_blocks: Number of leading sink KV blocks kept for every Q block.

    Returns:
      ``int32`` JAX array of shape ``(grid_height, k_active)``.
    """
    if k_active <= 0 or k_active > grid_width:
        raise ValueError(f"k_active={k_active} must be in [1, grid_width={grid_width}]")
    sink = max(0, min(sink_blocks, k_active))
    win = k_active - sink
    rows: list[list[int]] = []
    for i in range(grid_height):
        center = (i * block_q + block_q // 2) // block_kv
        w_start = max(sink, min(center - win // 2, grid_width - win))
        rows.append(list(range(sink)) + list(range(w_start, w_start + win)))
    return jnp.asarray(rows, dtype=jnp.int32)


def _flash_attention_sparse_kernel(
    mk_ref: Any,
    kv_idx_ref: Any,
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    *refs: Any,
    mask_value: float,
    grid_width: int,
    k_active: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    kv_seq_len: int,
    score_scale: float,
) -> None:
    """Uniform fixed-m block-sparse flash attention over ``k_active`` KV blocks."""
    float32 = jnp.float32
    m_scratch_ref, l_scratch_ref, o_scratch_ref, out_ref = refs
    _, rem = divmod(head_dim_v, NUM_SUBLANES)
    if rem != 0:
        raise NotImplementedError(
            f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}"
        )
    sv_dims = (((0,), (0,)), ((), ()))

    h = pl.program_id(0)
    i = pl.program_id(1)
    k = pl.program_id(2)

    @pl.when(k == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        m_scratch_ref[...] = jnp.full_like(
            m_scratch_ref, mk_ref[0, h] - FIXED_M_RECENTER
        )

    def _fixed_step(slice_k: Any, valid_len: Any | None = None) -> None:
        q = q_ref[...]
        k_chunk = k_ref[slice_k, :]
        v_chunk = v_ref[slice_k, :]
        qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
        m_fix = mk_ref[0, h] - FIXED_M_RECENTER
        if valid_len is not None:
            qk = qk * score_scale
            qk_mask = lax.broadcasted_iota(jnp.int32, qk.shape, 0) < valid_len
            qk = jnp.where(qk_mask, qk, mask_value)
            v_mask = lax.broadcasted_iota(jnp.int32, v_chunk.shape, 0) < valid_len
            v_chunk = jnp.where(v_mask, v_chunk, 0)
            s_curr = jnp.exp2(qk - m_fix)
        else:
            s_curr = jnp.exp2(qk * score_scale - m_fix)
        s_cast = s_curr.astype(v_ref.dtype)
        l_curr = s_curr.sum(axis=0, keepdims=True).astype(float32)
        o_curr = lax.dot_general(
            v_chunk, s_cast, sv_dims, preferred_element_type=float32
        )
        o_scratch_ref[:] = o_scratch_ref[:] + o_curr
        l_scratch_ref[...] = l_scratch_ref[...] + l_curr

    def compute_fixed(kv_compute_index, _):
        _fixed_step(pl.ds(kv_compute_index * bkv_compute, bkv_compute))

    assert bkv % bkv_compute == 0
    num_compute = bkv // bkv_compute

    @pl.when(k != k_active - 1)
    def body_main():
        lax.fori_loop(0, num_compute, compute_fixed, None, unroll=True)

    @pl.when(k == k_active - 1)
    def last_body():
        tail_len = kv_seq_len % bkv
        if tail_len == 0:
            lax.fori_loop(0, num_compute, compute_fixed, None, unroll=True)
        else:
            valid_len = jnp.where(kv_idx_ref[i, k] == grid_width - 1, tail_len, bkv)
            for c in range(num_compute):
                row_valid = jnp.clip(valid_len - c * bkv_compute, 0, bkv_compute)
                _fixed_step(pl.ds(c * bkv_compute, bkv_compute), valid_len=row_valid)

    @pl.when(k == k_active - 1)
    def end():
        l_sum = l_scratch_ref[...]
        l_inv = 1.0 / l_sum[0:1]
        out_ref[...] = (o_scratch_ref[...] * l_inv).astype(out_ref.dtype)


def _splash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mk: jax.Array,
    block_sizes: BlockSizes,
    *,
    score_scale: float,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_fixed_m: bool = False,
    uniform_fixed_m: bool = False,
    column_heads: tuple[int, int] | None = None,
    enable_compiler_hints: bool = False,
    kv_idx: jax.Array | None = None,
) -> jax.Array:
    """Runs the forked splash kernel over ``(heads, seq, dim)`` q/k/v."""
    if column_heads is None:
        num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
        head_dim_v = v.shape[-1]
        num_kv_heads = k.shape[0]
        padded_kv_seq_len = k.shape[1]
    else:
        num_q_heads, num_kv_heads = column_heads
        padded_q_seq_len, padded_kv_seq_len = q.shape[0], k.shape[0]
        head_dim_qk = q.shape[1] // num_q_heads
        head_dim_v = v.shape[1] // num_kv_heads
    bq, bkv = block_sizes.block_q, block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute
    actual_q_seq_len = q_seq_len if q_seq_len is not None else padded_q_seq_len
    actual_kv_seq_len = kv_seq_len if kv_seq_len is not None else padded_kv_seq_len
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    def q_index_map(h, i, j, *_):
        return (h, i, 0)

    if kv_idx is not None:

        def kv_index_map(h, i, k_step, mk_ref, kv_idx_ref):
            del mk_ref
            return (h // q_heads_per_kv_head, kv_idx_ref[i, k_step], 0)

    else:

        def kv_index_map(h, i, j, *_):
            return (h // q_heads_per_kv_head, j, 0)

    def out_index_map(h, i, j, *_):
        return (h, 0, i)

    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
        pl.BlockSpec((None, bkv, head_dim_qk), kv_index_map),
        pl.BlockSpec((None, bkv, head_dim_v), kv_index_map),
    ]
    if column_heads is not None:

        def q_column_map(h, i, j, *_):
            return (i, h)

        def kv_column_map(h, i, j, *_):
            return (j, h // q_heads_per_kv_head)

        in_specs = [
            pl.BlockSpec((bq, head_dim_qk), q_column_map),
            pl.BlockSpec((bkv, head_dim_qk), kv_column_map),
            pl.BlockSpec((bkv, head_dim_v), kv_column_map),
        ]

    out_shapes = [
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
    ]
    out_specs = [
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
    ]
    first_out = len(out_shapes)
    out_shapes.append(
        jax.ShapeDtypeStruct((num_q_heads, head_dim_v, actual_q_seq_len), q.dtype)
    )
    out_specs.append(pl.BlockSpec((None, head_dim_v, bq), out_index_map))

    grid_width = (actual_kv_seq_len + bkv - 1) // bkv
    grid_height = (actual_q_seq_len + bq - 1) // bq
    k_active = int(kv_idx.shape[1]) if kv_idx is not None else grid_width
    grid = (num_q_heads, grid_height, k_active)

    compiler_kwargs: dict[str, Any] = {
        "dimension_semantics": ("parallel", "arbitrary", "arbitrary"),
        "flags": {"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
        "vmem_limit_bytes": _splash_vmem_limit_bytes(),
    }
    pallas_kwargs: dict[str, Any] = {}
    if enable_compiler_hints or kv_idx is not None:
        compiler_kwargs["disable_bounds_checks"] = True
        compiler_kwargs["skip_device_barrier"] = True
        active_q_len = grid_height * bq
        active_kv_len = k_active * bkv
        eff_kv_len = (
            min(actual_kv_seq_len, active_kv_len)
            if kv_idx is not None
            else actual_kv_seq_len
        )
        q_bytes = q.dtype.itemsize
        k_bytes = k.dtype.itemsize
        v_bytes = v.dtype.itemsize
        bytes_accessed = num_q_heads * active_q_len * (
            head_dim_qk * q_bytes + head_dim_v * v_bytes
        ) + grid_height * num_q_heads * active_kv_len * (
            head_dim_qk * k_bytes + head_dim_v * v_bytes
        )
        pallas_kwargs["cost_estimate"] = _attention_cost_estimate(
            num_q_heads,
            actual_q_seq_len,
            eff_kv_len,
            head_dim_qk,
            head_dim_v,
            bytes_accessed=bytes_accessed,
        )

    if kv_idx is not None:
        kernel_fn = functools.partial(
            _flash_attention_sparse_kernel,
            mask_value=DEFAULT_MASK_VALUE,
            grid_width=grid_width,
            k_active=k_active,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            kv_seq_len=actual_kv_seq_len,
            score_scale=score_scale,
        )
        scalar_args: tuple[jax.Array, ...] = (mk, kv_idx)
    else:
        kernel_fn = functools.partial(
            _flash_attention_kernel,
            mask_value=DEFAULT_MASK_VALUE,
            grid_width=grid_width,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            kv_seq_len=actual_kv_seq_len,
            use_fixed_m=use_fixed_m,
            uniform_fixed_m=uniform_fixed_m,
            score_scale=score_scale,
        )
        scalar_args = (mk,)

    all_out = pl.pallas_call(
        kernel_fn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_args),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(**compiler_kwargs),
        out_shape=out_shapes,
        **pallas_kwargs,
    )(*scalar_args, q, k, v)
    return all_out[first_out]


def flash_attention_bshd_fork(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    scale: float | None,
    use_fixed_m: bool = True,
    block_q: int | None = None,
    block_kv: int | None = None,
    block_kv_compute: int | None = None,
    uniform_fixed_m: bool = False,
    sparse_kv_blocks: int | None = None,
    sparse_sink_blocks: int = 2,
) -> jax.Array:
    """Device-local forked custom splash attention.

    Args:
      query: ``(B, seq_q, heads, dim)`` device-local.
      key: ``(B, seq_kv, kv_heads, dim)`` device-local.
      value: ``(B, seq_kv, kv_heads, dim_v)`` device-local.
      scale: Softmax scale; ``None`` means ``1 / sqrt(dim)``.
      use_fixed_m: Hybrid fixed-m softmax when K/V spans more than one block.
      block_q: Optional explicit query block size override.
      block_kv: Optional explicit KV block size override.
      block_kv_compute: Optional explicit KV compute sub-block override.
      uniform_fixed_m: Lossless mode enabling single-body uniform fixed-m
        via ``jax.lax.cond`` plus ``pl.CostEstimate`` and compiler hints.
      sparse_kv_blocks: Fixed number of active KV blocks per Q-block
        (``None`` keeps dense attention).
      sparse_sink_blocks: Number of initial sink KV blocks in sparse mode.

    Returns:
      Normalized output ``(B, seq_q, heads, dim_v)`` in ``query``'s dtype.
    """
    q_orig_len, num_heads, head_dim = query.shape[1:]
    kv_orig_len, num_kv_heads, head_dim_v = value.shape[1:]

    spec = detect_spec()
    fixed_tiles = use_fixed_m and kv_orig_len > spec.safe_kv_block
    resolved = resolve_block_sizes(
        q_orig_len,
        kv_orig_len,
        head_dim,
        bytes_per_element=2,
        spec=spec,
        kernel_mode="fixedm" if fixed_tiles else "online",
        override_bq=block_q,
        override_bkv=block_kv,
        override_bkvc=block_kv_compute,
    )
    bsizes = BlockSizes(
        resolved["block_q"],
        resolved["block_kv"],
        resolved["block_kv_compute"],
    )
    use_fixed_m = use_fixed_m and kv_orig_len > bsizes.block_kv
    grid_width = (kv_orig_len + bsizes.block_kv - 1) // bsizes.block_kv
    grid_height = (q_orig_len + bsizes.block_q - 1) // bsizes.block_q
    use_sparse = (
        sparse_kv_blocks is not None
        and sparse_kv_blocks > 0
        and sparse_kv_blocks < grid_width
        and kv_orig_len > spec.safe_kv_block
        and q_orig_len == kv_orig_len
        and use_fixed_m
    )
    eff_uniform = uniform_fixed_m or use_sparse

    scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    score_scale = scale_factor * LOG2_E
    q = query.astype(jnp.bfloat16)
    k = key.astype(jnp.bfloat16)
    v = value.astype(jnp.bfloat16)

    use_column_heads = (
        head_dim % spec.num_lanes == 0
        and head_dim_v % spec.num_lanes == 0
        and not use_fixed_m
    )
    if not use_column_heads:
        q, k, v = (jnp.swapaxes(x, 1, 2) for x in (q, k, v))

    if use_fixed_m:
        mk_arr = fixed_m_operands(
            q.astype(jnp.float32) * score_scale,
            k,
            gate=FIXED_M_SAFE_BOUND,
            seq_axis=2,
        )
    else:
        mk_arr = jnp.zeros((2, 1), jnp.float32)

    if use_column_heads:
        column_heads = (num_heads, num_kv_heads)
        q, k, v = (x.reshape(*x.shape[:2], -1) for x in (q, k, v))
    else:
        column_heads = None

    kv_idx_arr = (
        build_sparse_kv_indices(
            grid_height,
            grid_width,
            bsizes.block_q,
            bsizes.block_kv,
            sparse_kv_blocks,
            sink_blocks=sparse_sink_blocks,
        )
        if use_sparse and sparse_kv_blocks is not None
        else None
    )

    def _run_kernel(
        q_in: jax.Array,
        k_in: jax.Array,
        v_in: jax.Array,
        mk_in: jax.Array,
        *,
        uniform_flag: bool,
        kv_idx_in: jax.Array | None = None,
    ) -> jax.Array:
        def _kernel_3d(q_3d, k_3d, v_3d):
            return _splash_attention_forward(
                q_3d,
                k_3d,
                v_3d,
                mk_in,
                bsizes,
                score_scale=score_scale,
                q_seq_len=q_orig_len,
                kv_seq_len=kv_orig_len,
                use_fixed_m=use_fixed_m,
                uniform_fixed_m=uniform_flag,
                column_heads=column_heads,
                enable_compiler_hints=eff_uniform,
                kv_idx=kv_idx_in,
            )

        return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(q_in, k_in, v_in)

    if use_fixed_m and eff_uniform:
        all_fixed = jnp.all(mk_arr[1] > 0.5)
        res = jax.lax.cond(
            all_fixed,
            lambda q_i, k_i, v_i, m_i: _run_kernel(
                q_i, k_i, v_i, m_i, uniform_flag=True, kv_idx_in=kv_idx_arr
            ),
            lambda q_i, k_i, v_i, m_i: _run_kernel(
                q_i, k_i, v_i, m_i, uniform_flag=False, kv_idx_in=None
            ),
            q,
            k,
            v,
            mk_arr,
        )
    else:
        res = _run_kernel(q, k, v, mk_arr, uniform_flag=False, kv_idx_in=None)

    return res.transpose(0, 3, 1, 2).astype(query.dtype)
