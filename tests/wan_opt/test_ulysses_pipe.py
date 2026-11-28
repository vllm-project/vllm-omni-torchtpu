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
"""Tests for W-D: head-chunked pipelined Ulysses self-attention."""

from __future__ import annotations

import os
import socket
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch_tpu._internal.utils import hardware
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanSelfAttention
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

from vllm_omni_torchtpu import rope as omni_rope
from vllm_omni_torchtpu.patches import apply_omni_model_specific_patches
from vllm_omni_torchtpu.wan_opt import ulysses_pipe


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pair_swap_matrix(
    dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    p = torch.zeros(dim, dim, dtype=torch.float32)
    even = torch.arange(0, dim, 2)
    p[even + 1, even] = -1.0
    p[even, even + 1] = 1.0
    return p.to(device=device, dtype=dtype)


def _reference_qk_norm_rope(qkv, wq, wk, cos, sin, eps, head_dim):
    dim = qkv.shape[1] // 3
    swap = _pair_swap_matrix(head_dim, qkv.dtype, qkv.device)
    outs = []
    for j, w in ((0, wq), (1, wk)):
        x32 = qkv[:, j * dim : (j + 1) * dim].float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        y = (w.float() * (x32 * rms)).to(qkv.dtype)
        y = y.unflatten(1, (-1, head_dim))
        out = (
            y.float() * cos[:, None].float() + (y @ swap).float() * sin[:, None].float()
        ).to(qkv.dtype)
        outs.append(out.flatten(1))
    return tuple(outs)


def _sdpa_impl_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any = None,
) -> torch.Tensor:
    del attn_metadata
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()


def _make_cpu_wan_self_attention(
    dim: int,
    heads: int,
    head_dim: int,
    *,
    with_bias: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> WanSelfAttention:
    gen = torch.Generator().manual_seed(seed)
    attn = object.__new__(WanSelfAttention)
    torch.nn.Module.__init__(attn)
    attn.dim = dim
    attn.num_heads = heads
    attn.num_kv_heads = heads
    attn.head_dim = head_dim
    inner_dim = heads * head_dim

    qkv_lin = torch.nn.Linear(dim, 3 * inner_dim, bias=with_bias)
    with torch.no_grad():
        qkv_lin.weight.copy_(torch.randn(3 * inner_dim, dim, generator=gen) * 0.02)
        if with_bias:
            qkv_lin.bias.copy_(torch.randn(3 * inner_dim, generator=gen) * 0.02)
    qkv_lin = qkv_lin.to(dtype)
    attn.to_qkv = lambda x: (qkv_lin(x), None)

    attn.norm_q = RMSNorm(inner_dim, eps=1e-6).to(dtype)
    attn.norm_k = RMSNorm(inner_dim, eps=1e-6).to(dtype)
    with torch.no_grad():
        attn.norm_q.weight.copy_(torch.linspace(0.7, 1.3, inner_dim).to(dtype))
        attn.norm_k.weight.copy_(torch.linspace(1.3, 0.7, inner_dim).to(dtype))

    to_out = torch.nn.Linear(inner_dim, dim, bias=with_bias)
    with torch.no_grad():
        to_out.weight.copy_(torch.randn(dim, inner_dim, generator=gen) * 0.02)
        if with_bias:
            to_out.bias.copy_(torch.randn(dim, generator=gen) * 0.02)
    attn.to_out = to_out.to(dtype)
    attn.dropout = torch.nn.Identity()
    attn.rotary_embedding = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    attn.to_gate_compress = None

    impl = SimpleNamespace(forward=_sdpa_impl_forward)
    attn.attn = SimpleNamespace(
        attention=impl,
        skip_sequence_parallel=False,
        use_ring=False,
        _has_custom_attention=False,
        causal=False,
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        _kv_cache_dtype=None,
        is_paged_kv_active=lambda: False,
    )
    return attn


# --------------------------------------------------------------------------- #
# Unit tests (single process, CPU)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "num_heads,u_size,num_chunks,expected",
    [
        (40, 4, 1, [(0, 40)]),
        (40, 4, 2, [(0, 20), (20, 40)]),
        (40, 4, 4, [(0, 8), (8, 16), (16, 24), (24, 40)]),
        (40, 4, 5, [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]),
        (
            40,
            4,
            10,
            [(4 * i, 4 * (i + 1)) for i in range(10)],
        ),
        (
            40,
            4,
            20,
            [(4 * i, 4 * (i + 1)) for i in range(10)],
        ),
    ],
)
def test_head_chunk_ranges(num_heads, u_size, num_chunks, expected, monkeypatch):
    ulysses_pipe._UNEVEN_WARNED_KEYS.clear()
    warns = []
    monkeypatch.setattr(
        ulysses_pipe.logger, "warning", lambda msg, *a: warns.append(msg % a)
    )
    ranges = ulysses_pipe._ulysses_head_chunk_ranges(num_heads, u_size, num_chunks)
    assert ranges == expected
    assert ranges[0][0] == 0
    assert ranges[-1][1] == num_heads
    for (s0, e0), (s1, _) in zip(ranges, ranges[1:]):
        assert e0 == s1
    for s, e in ranges:
        assert (e - s) % u_size == 0
    head_groups = num_heads // u_size
    if num_chunks > 1 and head_groups % min(num_chunks, head_groups) != 0:
        assert len(warns) == 1
        # Warning is logged once per (H, U, G).
        ulysses_pipe._ulysses_head_chunk_ranges(num_heads, u_size, num_chunks)
        assert len(warns) == 1
    else:
        assert warns == []


def test_apply_env_gate_and_idempotency(monkeypatch):
    apply_omni_model_specific_patches()
    orig_forward = WanSelfAttention.forward
    try:
        monkeypatch.delenv(ulysses_pipe.ENV, raising=False)
        assert ulysses_pipe.apply() is False
        assert WanSelfAttention.forward is orig_forward

        monkeypatch.setenv(ulysses_pipe.ENV, "1")
        assert ulysses_pipe.apply() is False
        assert WanSelfAttention.forward is orig_forward

        monkeypatch.setenv(ulysses_pipe.ENV, "4")
        assert ulysses_pipe.apply() is True
        patched_forward = WanSelfAttention.forward
        assert patched_forward is not orig_forward
        assert getattr(patched_forward, ulysses_pipe._ULYSSES_PIPE_PATCHED, False)
        assert hasattr(WanSelfAttention, "_tpu_orig_forward")

        # Second call is a no-op.
        assert ulysses_pipe.apply() is False
        assert WanSelfAttention.forward is patched_forward
    finally:
        WanSelfAttention.forward = orig_forward


def test_fallback_guards_and_once_logging(monkeypatch):
    apply_omni_model_specific_patches()
    orig_forward = WanSelfAttention.forward
    try:
        monkeypatch.setenv(ulysses_pipe.ENV, "2")
        assert ulysses_pipe.apply() is True

        # Replace wrapped forward spy by installing on fresh wrapper target.
        WanSelfAttention.forward = orig_forward
        spy_calls = []

        def fake_wrapped(self, hidden_states, rotary_emb=None, attn_metadata=None):
            spy_calls.append((hidden_states.shape, attn_metadata))
            return torch.zeros_like(hidden_states)

        WanSelfAttention.forward = fake_wrapped
        assert ulysses_pipe.apply() is True

        ulysses_pipe._FALLBACK_LOGGED_REASONS.clear()
        logged = []
        monkeypatch.setattr(
            ulysses_pipe.logger, "warning", lambda msg, *a: logged.append(msg % a)
        )

        attn = _make_cpu_wan_self_attention(dim=256, heads=4, head_dim=128)
        cos = torch.ones(1, 8, 1, 64, dtype=torch.bfloat16)
        sin = torch.zeros(1, 8, 1, 64, dtype=torch.bfloat16)
        x_b2 = torch.randn(2, 8, 256, dtype=torch.bfloat16)
        x_f32 = torch.randn(1, 8, 256, dtype=torch.float32)
        x_ok = torch.randn(1, 8, 256, dtype=torch.bfloat16)

        sp_group = SimpleNamespace(
            ulysses_world_size=4,
            ring_world_size=1,
            allgather_world_size=1,
            ulysses_group=None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.distributed.parallel_state.get_sp_group",
            lambda: sp_group,
        )

        # 1. Float32 falls back (can_fuse_qk_norm_rope=False) and logs once.
        attn(x_f32, (cos.float(), sin.float()))
        attn(x_f32, (cos.float(), sin.float()))
        assert len(spy_calls) == 2
        assert len(logged) == 1

        # 2. Batch != 1 falls back.
        attn(x_b2, (cos, sin))
        assert len(spy_calls) == 3
        assert len(logged) == 2

        # 3. No forward context falls back.
        attn(x_ok, (cos, sin))
        assert len(spy_calls) == 4
        assert len(logged) == 3

        # 4. Inside forward context: test ulysses_mode, ring, mask, joint, sp_group.
        od_cfg_strict = SimpleNamespace(
            parallel_config=SimpleNamespace(
                sequence_parallel_size=4, ulysses_mode="strict"
            )
        )
        od_cfg_uaa = SimpleNamespace(
            parallel_config=SimpleNamespace(
                sequence_parallel_size=4, ulysses_mode="advanced_uaa"
            )
        )
        with override_forward_context(ForwardContext(omni_diffusion_config=od_cfg_uaa)):
            attn(x_ok, (cos, sin))
        assert len(spy_calls) == 5

        with override_forward_context(
            ForwardContext(omni_diffusion_config=od_cfg_strict)
        ):
            # attn_mask triggers fallback
            attn(
                x_ok,
                (cos, sin),
                AttentionMetadata(attn_mask=torch.ones(1, 8, dtype=torch.bool)),
            )
            # joint_query triggers fallback
            attn(x_ok, (cos, sin), AttentionMetadata(joint_query=x_ok))
            # gate_compress triggers fallback
            attn(x_ok, (cos, sin), AttentionMetadata(extra={"gate_compress": x_ok}))
            # ring_world_size > 1 triggers fallback
            sp_group.ring_world_size = 2
            attn(x_ok, (cos, sin))
            sp_group.ring_world_size = 1
        assert len(spy_calls) == 9

        monkeypatch.setattr(
            "vllm_omni.diffusion.distributed.parallel_state.get_sp_group",
            lambda: (_ for _ in ()).throw(AssertionError("not initialized")),
        )
        with override_forward_context(
            ForwardContext(omni_diffusion_config=od_cfg_strict)
        ):
            attn(x_ok, (cos, sin))
        assert len(spy_calls) == 10
    finally:
        WanSelfAttention.forward = orig_forward


def test_resolve_flash_spec_uses_compiled_flash_op_contract(monkeypatch):
    calls: list[tuple[Any, ...]] = []

    def _fake_op(q, k, v, scale, bq, bkv, bkvc):
        calls.append((q.shape, k.shape, v.shape, scale, bq, bkv, bkvc))
        return q

    def _fake_resolve(seq_len: int, head_dim: int, scale: float) -> Any:
        assert seq_len == 75600
        assert head_dim == 128
        return SimpleNamespace(
            op=_fake_op,
            block_q=4736,
            block_kv=2048,
            block_kv_compute=2048,
            scale=scale,
            cache_key=("compiled", 4736, 2048, 2048, scale),
        )

    try:
        from vllm_omni_torchtpu.wan_opt import attn_kernel
    except ImportError:
        attn_kernel = SimpleNamespace()
        monkeypatch.setitem(
            sys.modules, "vllm_omni_torchtpu.wan_opt.attn_kernel", attn_kernel
        )

    monkeypatch.setattr(
        attn_kernel, "resolve_compiled_flash_op", _fake_resolve, raising=False
    )

    def _patched_forward(*args: Any, **kwargs: Any) -> Any:
        return None

    _patched_forward._tpu_wan_attn_fork_patched = True
    attn_mod = SimpleNamespace(
        attn=SimpleNamespace(
            attention=SimpleNamespace(
                use_sdpa=False,
                softmax_scale=0.08838834764831845,
                forward=_patched_forward,
            )
        )
    )

    spec = ulysses_pipe._resolve_flash_spec(attn_mod, 75600, 128, torch.bfloat16)
    assert spec is not None
    key, fn = spec
    assert key == (True, ("compiled", 4736, 2048, 2048, 0.08838834764831845))
    q = torch.zeros(1, 4, 2, 128, dtype=torch.bfloat16)
    out = fn(q, q, q)
    assert out is q
    assert calls == [
        (
            (1, 4, 2, 128),
            (1, 4, 2, 128),
            (1, 4, 2, 128),
            0.08838834764831845,
            4736,
            2048,
            2048,
        )
    ]

    monkeypatch.setattr(
        attn_kernel,
        "resolve_compiled_flash_op",
        lambda seq_len, head_dim, scale: None,
        raising=False,
    )
    assert (
        ulysses_pipe._resolve_flash_spec(attn_mod, 75600, 128, torch.bfloat16) is None
    )


def test_pure_bf16_simple_linear_without_fp8_core_ops(monkeypatch):
    lin = torch.nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.layers.adapter.linear_common", None)
    assert ulysses_pipe._get_weight_flipped_attr() == "_tpu_weight_flipped"
    assert ulysses_pipe._is_simple_linear(
        lin, in_features=64, out_features=64, dtype=torch.bfloat16
    )


def test_compiled_5d_linear_guards_and_info_log(monkeypatch):
    dim, heads, head_dim = 5120, 40, 128
    attn = _make_cpu_wan_self_attention(dim=dim, heads=heads, head_dim=head_dim)
    qkv_mod = torch.nn.Linear(dim, 3 * heads * head_dim, bias=True).to(torch.bfloat16)
    qkv_mod.return_bias = True
    qkv_mod.skip_bias_add = False
    attn.to_qkv = qkv_mod
    attn.to_out.return_bias = False
    attn.to_out.skip_bias_add = False
    attn.attn.attention = SimpleNamespace(
        forward=_sdpa_impl_forward,
        _op=lambda q, k, v, scale: _sdpa_impl_forward(q, k, v),
        softmax_scale=head_dim**-0.5,
    )
    sp_group = SimpleNamespace(
        ulysses_world_size=4,
        ring_world_size=1,
        allgather_world_size=1,
        ulysses_group="fake_pg",
    )
    x = torch.randn(1, 8, dim, dtype=torch.bfloat16)

    with torch.no_grad():
        assert ulysses_pipe._get_or_compile_pipe_fn(attn, x, sp_group, 2) is not None
        qkv_mod.skip_bias_add = True
        assert ulysses_pipe._get_or_compile_pipe_fn(attn, x, sp_group, 2) is None
        qkv_mod.skip_bias_add = False
        setattr(attn.to_out, WEIGHT_FLIPPED_ATTR, True)
        assert ulysses_pipe._get_or_compile_pipe_fn(attn, x, sp_group, 2) is None
        delattr(attn.to_out, WEIGHT_FLIPPED_ATTR)

    # Verify active info log reports mode=compiled_5d when pipe_fn is selected.
    info_msgs: list[str] = []
    warn_msgs: list[str] = []
    monkeypatch.setattr(
        ulysses_pipe.logger, "info", lambda msg, *a: info_msgs.append(msg % a)
    )
    monkeypatch.setattr(
        ulysses_pipe.logger, "warning", lambda msg, *a: warn_msgs.append(msg % a)
    )
    monkeypatch.setattr(ulysses_pipe, "_ACTIVE_LOGGED", False)
    monkeypatch.setattr(
        ulysses_pipe, "_get_or_compile_pipe_fn", lambda *a, **kw: lambda *x: x[0]
    )
    with torch.no_grad():
        cos = torch.ones(1, 8, 1, 64, dtype=torch.bfloat16)
        sin = torch.zeros(1, 8, 1, 64, dtype=torch.bfloat16)
        out = ulysses_pipe._run_chunked_ulysses_self_attention(
            attn, x, (cos, sin), sp_group, 2
        )
    assert out.shape == x.shape
    assert any("mode=compiled_5d" in m for m in info_msgs)
    assert warn_msgs == []


# --------------------------------------------------------------------------- #
# Multi-rank gloo CPU parity & permutation tests
# --------------------------------------------------------------------------- #


def _gloo_worker(rank: int, port: int, world_size: int) -> None:
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    pg = dist.group.WORLD
    sp_group = SimpleNamespace(
        ulysses_group=pg,
        ulysses_world_size=world_size,
        ring_world_size=1,
        allgather_world_size=1,
    )

    # 1. Exact permutation equivalence vs SeqAllToAll4D (G=2 and G=4, uneven chunks).
    dim = 5120
    heads = 40
    head_dim = 128
    seq_local = 8
    gen = torch.Generator().manual_seed(123)
    full_q = torch.randn(
        1, world_size * seq_local, heads, head_dim, generator=gen, dtype=torch.bfloat16
    )
    local_q = full_q[:, rank * seq_local : (rank + 1) * seq_local].contiguous()

    ref_fwd = SeqAllToAll4D.apply(pg, local_q, 2, 1, False)  # (1, S, H/U, D)
    ref_back = SeqAllToAll4D.apply(pg, ref_fwd, 1, 2, False)
    assert torch.equal(ref_back, local_q)

    # Check every chunk range [h0, h1) for G=2 and G=4 produces the exact global
    # token order on forward A2A and restores exact local heads on reverse A2A.
    local_flat = local_q.reshape(seq_local, heads * head_dim)
    for num_chunks in (2, 4):
        ranges = ulysses_pipe._ulysses_head_chunk_ranges(heads, world_size, num_chunks)
        rebuilt_chunks = []
        for h0, h1 in ranges:
            hl = (h1 - h0) // world_size
            col0, col1 = h0 * head_dim, h1 * head_dim
            chunk_w = hl * head_dim
            q_g = ulysses_pipe._a2a_forward_chunk(
                local_flat, col0, col1, seq_local, world_size, chunk_w, pg, False
            ).reshape(1, world_size * seq_local, hl, head_dim)
            want_g = full_q[:, :, h0 + rank * hl : h0 + (rank + 1) * hl, :]
            assert torch.equal(q_g, want_g), (
                f"rank={rank} G={num_chunks} [{h0},{h1}) forward A2A mismatch"
            )

            o_send = q_g.reshape(world_size, seq_local, chunk_w).contiguous()
            o_local = torch.empty_like(o_send)
            dist.all_to_all_single(o_local, o_send, group=pg)
            # Reshape (U, S_l, hl*D) -> (1, S_l, U*hl, D)
            rebuilt = o_local.transpose(0, 1).reshape(1, seq_local, h1 - h0, head_dim)
            rebuilt_chunks.append(rebuilt)
        assert torch.equal(torch.cat(rebuilt_chunks, dim=2), local_q), (
            f"rank={rank} G={num_chunks} reverse A2A mismatch"
        )

    # 2. Full WanSelfAttention parity vs wan_self_attention_fused + Ulysses.
    omni_rope._qk_norm_rope_op = lambda: _reference_qk_norm_rope
    od_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            sequence_parallel_size=world_size, ulysses_mode="strict"
        )
    )
    ulysses_strategy = UlyssesParallelAttention(
        sp_group=sp_group, scatter_idx=2, gather_idx=1, use_sync=False
    )

    def _ref_attn_call(impl):
        def _call(query, key, value, attn_metadata=None):
            q, k, v, meta, ctx = ulysses_strategy.pre_attention(
                query, key, value, attn_metadata
            )
            out = impl.forward(q, k, v, meta)
            return ulysses_strategy.post_attention(out, ctx)

        return _call

    for with_bias in (True, False):
        attn = _make_cpu_wan_self_attention(
            dim=dim, heads=heads, head_dim=head_dim, with_bias=with_bias, seed=7
        )
        attn.attn = _ref_attn_call(attn.attn.attention)
        # Re-attach attributes expected by ulysses_pipe alongside callable attn.
        attn.attn.attention = SimpleNamespace(forward=_sdpa_impl_forward)
        attn.attn.skip_sequence_parallel = False
        attn.attn.use_ring = False
        attn.attn._has_custom_attention = False
        attn.attn.causal = False
        attn.attn.scatter_idx = 2
        attn.attn.gather_idx = 1
        attn.attn.use_sync = False
        attn.attn._kv_cache_dtype = None
        attn.attn.is_paged_kv_active = lambda: False

        g_in = torch.Generator().manual_seed(99)
        s_glob = world_size * seq_local
        x_full = torch.randn(1, s_glob, dim, generator=g_in, dtype=torch.bfloat16)
        cos_full = torch.randn(
            1, s_glob, 1, head_dim // 2, generator=g_in, dtype=torch.bfloat16
        )
        sin_full = torch.randn(
            1, s_glob, 1, head_dim // 2, generator=g_in, dtype=torch.bfloat16
        )

        x_loc = x_full[:, rank * seq_local : (rank + 1) * seq_local].contiguous()
        cos_loc = cos_full[:, rank * seq_local : (rank + 1) * seq_local].contiguous()
        sin_loc = sin_full[:, rank * seq_local : (rank + 1) * seq_local].contiguous()

        with override_forward_context(ForwardContext(omni_diffusion_config=od_cfg)):
            ref_out = omni_rope.wan_self_attention_fused(
                attn, x_loc, (cos_loc, sin_loc), None
            )
            for num_chunks in (2, 4, 5):
                # Default path is bit-exact (torch.equal) vs unchunked forward.
                got_exact = ulysses_pipe._run_chunked_ulysses_self_attention(
                    attn,
                    x_loc,
                    (cos_loc, sin_loc),
                    sp_group,
                    num_chunks,
                )
                assert torch.equal(got_exact, ref_out), (
                    f"rank={rank} G={num_chunks} bias={with_bias} not bit-exact"
                )

            # 3. Also test 5D funcol prefetch path (exact production TPU math) on CPU.
            class _QkvLinear(torch.nn.Linear):
                def forward(self, inp):
                    return super().forward(inp), None

            qkv_mod = _QkvLinear(dim, 3 * heads * head_dim, bias=with_bias).to(
                dtype=torch.bfloat16
            )
            with torch.no_grad():
                g_5d = torch.Generator().manual_seed(11)
                qkv_mod.weight.copy_(
                    torch.randn(3 * heads * head_dim, dim, generator=g_5d) * 0.02
                )
                if with_bias:
                    qkv_mod.bias.copy_(
                        torch.randn(3 * heads * head_dim, generator=g_5d) * 0.02
                    )
            attn.to_qkv = qkv_mod
            attn.attn.attention = SimpleNamespace(
                forward=_sdpa_impl_forward,
                _op=lambda q, k, v, scale: _sdpa_impl_forward(q, k, v),
                softmax_scale=head_dim**-0.5,
            )
            with torch.no_grad():
                ref_5d = omni_rope.wan_self_attention_fused(
                    attn, x_loc, (cos_loc, sin_loc), None
                )
                for num_chunks in (2, 5):
                    assert (
                        ulysses_pipe._get_or_compile_pipe_fn(
                            attn, x_loc, sp_group, num_chunks
                        )
                        is not None
                    )
                    got_5d = ulysses_pipe._run_chunked_ulysses_self_attention(
                        attn,
                        x_loc,
                        (cos_loc, sin_loc),
                        sp_group,
                        num_chunks,
                    )
                    assert torch.equal(got_5d, ref_5d), (
                        f"rank={rank} 5D G={num_chunks} bias={with_bias} not bit-exact"
                    )

    dist.destroy_process_group()


def test_gloo_chunked_ulysses_parity():
    world_size = 4
    port = _free_port()
    mp.spawn(_gloo_worker, args=(port, world_size), nprocs=world_size, join=True)


# --------------------------------------------------------------------------- #
# TPU multi-rank block-level parity test (production shape 1, 18900, 5120)
# --------------------------------------------------------------------------- #


def _tpu_parity_worker(rank: int, port: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context

    from vllm_omni_torchtpu.platform import OmniTpuPlatform

    OmniTpuPlatform.set_device(0)
    apply_omni_model_specific_patches()

    backend = OmniTpuPlatform().dist_backend
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend=backend,
    )
    initialize_model_parallel(
        sequence_parallel_size=world_size, ulysses_degree=world_size
    )

    device = torch.device("tpu")
    dim, heads, head_dim, seq_local = 5120, 40, 128, 18900
    torch.manual_seed(0)
    attn = WanSelfAttention(dim=dim, num_heads=heads, head_dim=head_dim).to(
        device=device, dtype=torch.bfloat16
    )
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.02)

    torch.manual_seed(100 + rank)
    x = torch.randn(1, seq_local, dim, device=device, dtype=torch.bfloat16)
    cos = torch.randn(
        1, seq_local, 1, head_dim // 2, device=device, dtype=torch.bfloat16
    )
    sin = torch.randn(
        1, seq_local, 1, head_dim // 2, device=device, dtype=torch.bfloat16
    )

    od_cfg = OmniDiffusionConfig(
        parallel_config=DiffusionParallelConfig(ulysses_degree=world_size),
    )
    sp_group = get_sp_group()
    with set_forward_context(omni_diffusion_config=od_cfg):
        ref = omni_rope.wan_self_attention_fused(attn, x, (cos, sin), None)
        for num_chunks in (2, 5):
            got = ulysses_pipe._run_chunked_ulysses_self_attention(
                attn, x, (cos, sin), sp_group, num_chunks
            )
            torch.accelerator.synchronize()
            assert torch.equal(got.cpu(), ref.cpu()), (
                f"tpu_rank{rank} G={num_chunks} not bit-exact"
            )

    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.tpu_test
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.skipif(
    hardware.get_tpu_device_count() < 4,
    reason="requires >= 4 TPU chips",
)
def test_tpu_chunked_ulysses_block_parity():
    world_size = 4
    port = _free_port()
    mp.spawn(_tpu_parity_worker, args=(port, world_size), nprocs=world_size, join=True)
