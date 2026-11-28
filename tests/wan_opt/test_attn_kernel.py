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
"""CPU (Pallas interpret) and TPU tests for Wan2.2 forked splash attention."""

from __future__ import annotations

import functools
import math
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
import torch
from vllm_torchtpu.kernels.diffusion_attention import kernel as stock_kernel

from vllm_omni_torchtpu.attention import TpuFlashAttentionImpl
from vllm_omni_torchtpu.wan_opt import _splash_blocks, _splash_fork, attn_kernel


def _f32_attention_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
) -> jax.Array:
    """Exact fp32 reference returning ``out_bshd``."""
    q32 = q.astype(jnp.float32)
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q32, k32) * scale
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, v32)


def _rel_l2(got: jax.Array, ref: jax.Array) -> float:
    diff = got.astype(jnp.float32) - ref.astype(jnp.float32)
    return float(jnp.linalg.norm(diff) / jnp.linalg.norm(ref.astype(jnp.float32)))


def _cosine(got: jax.Array, ref: jax.Array) -> float:
    g = got.astype(jnp.float32).ravel()
    r = ref.astype(jnp.float32).ravel()
    return float(jnp.dot(g, r) / (jnp.linalg.norm(g) * jnp.linalg.norm(r)))


@pytest.fixture
def interpret_pallas(monkeypatch):
    """Forces Pallas ``interpret=True`` so kernel bodies run on CPU."""
    orig_pallas_call = _splash_fork.pl.pallas_call
    monkeypatch.setattr(
        _splash_fork.pl,
        "pallas_call",
        functools.partial(orig_pallas_call, interpret=True),
    )
    orig_stock = stock_kernel.pl.pallas_call
    monkeypatch.setattr(
        stock_kernel.pl,
        "pallas_call",
        functools.partial(orig_stock, interpret=True),
    )


@pytest.fixture
def restore_flash_impl():
    """Restores ``TpuFlashAttentionImpl.forward`` and ``_ACTIVE_CONFIG``."""
    orig_forward = getattr(
        TpuFlashAttentionImpl,
        "_tpu_wan_attn_orig_forward",
        TpuFlashAttentionImpl.forward,
    )
    orig_cfg = attn_kernel._ACTIVE_CONFIG
    yield
    TpuFlashAttentionImpl.forward = orig_forward
    attn_kernel._ACTIVE_CONFIG = orig_cfg


def test_default_off_has_no_heavy_import_overhead():
    """Importing and calling apply() when OFF does not load heavy runtime deps."""
    import ast
    from pathlib import Path

    from vllm_omni_torchtpu.wan_opt import vae_parallel

    for path in (attn_kernel.__file__, vae_parallel.__file__):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        top_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.append(node.module)
        forbidden = ("jax", "torch", "torch_tpu")
        assert not any(
            mod == tok or mod.startswith(f"{tok}.") or "_splash_fork" in mod
            for mod in top_imports
            for tok in forbidden
        ), f"Unexpected top-level imports in {path}: {top_imports}"

    code = (
        "import os, sys\n"
        "os.environ.pop('VLLM_OMNI_WAN_ATTN_FORK', None)\n"
        "os.environ.pop('VLLM_OMNI_WAN_VAE_PP_FIX', None)\n"
        "from vllm_omni_torchtpu.wan_opt import attn_kernel, vae_parallel\n"
        "assert attn_kernel.apply() is False\n"
        "assert attn_kernel.resolve_compiled_flash_op(4096, 128, 0.088) is None\n"
        "assert vae_parallel.apply() is False\n"
        "heavy = ('torch', 'torch_tpu', 'torch.distributed', 'vllm_omni', "
        "'vllm_omni_torchtpu.attention', "
        "'vllm_omni_torchtpu.wan_opt._splash_blocks', "
        "'vllm_omni_torchtpu.wan_opt._splash_fork')\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, f'Unexpected heavy imports: {loaded}'\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_apply_disabled_by_default_and_rejects_bad_env(monkeypatch, restore_flash_impl):
    monkeypatch.delenv(attn_kernel.ENV, raising=False)
    orig = TpuFlashAttentionImpl.forward
    assert attn_kernel.apply() is False
    assert TpuFlashAttentionImpl.forward is orig
    assert attn_kernel.resolve_compiled_flash_op(4096, 128, 0.088) is None

    monkeypatch.setenv(attn_kernel.ENV, "treu")
    with pytest.raises(ValueError, match="Invalid boolean value"):
        attn_kernel.apply()


def test_resolve_block_sizes_overrides_and_validation():
    spec = _splash_blocks.V7_SPEC
    # Default Wan2.2 shape (75600) under fixed-m caps at bq=3840, bkv=2048.
    default_bs = _splash_blocks.resolve_block_sizes(
        75600, 75600, 128, spec=spec, kernel_mode="fixedm"
    )
    assert default_bs == {"block_q": 3840, "block_kv": 2048, "block_kv_compute": 2048}

    # Explicit BQ=4736 exceeds 0.6 VMEM heuristic and is allowed up to 63 MiB.
    override_bs = _splash_blocks.resolve_block_sizes(
        75600,
        75600,
        128,
        spec=spec,
        kernel_mode="fixedm",
        override_bq=4736,
        override_bkv=2048,
    )
    assert override_bs == {
        "block_q": 4736,
        "block_kv": 2048,
        "block_kv_compute": 2048,
    }

    # Overriding BKV without BKVC tracks BKV.
    bkv_only = _splash_blocks.resolve_block_sizes(
        75600,
        75600,
        128,
        spec=spec,
        kernel_mode="fixedm",
        override_bq=4608,
        override_bkv=1024,
    )
    assert bkv_only == {"block_q": 4608, "block_kv": 1024, "block_kv_compute": 1024}

    # Invalid overrides fall back to autotuned:
    # 1) bkv % bkvc != 0
    assert (
        _splash_blocks.resolve_block_sizes(
            75600,
            75600,
            128,
            spec=spec,
            kernel_mode="fixedm",
            override_bkv=2048,
            override_bkvc=1536,
        )
        == default_bs
    )
    # 2) bq % 128 != 0 when bq < S
    assert (
        _splash_blocks.resolve_block_sizes(
            75600,
            75600,
            128,
            spec=spec,
            kernel_mode="fixedm",
            override_bq=4000,
        )
        == default_bs
    )
    # 3) bkv % 16 != 0 or bkvc % 16 != 0 when < S
    assert (
        _splash_blocks.resolve_block_sizes(
            75600,
            75600,
            128,
            spec=spec,
            kernel_mode="fixedm",
            override_bkv=1000,
        )
        == default_bs
    )
    assert (
        _splash_blocks.resolve_block_sizes(
            75600,
            75600,
            128,
            spec=spec,
            kernel_mode="fixedm",
            override_bkv=2000,
            override_bkvc=1000,
        )
        == default_bs
    )
    # 4) VMEM > 63 MiB
    assert (
        _splash_blocks.resolve_block_sizes(
            75600,
            75600,
            128,
            spec=spec,
            kernel_mode="fixedm",
            override_bq=8192,
            override_bkv=2048,
        )
        == default_bs
    )


@pytest.mark.parametrize("sq,skv", [(5968, 5968), (777, 512)])
def test_default_fork_bit_exact_to_stock_interpret(interpret_pallas, sq, skv):
    """Default fork (no overrides, uniform=False, sparse=None) is bit-exact."""
    key = jax.random.PRNGKey(7)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, sq, 2, 128), jnp.bfloat16)
    k = jax.random.normal(kk, (1, skv, 2, 128), jnp.bfloat16)
    v = jax.random.normal(kv, (1, skv, 2, 128), jnp.bfloat16)
    scale = 1.0 / math.sqrt(128)

    got = _splash_fork.flash_attention_bshd_fork(q, k, v, scale=scale)
    stock = stock_kernel.flash_attention_bshd(q, k, v, scale=scale)
    assert bool(jnp.array_equal(got, stock))


@pytest.mark.parametrize(
    "sq,skv,bq,bkv,bkvc",
    [
        (4500, 4500, None, None, None),  # multi-block fixed-m + ragged KV tail
        (777, 512, None, None, None),  # cross-attention shape
        (4500, 4500, 1024, 1024, 512),  # explicit block overrides
    ],
)
def test_fork_kernel_parity_interpret(interpret_pallas, sq, skv, bq, bkv, bkvc):
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, sq, 2, 128), jnp.bfloat16)
    k = jax.random.normal(kk, (1, skv, 2, 128), jnp.bfloat16)
    v = jax.random.normal(kv, (1, skv, 2, 128), jnp.bfloat16)
    scale = 1.0 / math.sqrt(128)

    got = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkvc,
    )
    ref = _f32_attention_ref(q, k, v, scale)
    stock = stock_kernel.flash_attention_bshd(q, k, v, scale=scale)
    assert got.dtype == jnp.bfloat16
    assert got.shape == ref.shape
    assert _rel_l2(got, stock) <= 2e-3
    assert _rel_l2(got, ref) <= 3e-3
    assert _cosine(got, ref) >= 0.9999


def test_apply_installs_wrapper_and_routes_blocks(monkeypatch, restore_flash_impl):
    calls: list[tuple] = []

    def fake_op(q, k, v, scale, bq, bkv, bkvc):
        calls.append((q.shape, k.shape, scale, bq, bkv, bkvc))
        return q.clone()

    monkeypatch.setattr(attn_kernel, "_flash_attention_fork_op", lambda: fake_op)
    monkeypatch.setenv(attn_kernel.ENV, "1")
    monkeypatch.setenv(attn_kernel.ENV_BQ, "4736")
    monkeypatch.setenv(attn_kernel.ENV_BKV, "2048")
    monkeypatch.delenv(attn_kernel.ENV_BKVC, raising=False)

    mock_info = MagicMock()
    mock_warn = MagicMock()
    monkeypatch.setattr(attn_kernel.logger, "info", mock_info)
    monkeypatch.setattr(attn_kernel.logger, "warning", mock_warn)

    assert attn_kernel.apply() is True
    assert mock_info.call_count == 1
    assert mock_warn.call_count == 0
    # Idempotent re-apply returns True without re-wrapping.
    assert attn_kernel.apply() is True

    impl = TpuFlashAttentionImpl(num_heads=10, head_size=128, softmax_scale=0.088)
    # 1. Self-attention uses explicit BQ/BKV overrides.
    q = torch.randn(1, 75600, 10, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 75600, 10, 128, dtype=torch.bfloat16)
    v = torch.randn(1, 75600, 10, 128, dtype=torch.bfloat16)
    out = impl.forward(q, k, v)
    assert out.shape == q.shape
    assert calls[-1] == (
        (1, 75600, 10, 128),
        (1, 75600, 10, 128),
        0.088,
        4736,
        2048,
        2048,
    )

    # 2. Cross-attention (kv_len=512 <= safe_kv_block) uses autotuned tiles.
    qc = torch.randn(1, 18944, 40, 128, dtype=torch.bfloat16)
    kc = torch.randn(1, 512, 40, 128, dtype=torch.bfloat16)
    vc = torch.randn(1, 512, 40, 128, dtype=torch.bfloat16)
    impl.forward(qc, kc, vc)
    cross_auto = _splash_blocks.autotune_block_sizes(
        18944, 128, 2, kv_seq_len=512, kernel_mode="online"
    )
    assert calls[-1][3:] == (
        cross_auto["block_q"],
        cross_auto["block_kv"],
        cross_auto["block_kv_compute"],
    )

    # 3. SDPA fallbacks still work (masked, non-bf16, causal).
    n_calls = len(calls)
    meta = SimpleNamespace(attn_mask=torch.ones(1, 16, dtype=torch.bool))
    qs = torch.randn(1, 16, 10, 128, dtype=torch.bfloat16)
    ks = torch.randn(1, 16, 10, 128, dtype=torch.bfloat16)
    vs = torch.randn(1, 16, 10, 128, dtype=torch.bfloat16)
    impl.forward(qs, ks, vs, attn_metadata=meta)
    impl.forward(qs.float(), ks.float(), vs.float())
    impl_causal = TpuFlashAttentionImpl(
        num_heads=10, head_size=128, softmax_scale=0.088, causal=True
    )
    impl_causal.forward(qs, ks, vs)
    assert len(calls) == n_calls

    # 4. Repeat apply() updates _ACTIVE_CONFIG seen by _forked_flash_forward.
    monkeypatch.setenv(attn_kernel.ENV_BQ, "4608")
    assert attn_kernel.apply() is True
    impl.forward(q, k, v)
    assert calls[-1][3:] == (4608, 2048, 2048)


def test_config_keyed_op_factory_and_env_toggle(monkeypatch, restore_flash_impl):
    monkeypatch.delenv(attn_kernel.ENV_SPLASH_UNIFORM, raising=False)
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_ATTN, raising=False)
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_KV_BLOCKS, raising=False)
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_SINK_BLOCKS, raising=False)
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_WARMUP_STEPS, raising=False)
    attn_kernel._ACTIVE_CONFIG = None

    op_base = attn_kernel._flash_attention_fork_op()
    assert op_base._qualname == "pallas::wan_flash_attention_fork"

    monkeypatch.setenv(attn_kernel.ENV_SPLASH_UNIFORM, "1")
    op_uni = attn_kernel._flash_attention_fork_op()
    assert op_uni is not op_base
    assert op_uni._qualname == "pallas::wan_flash_attention_fork_u1"

    monkeypatch.setenv(attn_kernel.ENV_SPARSE_ATTN, "1")
    monkeypatch.setenv(attn_kernel.ENV_SPARSE_WARMUP_STEPS, "0")
    op_sparse = attn_kernel._flash_attention_fork_op()
    assert op_sparse._qualname == "pallas::wan_flash_attention_fork_u1_sk22_ss2"


def test_uniform_fixed_m_parity_interpret(interpret_pallas):
    key = jax.random.PRNGKey(11)
    kq, kk, kv = jax.random.split(key, 3)
    sq, skv = 2560, 4500
    q = jax.random.normal(kq, (1, sq, 2, 128), jnp.bfloat16)
    k = jax.random.normal(kk, (1, skv, 2, 128), jnp.bfloat16)
    v = jax.random.normal(kv, (1, skv, 2, 128), jnp.bfloat16)
    scale = 1.0 / math.sqrt(128)

    ref_out = _f32_attention_ref(q, k, v, scale)
    bf16_base = _splash_fork.flash_attention_bshd_fork(
        q, k, v, scale=scale, block_q=1024, block_kv=1024, block_kv_compute=1024
    )
    got = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=1024,
        block_kv=1024,
        block_kv_compute=1024,
        uniform_fixed_m=True,
    )
    assert got.dtype == jnp.bfloat16
    assert _rel_l2(got, bf16_base) <= 2e-3
    assert _rel_l2(got, ref_out) <= 3e-3
    assert _cosine(got, ref_out) >= 0.9999


def test_uniform_fixed_m_lax_cond_mixed_fallback_interpret(interpret_pallas):
    """When any head fails fixed-m, lax.cond falls back to hybrid cleanly."""
    key = jax.random.PRNGKey(12)
    kq, kk, kv = jax.random.split(key, 3)
    sq, skv = 4500, 4500
    q = jax.random.normal(kq, (1, sq, 2, 128), jnp.bfloat16)
    q = q.at[:, 1024:2048, 0, :].multiply(25.0)
    k = jax.random.normal(kk, (1, skv, 2, 128), jnp.bfloat16)
    v = jax.random.normal(kv, (1, skv, 2, 128), jnp.bfloat16)
    scale = 1.0 / math.sqrt(128)
    ref_out = _f32_attention_ref(q, k, v, scale)

    got = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=1024,
        block_kv=1024,
        block_kv_compute=1024,
        uniform_fixed_m=True,
    )
    assert _rel_l2(got, ref_out) <= 3e-3
    assert _cosine(got, ref_out) >= 0.9999


def test_build_sparse_kv_indices():
    kv_idx = _splash_fork.build_sparse_kv_indices(
        grid_height=16,
        grid_width=37,
        block_q=4736,
        block_kv=2048,
        k_active=22,
        sink_blocks=2,
    )
    assert kv_idx.shape == (16, 22)
    for row in kv_idx.tolist():
        assert row[:2] == [0, 1]
        assert len(set(row)) == 22
        assert row == sorted(row)
        assert all(0 <= x < 37 for x in row)
        assert 36 not in row[:-1]


def test_sparse_block_masked_ref_interpret(interpret_pallas):
    key = jax.random.PRNGKey(21)
    kq, kk, kv = jax.random.split(key, 3)
    sq, skv = 4500, 4500
    bq, bkv = 1024, 1024
    k_active, sink = 3, 1
    q = jax.random.normal(kq, (1, sq, 2, 128), jnp.bfloat16)
    k = jax.random.normal(kk, (1, skv, 2, 128), jnp.bfloat16)
    v = jax.random.normal(kv, (1, skv, 2, 128), jnp.bfloat16)
    scale = 1.0 / math.sqrt(128)

    # k_active == grid_width falls back cleanly to dense uniform_fixed_m.
    dense_ref = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        uniform_fixed_m=True,
    )
    sparse_full = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        uniform_fixed_m=True,
        sparse_kv_blocks=5,
        sparse_sink_blocks=sink,
    )
    assert _rel_l2(sparse_full, dense_ref) <= 1e-3

    # Build explicit token-level block-sparse mask matching kv_idx[i, :].
    gh = (sq + bq - 1) // bq
    gw = (skv + bkv - 1) // bkv
    kv_idx = _splash_fork.build_sparse_kv_indices(
        gh, gw, bq, bkv, k_active=k_active, sink_blocks=sink
    )
    mask_2d = jnp.zeros((sq, skv), dtype=jnp.bool_)
    for i in range(gh):
        q0, q1 = i * bq, min((i + 1) * bq, sq)
        for j_idx in kv_idx[i].tolist():
            k0, k1 = j_idx * bkv, min((j_idx + 1) * bkv, skv)
            mask_2d = mask_2d.at[q0:q1, k0:k1].set(True)

    logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
    logits = jnp.where(mask_2d[None, None, :, :], logits * scale, -1e30)
    probs = jax.nn.softmax(logits, axis=-1)
    ref_out = jnp.einsum("bhqk,bkhd->bqhd", probs, v.astype(jnp.float32))

    got_out = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        uniform_fixed_m=True,
        sparse_kv_blocks=k_active,
        sparse_sink_blocks=sink,
    )
    assert got_out.dtype == jnp.bfloat16
    assert _rel_l2(got_out, ref_out) <= 3e-3
    assert _cosine(got_out, ref_out) >= 0.9999


def test_resolve_compiled_flash_op_and_step_warmup(monkeypatch, restore_flash_impl):
    from vllm_omni.diffusion.forward_context import (
        set_forward_context,
        set_forward_context_denoise_step_idx,
    )

    # Disabled when ENV is unset or apply() has not been called.
    monkeypatch.delenv(attn_kernel.ENV, raising=False)
    attn_kernel._ACTIVE_CONFIG = None
    assert attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088) is None

    monkeypatch.setenv(attn_kernel.ENV, "1")
    assert attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088) is None

    monkeypatch.setenv(attn_kernel.ENV_SPLASH_UNIFORM, "1")
    monkeypatch.setenv(attn_kernel.ENV_SPARSE_ATTN, "1")
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_KV_BLOCKS, raising=False)
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_SINK_BLOCKS, raising=False)
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_WARMUP_STEPS, raising=False)
    assert attn_kernel.apply() is True

    # Outside ForwardContext: stays dense during default 20-step warmup.
    dense_spec = attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088)
    assert isinstance(dense_spec, attn_kernel.CompiledFlashSpec)
    assert dense_spec.op._qualname == "pallas::wan_flash_attention_fork_u1"
    assert dense_spec.cache_key == (
        id(dense_spec.op),
        dense_spec.block_q,
        dense_spec.block_kv,
        dense_spec.block_kv_compute,
        0.088,
    )

    with set_forward_context():
        set_forward_context_denoise_step_idx(0)
        assert attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088) == dense_spec
        set_forward_context_denoise_step_idx(19)
        assert attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088) == dense_spec

        set_forward_context_denoise_step_idx(20)
        sparse_spec = attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088)
        assert sparse_spec is not None
        assert (
            sparse_spec.op._qualname == "pallas::wan_flash_attention_fork_u1_sk22_ss2"
        )
        assert sparse_spec.cache_key != dense_spec.cache_key
        assert sparse_spec.cache_key == (
            id(sparse_spec.op),
            sparse_spec.block_q,
            sparse_spec.block_kv,
            sparse_spec.block_kv_compute,
            0.088,
        )

    # warmup_steps=0 enables sparse op unconditionally.
    monkeypatch.setenv(attn_kernel.ENV_SPARSE_WARMUP_STEPS, "0")
    assert attn_kernel.apply() is True
    uncond_spec = attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088)
    assert uncond_spec == sparse_spec

    # Turning ENV off disables resolve_compiled_flash_op immediately.
    monkeypatch.setenv(attn_kernel.ENV, "0")
    assert attn_kernel.resolve_compiled_flash_op(18900, 128, 0.088) is None


@pytest.mark.parametrize(
    "seq_len,head_dim",
    [
        (4096, 64),  # LTX-2 head_dim=64, 2 KV blocks -> stays dense
        (4500, 64),  # LTX-2 head_dim=64, 3 KV blocks (< 22) -> stays dense
        (4096, 128),  # Flux / Qwen-Image S=4096 -> stays dense
    ],
)
def test_non_wan_shapes_stay_dense_and_accurate_interpret(
    interpret_pallas,
    monkeypatch,
    restore_flash_impl,
    seq_len,
    head_dim,
):
    """Non-Wan shapes stay dense and accurate with ATTN_FORK + UNIFORM + SPARSE."""
    sparse_calls = 0
    orig_build_sparse = _splash_fork.build_sparse_kv_indices

    def _spy_build_sparse(*args, **kwargs):
        nonlocal sparse_calls
        sparse_calls += 1
        return orig_build_sparse(*args, **kwargs)

    monkeypatch.setattr(_splash_fork, "build_sparse_kv_indices", _spy_build_sparse)
    monkeypatch.setenv(attn_kernel.ENV, "1")
    monkeypatch.setenv(attn_kernel.ENV_SPLASH_UNIFORM, "1")
    monkeypatch.setenv(attn_kernel.ENV_SPARSE_ATTN, "1")
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_KV_BLOCKS, raising=False)
    monkeypatch.delenv(attn_kernel.ENV_SPARSE_SINK_BLOCKS, raising=False)
    monkeypatch.setenv(attn_kernel.ENV_SPARSE_WARMUP_STEPS, "0")
    assert attn_kernel.apply() is True

    cfg = attn_kernel._current_config()
    spec = attn_kernel.resolve_compiled_flash_op(seq_len, head_dim, head_dim**-0.5)
    assert spec is not None

    key = jax.random.PRNGKey(31 + seq_len + head_dim)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, seq_len, 2, head_dim), jnp.bfloat16)
    k = jax.random.normal(kk, (1, seq_len, 2, head_dim), jnp.bfloat16)
    v = jax.random.normal(kv, (1, seq_len, 2, head_dim), jnp.bfloat16)
    scale = 1.0 / math.sqrt(head_dim)

    got = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=spec.block_q,
        block_kv=spec.block_kv,
        block_kv_compute=spec.block_kv_compute,
        uniform_fixed_m=cfg.uniform_fixed_m,
        sparse_kv_blocks=cfg.sparse_kv_blocks,
        sparse_sink_blocks=cfg.sparse_sink_blocks,
    )
    assert sparse_calls == 0

    ref = _f32_attention_ref(q, k, v, scale)
    dense_ref = _splash_fork.flash_attention_bshd_fork(
        q,
        k,
        v,
        scale=scale,
        block_q=spec.block_q,
        block_kv=spec.block_kv,
        block_kv_compute=spec.block_kv_compute,
        uniform_fixed_m=cfg.uniform_fixed_m,
    )
    stock = stock_kernel.flash_attention_bshd(q, k, v, scale=scale)
    assert got.dtype == jnp.bfloat16
    assert got.shape == ref.shape
    assert bool(jnp.array_equal(got, dense_ref))
    assert _rel_l2(got, stock) <= 3e-3
    assert _rel_l2(got, ref) <= 3e-3
    assert _cosine(got, ref) >= 0.9999


@pytest.mark.tpu_test
def test_tpu_fork_parity_real_shapes(monkeypatch, restore_flash_impl):
    """Runs on TPU pod via torch_tpu pallas.jax_op at real Wan2.2 shapes."""
    if not torch.accelerator.is_available():
        pytest.skip("TPU accelerator required")
    from vllm_omni_torchtpu.attention import _flash_attention_op

    dev = torch.device("tpu")
    sq, heads, dim = 75600, 10, 128
    scale = 1.0 / math.sqrt(dim)
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, sq, heads, dim, generator=g, dtype=torch.bfloat16).to(dev)
    k = torch.randn(1, sq, heads, dim, generator=g, dtype=torch.bfloat16).to(dev)
    v = torch.randn(1, sq, heads, dim, generator=g, dtype=torch.bfloat16).to(dev)

    stock_op = _flash_attention_op()
    fork_op = attn_kernel._flash_attention_fork_op()
    stock = stock_op(q, k, v, scale)
    got_default = fork_op(q, k, v, scale, 3840, 2048, 2048)
    torch.accelerator.synchronize()
    assert torch.equal(got_default.cpu(), stock.cpu())

    # BQ=4736, BKV=2048 parity
    got_bq = fork_op(q, k, v, scale, 4736, 2048, 2048)
    torch.accelerator.synchronize()
    diff_bq = got_bq.float().cpu() - stock.float().cpu()
    rel_bq = float(diff_bq.norm() / stock.float().cpu().norm())
    assert rel_bq <= 2e-3
