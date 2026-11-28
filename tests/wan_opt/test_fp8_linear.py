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
"""CPU unit and gloo parity tests for P3-C FP8 DiT dense linears."""

from __future__ import annotations

import contextlib
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
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    WanSelfAttention,
    WanTransformer3DModel,
)
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)

from tests.omni_test_utils import assert_parity
from vllm_omni_torchtpu import rope as omni_rope
from vllm_omni_torchtpu.wan_opt import fp8_linear, ulysses_pipe


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _single_rank_tp_group():
    if model_parallel_is_initialized():
        yield
        return
    port = _free_port()
    prev_env = {
        k: os.environ.get(k)
        for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    prev_device_type = current_platform.device_type
    prev_device_name = current_platform.device_name
    prev_is_oot = current_platform.is_out_of_tree
    current_platform.device_type = "cpu"
    current_platform.device_name = "cpu"
    current_platform.is_out_of_tree = lambda: False
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend="gloo",
            )
            initialize_model_parallel()
            yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        current_platform.device_type = prev_device_type
        current_platform.device_name = prev_device_name
        current_platform.is_out_of_tree = prev_is_oot
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cpu_quantized_matmul_ref(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w8a16_max_rows: int = 0,
) -> torch.Tensor:
    """CPU reference matching ``linear_common._quantized_matmul_jax``."""
    del w8a16_max_rows
    orig_out_shape = (*x.shape[:-1], w_q.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    inv_max = 1.0 / fp8_max

    x_abs = x_2d.abs().amax(dim=-1, keepdim=True)
    x_scale = x_abs * inv_max
    x_inv = torch.where(x_scale == 0, torch.zeros_like(x_scale), 1.0 / x_scale)
    x_q = (
        torch.clamp(x_2d * x_inv, -fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    out_2d = (x_q @ w_q.to(torch.float32)) * x_scale * w_s.unsqueeze(0)
    return out_2d.to(x.dtype).reshape(orig_out_shape)


def test_vllm_platform_accepts_fp8_env_vars() -> None:
    for name in (
        fp8_linear.ENV,
        fp8_linear.SKIP_EDGE_BLOCKS_ENV,
        fp8_linear.CROSS_KV_ENV,
        fp8_linear.EXCLUDE_ENV,
        fp8_linear.SKIP_HIGH_NOISE_ENV,
    ):
        assert name.startswith("VLLM_OMNI_")
    saved = dict(os.environ)
    try:
        os.environ.update(
            {
                fp8_linear.ENV: "1",
                fp8_linear.SKIP_EDGE_BLOCKS_ENV: "0",
                fp8_linear.CROSS_KV_ENV: "1",
                fp8_linear.EXCLUDE_ENV: "attn2",
                fp8_linear.SKIP_HIGH_NOISE_ENV: "1",
            }
        )
        current_platform.validate_environ(hard_fail=True)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_apply_env_gate_and_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_forward = WanTransformer3DModel.forward
    try:
        monkeypatch.delenv(fp8_linear.ENV, raising=False)
        assert fp8_linear.apply() is False
        assert WanTransformer3DModel.forward is orig_forward

        monkeypatch.setenv(fp8_linear.ENV, "0")
        assert fp8_linear.apply() is False
        assert WanTransformer3DModel.forward is orig_forward

        monkeypatch.setenv(fp8_linear.ENV, "1")
        assert fp8_linear.apply() is True
        patched = WanTransformer3DModel.forward
        assert patched is not orig_forward
        assert getattr(patched, fp8_linear._FP8_LINEAR_PATCHED, False)

        # Idempotent on repeat apply().
        assert fp8_linear.apply() is False
        assert WanTransformer3DModel.forward is patched
    finally:
        WanTransformer3DModel.forward = orig_forward


def test_missing_linear_common_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.layers.adapter.linear_common", None)
    with pytest.raises(ImportError, match="vllm-torchtpu-core"):
        fp8_linear._import_linear_common()

    monkeypatch.setitem(
        sys.modules,
        "vllm_torchtpu.layers.adapter.linear_common",
        SimpleNamespace(KEEP_VLLM_LAYOUT_ATTR="a"),
    )
    with pytest.raises(ImportError, match="vllm-torchtpu-core"):
        fp8_linear._import_linear_common()


@pytest.mark.parametrize("flipped", [False, True])
def test_quantize_weight_fp8_layouts_and_parity(flipped: bool) -> None:
    gen = torch.Generator().manual_seed(42)
    k_in, n_out = 256, 384
    w_n_k = torch.randn(n_out, k_in, generator=gen, dtype=torch.bfloat16) * 0.05
    # Include an all-zero row to verify zero-scale safety (no NaNs).
    w_n_k[0, :] = 0.0
    w_in = w_n_k.transpose(0, 1).contiguous() if flipped else w_n_k

    w_q, w_s = fp8_linear.quantize_weight_fp8(w_in, flipped=flipped)
    assert w_q.shape == (k_in, n_out)
    assert w_q.dtype == torch.float8_e4m3fn
    assert w_q.is_contiguous()
    assert w_s.dtype == torch.float32
    assert w_s.is_contiguous()
    assert torch.isfinite(w_s).all()
    assert w_s.shape == (n_out,)
    assert float(w_s[0]) == 0.0

    x = torch.randn(2, 64, k_in, generator=gen, dtype=torch.bfloat16)
    ref = F.linear(x, w_n_k)
    got = _cpu_quantized_matmul_ref(x, w_q, w_s)
    assert torch.isfinite(got).all()
    assert_parity(got, ref, name=f"fp8_per_channel_flip{flipped}", max_rel_l2=5e-2)


def test_quantize_weight_fp8_errors() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fp8_linear.quantize_weight_fp8(torch.zeros(4, 4, 4), flipped=False)


def _make_dummy_wan_model(num_blocks: int = 3) -> torch.nn.Module:
    class _DummyNorm(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    class _DummyBlock(torch.nn.Module):
        def __init__(self, dim: int, ffn_dim: int, idx: int) -> None:
            super().__init__()
            self.norm1 = _DummyNorm(dim)
            self.attn1 = torch.nn.Module()
            self.attn1.to_qkv = QKVParallelLinear(
                hidden_size=dim,
                head_size=128,
                total_num_heads=2,
                bias=True,
                params_dtype=torch.bfloat16,
                prefix=f"blocks.{idx}.attn1.to_qkv",
            )
            self.attn1.to_out = RowParallelLinear(
                dim,
                dim,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                params_dtype=torch.bfloat16,
                prefix=f"blocks.{idx}.attn1.to_out",
            )
            self.attn1.to_gate_compress = ColumnParallelLinear(
                dim,
                dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                params_dtype=torch.bfloat16,
                prefix=f"blocks.{idx}.attn1.to_gate_compress",
            )

            self.attn2 = torch.nn.Module()
            for proj in ("to_q", "to_k", "to_v"):
                setattr(
                    self.attn2,
                    proj,
                    ColumnParallelLinear(
                        dim,
                        dim,
                        bias=True,
                        gather_output=False,
                        return_bias=False,
                        params_dtype=torch.bfloat16,
                        prefix=f"blocks.{idx}.attn2.{proj}",
                    ),
                )
            self.attn2.to_out = RowParallelLinear(
                dim,
                dim,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                params_dtype=torch.bfloat16,
                prefix=f"blocks.{idx}.attn2.to_out",
            )

            self.ffn = torch.nn.Module()
            self.ffn.net_0 = torch.nn.Module()
            self.ffn.net_0.proj = ColumnParallelLinear(
                dim,
                ffn_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                params_dtype=torch.bfloat16,
                prefix=f"blocks.{idx}.ffn.net_0.proj",
            )
            self.ffn.net_2 = RowParallelLinear(
                ffn_dim,
                dim,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                params_dtype=torch.bfloat16,
                prefix=f"blocks.{idx}.ffn.net_2",
            )
            self.scale_shift_table = torch.nn.Parameter(
                torch.zeros(1, 6, dim, dtype=torch.bfloat16)
            )

    class _DummyWanModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim, ffn_dim = 256, 384
            self.patch_embedding = torch.nn.Conv3d(16, dim, kernel_size=1)
            self.condition_embedder = torch.nn.Module()
            self.condition_embedder.time_proj = torch.nn.Linear(
                dim, dim * 6, dtype=torch.bfloat16
            )
            self.condition_embedder.text_embedder = ColumnParallelLinear(
                dim,
                dim,
                params_dtype=torch.bfloat16,
                prefix="condition_embedder.text_embedder",
            )
            self.blocks = torch.nn.ModuleList(
                [_DummyBlock(dim, ffn_dim, i) for i in range(num_blocks)]
            )
            self.proj_out = torch.nn.Linear(dim, 16, dtype=torch.bfloat16)

        def forward(self, hidden_states: Any, timestep: Any) -> Any:
            del hidden_states, timestep
            return torch.zeros(1)

    return _DummyWanModel()


def test_quantize_wan_model_linears_selection_and_knobs() -> None:
    with _single_rank_tp_group():
        model = _make_dummy_wan_model(num_blocks=3)
        count = fp8_linear.quantize_wan_model_linears(model)
        assert count == 24  # 3 blocks * 8 target linears

        # Non-block & excluded layers stay BF16.
        assert model.patch_embedding.weight.dtype == torch.float32
        assert model.condition_embedder.time_proj.weight.dtype == torch.bfloat16
        assert model.condition_embedder.text_embedder.weight.dtype == torch.bfloat16
        assert not getattr(
            model.condition_embedder.text_embedder, "_wan_fp8_active", False
        )
        assert model.proj_out.weight.dtype == torch.bfloat16
        assert model.blocks[0].norm1.weight.dtype == torch.bfloat16
        assert model.blocks[0].scale_shift_table.dtype == torch.bfloat16
        assert model.blocks[0].attn1.to_gate_compress.weight.dtype == torch.bfloat16
        assert not getattr(
            model.blocks[0].attn1.to_gate_compress, "_wan_fp8_active", False
        )

        # Target layers are K-major FP8 with _wan_fp8_active & _wan_fp8_scale.
        qkv = model.blocks[1].attn1.to_qkv
        assert qkv._wan_fp8_active is True
        assert qkv.weight.dtype == torch.float8_e4m3fn
        assert qkv.weight.shape == (256, 768)
        assert qkv._wan_fp8_scale.shape == (768,)
        assert qkv._wan_fp8_scale.dtype == torch.float32
        assert getattr(qkv, WEIGHT_FLIPPED_ATTR, False) is True

        # Re-running is idempotent.
        assert fp8_linear.quantize_wan_model_linears(model) == 0

        # Skip edge blocks skips blocks.0 and blocks.2 (leaves only block 1 -> 8).
        model_edge = _make_dummy_wan_model(num_blocks=3)
        assert (
            fp8_linear.quantize_wan_model_linears(model_edge, skip_edge_blocks=True)
            == 8
        )
        assert not getattr(model_edge.blocks[0].attn1.to_qkv, "_wan_fp8_active", False)
        assert getattr(model_edge.blocks[1].attn1.to_qkv, "_wan_fp8_active", False)
        assert model_edge.blocks[1].attn1.to_qkv._wan_fp8_scale.shape == (768,)
        assert not getattr(model_edge.blocks[2].attn1.to_qkv, "_wan_fp8_active", False)

        # quantize_cross_kv=False skips attn2.to_k and attn2.to_v (6 per block).
        model_no_kv = _make_dummy_wan_model(num_blocks=3)
        assert (
            fp8_linear.quantize_wan_model_linears(model_no_kv, quantize_cross_kv=False)
            == 18
        )
        assert not getattr(model_no_kv.blocks[0].attn2.to_k, "_wan_fp8_active", False)
        assert not getattr(model_no_kv.blocks[0].attn2.to_v, "_wan_fp8_active", False)
        assert getattr(model_no_kv.blocks[0].attn2.to_q, "_wan_fp8_active", False)
        assert getattr(model_no_kv.blocks[0].attn2.to_out, "_wan_fp8_active", False)

        # exclude filters dot-prefixes ("attn2") and exact suffixes ("ffn.net_2").
        model_excl = _make_dummy_wan_model(num_blocks=3)
        assert (
            fp8_linear.quantize_wan_model_linears(
                model_excl,
                exclude=frozenset({"attn2", "attn1.to_out", "ffn.net_2"}),
            )
            == 6
        )
        assert getattr(model_excl.blocks[0].attn1.to_qkv, "_wan_fp8_active", False)
        assert getattr(model_excl.blocks[0].ffn.net_0.proj, "_wan_fp8_active", False)
        assert not getattr(model_excl.blocks[0].attn1.to_out, "_wan_fp8_active", False)
        assert not getattr(model_excl.blocks[0].attn2.to_q, "_wan_fp8_active", False)
        assert not getattr(model_excl.blocks[0].ffn.net_2, "_wan_fp8_active", False)

        # KEEP_VLLM_LAYOUT_ATTR is respected.
        model_keep = _make_dummy_wan_model(num_blocks=1)
        setattr(model_keep.blocks[0].attn1.to_out, KEEP_VLLM_LAYOUT_ATTR, True)
        assert fp8_linear.quantize_wan_model_linears(model_keep) == 7

        # Works cleanly inside torch.inference_mode() (actual Wan22Pipeline context)
        # and creates non-inference quantized parameters/buffers.
        model_inf = _make_dummy_wan_model(num_blocks=1)
        with torch.inference_mode():
            assert fp8_linear.quantize_wan_model_linears(model_inf) == 8
            qkv_inf = model_inf.blocks[0].attn1.to_qkv
            assert getattr(qkv_inf, "_wan_fp8_active", False)
            assert not qkv_inf.weight.is_inference()
            assert not qkv_inf._wan_fp8_scale.is_inference()


def test_lazy_forward_hook_covers_both_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_forward = WanTransformer3DModel.forward
    try:
        base_calls: list[int] = []

        def _fake_base_forward(self: Any, hidden_states: Any, timestep: Any) -> Any:
            del timestep
            base_calls.append(id(self))
            return hidden_states

        WanTransformer3DModel.forward = _fake_base_forward
        monkeypatch.setenv(fp8_linear.ENV, "1")
        assert fp8_linear.apply() is True

        quant_calls: list[int] = []
        monkeypatch.setattr(
            fp8_linear,
            "quantize_wan_model_linears",
            lambda model, **kw: quant_calls.append(id(model)) or 8,
        )

        class _DummyExpert:
            forward = WanTransformer3DModel.forward

        e1 = _DummyExpert()
        e2 = _DummyExpert()

        x = torch.zeros(1)
        t = torch.zeros(1)
        assert e1.forward(x, t) is x
        assert e1.forward(x, t) is x
        assert e2.forward(x, t) is x

        assert getattr(e1, "_wan_fp8_initialized", False) is True
        assert getattr(e2, "_wan_fp8_initialized", False) is True
        assert quant_calls == [id(e1), id(e2)]
        assert base_calls == [id(e1), id(e1), id(e2)]
    finally:
        WanTransformer3DModel.forward = orig_forward


def test_skip_high_noise_and_exclude_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion import forward_context

    orig_forward = WanTransformer3DModel.forward
    try:
        WanTransformer3DModel.forward = lambda self, hidden_states, timestep: (
            hidden_states
        )
        monkeypatch.setenv(fp8_linear.ENV, "1")
        monkeypatch.setenv(fp8_linear.SKIP_HIGH_NOISE_ENV, "1")
        monkeypatch.setenv(fp8_linear.EXCLUDE_ENV, "attn2, ffn.net_2")
        assert fp8_linear.apply() is True

        ctx = SimpleNamespace(denoise_step_idx=0)
        monkeypatch.setattr(
            forward_context, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(forward_context, "get_forward_context", lambda: ctx)

        quant_calls: list[tuple[int, frozenset[str]]] = []
        monkeypatch.setattr(
            fp8_linear,
            "quantize_wan_model_linears",
            lambda model, **kw: quant_calls.append((id(model), kw["exclude"])) or 6,
        )

        class _DummyExpert:
            forward = WanTransformer3DModel.forward

        high_noise = _DummyExpert()
        low_noise = _DummyExpert()
        x = torch.zeros(1)
        t = torch.zeros(1)

        ctx.denoise_step_idx = 0
        assert high_noise.forward(x, t) is x
        ctx.denoise_step_idx = 20
        assert low_noise.forward(x, t) is x

        assert getattr(high_noise, "_wan_fp8_initialized", False) is True
        assert getattr(low_noise, "_wan_fp8_initialized", False) is True
        assert quant_calls == [(id(low_noise), frozenset({"attn2", "ffn.net_2"}))]
    finally:
        WanTransformer3DModel.forward = orig_forward


def test_ulysses_pipe_flash_spec_key_tracks_fork_op_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op_dense = lambda q, k, v, scale, bq, bkv, bkvc: q  # noqa: E731
    op_sparse = lambda q, k, v, scale, bq, bkv, bkvc: k  # noqa: E731
    current_op = op_dense

    def _mock_resolve(seq_len: int, head_dim: int, scale: float) -> Any:
        del seq_len, head_dim
        return SimpleNamespace(
            op=current_op,
            block_q=4736,
            block_kv=2048,
            block_kv_compute=2048,
            scale=scale,
            cache_key=("fork", id(current_op), 4736, 2048, 2048, scale),
        )

    try:
        from vllm_omni_torchtpu.wan_opt import attn_kernel
    except ImportError:
        attn_kernel = SimpleNamespace()
        monkeypatch.setitem(
            sys.modules, "vllm_omni_torchtpu.wan_opt.attn_kernel", attn_kernel
        )

    monkeypatch.setattr(
        attn_kernel, "resolve_compiled_flash_op", _mock_resolve, raising=False
    )

    def _fake_attn_forward(*args: Any, **kwargs: Any) -> Any:
        return None

    _fake_attn_forward._tpu_wan_attn_fork_patched = True
    attn_mod = SimpleNamespace(
        attn=SimpleNamespace(
            attention=SimpleNamespace(
                use_sdpa=False,
                softmax_scale=0.08838834764831845,
                forward=_fake_attn_forward,
            )
        )
    )

    spec_dense = ulysses_pipe._resolve_flash_spec(attn_mod, 75600, 128, torch.bfloat16)
    current_op = op_sparse
    spec_sparse = ulysses_pipe._resolve_flash_spec(attn_mod, 75600, 128, torch.bfloat16)
    assert spec_dense is not None and spec_sparse is not None
    assert spec_dense[0] != spec_sparse[0]
    assert id(op_dense) in spec_dense[0][1]
    assert id(op_sparse) in spec_sparse[0][1]

    # When resolve_compiled_flash_op returns None, _resolve_flash_spec returns None.
    monkeypatch.setattr(
        attn_kernel,
        "resolve_compiled_flash_op",
        lambda seq_len, head_dim, scale: None,
        raising=False,
    )
    assert (
        ulysses_pipe._resolve_flash_spec(attn_mod, 75600, 128, torch.bfloat16) is None
    )


def test_linear_base_forward_return_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(linear_common, "quantized_matmul", _cpu_quantized_matmul_ref)
    with _single_rank_tp_group():
        gen = torch.Generator().manual_seed(10)
        x = torch.randn(2, 16, 256, generator=gen, dtype=torch.bfloat16)

        qkv = QKVParallelLinear(
            hidden_size=256,
            head_size=128,
            total_num_heads=2,
            bias=True,
            params_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            qkv.weight.copy_(torch.randn_like(qkv.weight) * 0.05)
            qkv.bias.copy_(torch.randn_like(qkv.bias) * 0.05)
        ref_qkv, ref_bias = qkv(x)
        assert ref_bias is None
        assert fp8_linear.quantize_linear_to_fp8(qkv)
        got_qkv, got_bias = qkv(x)
        assert got_bias is None
        assert_parity(got_qkv, ref_qkv, name="qkv_return_bias", max_rel_l2=5e-2)

        qkv_skip = QKVParallelLinear(
            hidden_size=256,
            head_size=128,
            total_num_heads=2,
            bias=True,
            skip_bias_add=True,
            params_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            qkv_skip.weight.copy_(torch.randn_like(qkv_skip.weight) * 0.05)
            qkv_skip.bias.copy_(torch.randn_like(qkv_skip.bias) * 0.05)
        ref_skip, ref_skip_b = qkv_skip(x)
        assert ref_skip_b is qkv_skip.bias
        assert fp8_linear.quantize_linear_to_fp8(qkv_skip)
        got_skip, got_skip_b = qkv_skip(x)
        assert got_skip_b is qkv_skip.bias
        assert_parity(got_skip, ref_skip, name="qkv_skip_bias", max_rel_l2=5e-2)

        row = RowParallelLinear(
            256,
            256,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            params_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            row.weight.copy_(torch.randn_like(row.weight) * 0.05)
            row.bias.copy_(torch.randn_like(row.bias) * 0.05)
        ref_row = row(x)
        assert isinstance(ref_row, torch.Tensor)
        assert fp8_linear.quantize_linear_to_fp8(row)
        got_row = row(x)
        assert isinstance(got_row, torch.Tensor)
        assert_parity(got_row, ref_row, name="row_no_return_bias", max_rel_l2=5e-2)


def test_ulysses_pipe_accepts_fp8_contract_and_rejects_4d_scale() -> None:
    with _single_rank_tp_group():
        dim, heads, head_dim = 512, 4, 128
        attn = object.__new__(WanSelfAttention)
        torch.nn.Module.__init__(attn)
        attn.dim = dim
        attn.num_heads = heads
        attn.num_kv_heads = heads
        attn.head_dim = head_dim
        attn.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=heads,
            bias=True,
            params_dtype=torch.bfloat16,
        )
        attn.to_out = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            params_dtype=torch.bfloat16,
        )
        attn.norm_q = SimpleNamespace(variance_epsilon=1e-6, weight=torch.ones(dim))
        attn.norm_k = SimpleNamespace(variance_epsilon=1e-6, weight=torch.ones(dim))
        attn.dropout = torch.nn.Identity()
        attn.attn = SimpleNamespace(
            use_sync=False,
            attention=SimpleNamespace(
                forward=_sdpa_impl_forward,
                _op=lambda q, k, v, scale: q,
                softmax_scale=head_dim**-0.5,
            ),
        )
        sp_group = SimpleNamespace(
            ulysses_world_size=4,
            ring_world_size=1,
            allgather_world_size=1,
            ulysses_group="fake_pg",
        )
        x = torch.randn(1, 8, dim, dtype=torch.bfloat16)

        assert fp8_linear.quantize_linear_to_fp8(attn.to_qkv)
        assert fp8_linear.quantize_linear_to_fp8(attn.to_out)
        with torch.no_grad():
            assert (
                ulysses_pipe._get_or_compile_pipe_fn(attn, x, sp_group, 1) is not None
            )

        # Legacy 4-D block128 scale buffers are rejected cleanly.
        attn.to_out._wan_fp8_scale = torch.ones(1, 4, 1, dim, dtype=torch.float32)
        assert not ulysses_pipe._is_fp8_linear(
            attn.to_out, in_features=heads * head_dim, out_features=dim
        )


def _reference_qk_norm_rope(qkv, wq, wk, cos, sin, eps, head_dim):
    dim = qkv.shape[1] // 3
    even = torch.arange(0, head_dim, 2)
    swap = torch.zeros(head_dim, head_dim, dtype=torch.float32)
    swap[even + 1, even] = -1.0
    swap[even, even + 1] = 1.0
    swap = swap.to(device=qkv.device, dtype=qkv.dtype)
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
    return (
        F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        .transpose(1, 2)
        .contiguous()
    )


def _gloo_fp8_worker(rank: int, port: int, world_size: int) -> None:
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    current_platform.device_type = "cpu"
    current_platform.device_name = "cpu"
    current_platform.is_out_of_tree = lambda: False
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        initialize_model_parallel()
        pg = dist.group.WORLD
        sp_group = SimpleNamespace(
            ulysses_group=pg,
            ulysses_world_size=world_size,
            ring_world_size=1,
            allgather_world_size=1,
        )

        # Monkeypatch quantized_matmul for CPU execution in linear_common.
        linear_common.quantized_matmul = _cpu_quantized_matmul_ref
        omni_rope._qk_norm_rope_op = lambda: _reference_qk_norm_rope

        dim, heads, head_dim, seq_local = 1024, 8, 128, 8
        for with_bias in (True, False):
            attn = object.__new__(WanSelfAttention)
            torch.nn.Module.__init__(attn)
            attn.dim = dim
            attn.num_heads = heads
            attn.num_kv_heads = heads
            attn.head_dim = head_dim

            gen_w = torch.Generator().manual_seed(17)
            attn.to_qkv = QKVParallelLinear(
                hidden_size=dim,
                head_size=head_dim,
                total_num_heads=heads,
                bias=with_bias,
                params_dtype=torch.bfloat16,
            )
            attn.to_out = RowParallelLinear(
                dim,
                dim,
                bias=with_bias,
                input_is_parallel=True,
                return_bias=False,
                params_dtype=torch.bfloat16,
            )
            with torch.no_grad():
                attn.to_qkv.weight.copy_(
                    torch.randn(attn.to_qkv.weight.shape, generator=gen_w) * 0.03
                )
                attn.to_out.weight.copy_(
                    torch.randn(attn.to_out.weight.shape, generator=gen_w) * 0.03
                )
                if with_bias:
                    attn.to_qkv.bias.copy_(
                        torch.randn(attn.to_qkv.bias.shape, generator=gen_w) * 0.02
                    )
                    attn.to_out.bias.copy_(
                        torch.randn(attn.to_out.bias.shape, generator=gen_w) * 0.02
                    )
            attn.norm_q = RMSNorm(dim, eps=1e-6).to(torch.bfloat16)
            attn.norm_k = RMSNorm(dim, eps=1e-6).to(torch.bfloat16)
            attn.dropout = torch.nn.Identity()
            attn.rotary_embedding = RotaryEmbeddingWan(
                is_neox_style=False, half_head_dim=True
            )
            attn.to_gate_compress = None
            attn.attn = SimpleNamespace(
                attention=SimpleNamespace(
                    forward=_sdpa_impl_forward,
                    _op=lambda q, k, v, scale: _sdpa_impl_forward(q, k, v),
                    softmax_scale=head_dim**-0.5,
                ),
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

            gen_in = torch.Generator().manual_seed(99)
            s_glob = world_size * seq_local
            x_loc = torch.randn(1, s_glob, dim, generator=gen_in, dtype=torch.bfloat16)[
                :, rank * seq_local : (rank + 1) * seq_local
            ].contiguous()
            cos_loc = torch.randn(
                1, s_glob, 1, head_dim // 2, generator=gen_in, dtype=torch.bfloat16
            )[:, rank * seq_local : (rank + 1) * seq_local].contiguous()
            sin_loc = torch.randn(
                1, s_glob, 1, head_dim // 2, generator=gen_in, dtype=torch.bfloat16
            )[:, rank * seq_local : (rank + 1) * seq_local].contiguous()

            with torch.no_grad():
                assert (
                    ulysses_pipe._get_or_compile_pipe_fn(attn, x_loc, sp_group, 2)
                    is not None
                )
                ref_bf16 = ulysses_pipe._run_chunked_ulysses_self_attention(
                    attn,
                    x_loc,
                    (cos_loc, sin_loc),
                    sp_group,
                    2,
                )
                assert fp8_linear.quantize_linear_to_fp8(attn.to_qkv)
                assert fp8_linear.quantize_linear_to_fp8(attn.to_out)
                assert (
                    ulysses_pipe._get_or_compile_pipe_fn(attn, x_loc, sp_group, 2)
                    is not None
                )
                got_fp8 = ulysses_pipe._run_chunked_ulysses_self_attention(
                    attn,
                    x_loc,
                    (cos_loc, sin_loc),
                    sp_group,
                    2,
                )
                assert_parity(
                    got_fp8,
                    ref_bf16,
                    name=f"gloo_rank{rank}_per_channel_bias{with_bias}",
                    max_rel_l2=1e-1,
                )

        destroy_model_parallel()
        destroy_distributed_environment()


def test_gloo_fp8_ulysses_pipe_parity() -> None:
    world_size = 4
    port = _free_port()
    mp.spawn(_gloo_fp8_worker, args=(port, world_size), nprocs=world_size, join=True)
