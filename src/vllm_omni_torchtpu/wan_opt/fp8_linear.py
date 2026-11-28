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
"""P3-C: Opt-in per-channel FP8 W8A8 dense linear quantization for Wan2.2 DiT.

Gated on ``VLLM_OMNI_WAN_FP8_LINEAR=1`` (default off). Lazily quantizes the
transformer block ``LinearBase`` layers (``attn1.{to_qkv,to_out}``,
``attn2.{to_q,to_k,to_v,to_out}``, ``ffn.{net_0.proj,net_2}``) on each
``WanTransformer3DModel`` expert's first forward call, keeping norms,
modulation, embeddings, ``proj_out``, and VAE in BF16/FP32.

Quantized layers expose the duck-type contract ``_wan_fp8_active = True``,
K-major ``(K_in, N_out)`` ``float8_e4m3fn`` ``weight``, and 1-D FP32
``_wan_fp8_scale`` buffer so compiled attention graphs (``ulysses_pipe``) and
eager ``LinearBase`` forwards both dispatch to
``vllm_torchtpu.layers.adapter.linear_common.quantized_matmul`` with zero graph
breaks.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm_torchtpu.envs import env_bool
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_OMNI_WAN_FP8_LINEAR"
SKIP_EDGE_BLOCKS_ENV = "VLLM_OMNI_WAN_FP8_SKIP_EDGE_BLOCKS"
CROSS_KV_ENV = "VLLM_OMNI_WAN_FP8_CROSS_KV"
EXCLUDE_ENV = "VLLM_OMNI_WAN_FP8_EXCLUDE"
SKIP_HIGH_NOISE_ENV = "VLLM_OMNI_WAN_FP8_SKIP_HIGH_NOISE"

_FP8_LINEAR_PATCHED = "_tpu_wan_fp8_linear_patched"
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_E4M3_MAX = float(torch.finfo(_FP8_DTYPE).max)

_BASE_TARGET_SUFFIXES = frozenset(
    {
        "attn1.to_qkv",
        "attn1.to_out",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out",
        "ffn.net_0.proj",
        "ffn.net_2",
    }
)
_CROSS_KV_SUFFIXES = frozenset({"attn2.to_k", "attn2.to_v"})


def _import_linear_common() -> Any:
    """Import ``linear_common`` lazily with an explicit version-compat error."""
    try:
        linear_common = importlib.import_module(
            "vllm_torchtpu.layers.adapter.linear_common"
        )
    except ImportError as exc:
        raise ImportError(
            "wan_opt.fp8_linear requires "
            "vllm_torchtpu.layers.adapter.linear_common; upgrade "
            "vllm-torchtpu-core."
        ) from exc
    for attr in (
        "KEEP_VLLM_LAYOUT_ATTR",
        "WEIGHT_FLIPPED_ATTR",
        "_get_quantized_matmul_op",
        "quantized_matmul",
    ):
        if not hasattr(linear_common, attr):
            raise ImportError(
                f"vllm_torchtpu.layers.adapter.linear_common is missing {attr!r}; "
                "upgrade vllm-torchtpu-core to use VLLM_OMNI_WAN_FP8_LINEAR."
            )
    return linear_common


def _resolve_exclude_tokens() -> frozenset[str]:
    raw = os.environ.get(EXCLUDE_ENV, "")
    return frozenset(tok.strip() for tok in raw.split(",") if tok.strip())


def _is_excluded_suffix(suffix: str, exclude: frozenset[str]) -> bool:
    return any(suffix == tok or suffix.startswith(f"{tok}.") for tok in exclude)


def _is_high_noise_step() -> bool:
    try:
        from vllm_omni.diffusion.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except ImportError:
        return False
    if not is_forward_context_available():
        return False
    return get_forward_context().denoise_step_idx == 0


class _WanFp8LinearMethod(QuantizeMethodBase):
    """Runtime FP8 W8A8 matmul method for quantized Wan2.2 DiT linears."""

    def __init__(self, matmul_fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self._quantized_matmul = matmul_fn

    def create_weights(
        self,
        layer: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del layer, args, kwargs

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self._quantized_matmul(
            x, layer.weight, layer._wan_fp8_scale, w8a16_max_rows=0
        )
        if bias is not None:
            out = out + (bias if bias.dtype == out.dtype else bias.to(out.dtype))
        return out


def quantize_weight_fp8(
    weight: torch.Tensor,
    *,
    flipped: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D linear weight to K-major ``float8_e4m3fn`` and FP32 scale.

    Args:
        weight: 2-D tensor in ``(K_in, N_out)`` if ``flipped`` else
            ``(N_out, K_in)``.
        flipped: Whether ``weight`` is already transposed to K-major layout.

    Returns:
        Tuple ``(w_q, w_s)`` where ``w_q`` is contiguous ``(K_in, N_out)``
        ``torch.float8_e4m3fn`` and ``w_s`` is contiguous 1-D ``(N_out,)``
        ``torch.float32``.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D weight tensor, got shape {tuple(weight.shape)}")

    w_kn = weight.detach() if flipped else weight.detach().transpose(0, 1).contiguous()
    w_f32 = w_kn.to(torch.float32)
    inv_max = 1.0 / _FP8_E4M3_MAX

    w_s = (w_f32.abs().amax(dim=0) * inv_max).contiguous()
    scale_inv = torch.where(w_s == 0, torch.zeros_like(w_s), 1.0 / w_s)
    w_q = (
        torch.clamp(w_f32 * scale_inv.unsqueeze(0), -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(_FP8_DTYPE)
        .contiguous()
    )
    return w_q, w_s


def _parse_block_linear_name(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "blocks" or not parts[1].isdigit():
        return None
    return int(parts[1]), ".".join(parts[2:])


def quantize_linear_to_fp8(layer: torch.nn.Module) -> bool:
    """Convert a single ``LinearBase`` layer in-place to per-channel FP8 W8A8.

    Args:
        layer: Candidate module to quantize in-place.

    Returns:
        True if ``layer`` was quantized, False if skipped or already active.
    """
    if not isinstance(layer, LinearBase):
        return False
    if getattr(layer, "_wan_fp8_active", False):
        return False
    linear_common = _import_linear_common()
    if getattr(layer, linear_common.KEEP_VLLM_LAYOUT_ATTR, False):
        return False
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return False
    if weight.dtype in (_FP8_DTYPE, torch.float8_e5m2, torch.int8):
        return False

    flipped = bool(getattr(layer, linear_common.WEIGHT_FLIPPED_ATTR, False))
    with torch.inference_mode(False), torch.no_grad():
        w_q, w_s = quantize_weight_fp8(weight.data, flipped=flipped)

        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
        layer.register_buffer("_wan_fp8_scale", w_s, persistent=False)
        layer._wan_fp8_active = True
        setattr(layer, linear_common.WEIGHT_FLIPPED_ATTR, True)
        layer.quant_method = _WanFp8LinearMethod(linear_common.quantized_matmul)
    return True


def quantize_wan_model_linears(
    model: torch.nn.Module,
    *,
    skip_edge_blocks: bool = False,
    quantize_cross_kv: bool = True,
    exclude: frozenset[str] = frozenset(),
) -> int:
    """Quantize eligible Wan block dense linears in ``model`` in-place.

    Args:
        model: ``WanTransformer3DModel`` (or compatible module with ``blocks``).
        skip_edge_blocks: Skip first and last transformer blocks when True.
        quantize_cross_kv: Whether to quantize ``attn2.to_k`` and ``attn2.to_v``.
        exclude: Suffixes or dot-prefixes to keep in BF16.

    Returns:
        Number of ``LinearBase`` layers quantized.
    """
    linear_common = _import_linear_common()
    allowed_suffixes = (
        _BASE_TARGET_SUFFIXES
        if quantize_cross_kv
        else _BASE_TARGET_SUFFIXES - _CROSS_KV_SUFFIXES
    )
    if exclude:
        allowed_suffixes = frozenset(
            s for s in allowed_suffixes if not _is_excluded_suffix(s, exclude)
        )
    blocks = getattr(model, "blocks", None)
    last_block_idx = len(blocks) - 1 if blocks is not None and len(blocks) > 0 else None

    quantized_count = 0
    pending_sync: list[torch.Tensor] = []

    def _flush_tpu() -> None:
        if not pending_sync:
            return
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(pending_sync)
        pending_sync.clear()

    for name, module in model.named_modules():
        parsed = _parse_block_linear_name(name)
        if parsed is None:
            continue
        block_idx, suffix = parsed
        if suffix not in allowed_suffixes:
            continue
        if skip_edge_blocks and (block_idx == 0 or block_idx == last_block_idx):
            continue
        if quantize_linear_to_fp8(module):
            quantized_count += 1
            if module.weight.device.type == "tpu":
                pending_sync.append(module.weight)
                pending_sync.append(module._wan_fp8_scale)
                if len(pending_sync) >= 16:
                    _flush_tpu()

    _flush_tpu()
    if quantized_count > 0:
        # Eagerly register pallas::quantized_matmul_kernel outside any compiled
        # region so Block 0's to_qkv does not trace Lock / pallas.jax_op inside
        # torch.compile(fullgraph=True).
        linear_common._get_quantized_matmul_op()
    logger.info(
        "wan_opt.fp8_linear: quantized %d DiT LinearBase layers "
        "(skip_edge_blocks=%s, cross_kv=%s, exclude=%s).",
        quantized_count,
        skip_edge_blocks,
        quantize_cross_kv,
        sorted(exclude),
    )
    return quantized_count


def apply() -> bool:
    """Install lazy FP8 DiT linear quantization on ``WanTransformer3DModel``.

    Returns:
        True if the wrapper was installed, False when disabled or already active.
    """
    if not env_bool(ENV, False)():
        return False

    skip_edge_blocks = env_bool(SKIP_EDGE_BLOCKS_ENV, False)()
    quantize_cross_kv = env_bool(CROSS_KV_ENV, True)()
    exclude = _resolve_exclude_tokens()
    skip_high_noise = env_bool(SKIP_HIGH_NOISE_ENV, False)()

    linear_common = _import_linear_common()

    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanTransformer3DModel,
    )

    if getattr(WanTransformer3DModel.forward, _FP8_LINEAR_PATCHED, False):
        return False

    # Pre-warm pallas::quantized_matmul_kernel outside any compiled graph.
    linear_common._get_quantized_matmul_op()

    wrapped_forward = WanTransformer3DModel.forward

    def _fp8_linear_forward(
        self: Any,
        hidden_states: Any,
        timestep: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not torch.compiler.is_compiling() and not getattr(
            self, "_wan_fp8_initialized", False
        ):
            if skip_high_noise and _is_high_noise_step():
                logger.info("wan_opt.fp8_linear: skipping high-noise expert at step 0.")
            else:
                quantize_wan_model_linears(
                    self,
                    skip_edge_blocks=skip_edge_blocks,
                    quantize_cross_kv=quantize_cross_kv,
                    exclude=exclude,
                )
            self._wan_fp8_initialized = True
        return wrapped_forward(self, hidden_states, timestep, *args, **kwargs)

    setattr(_fp8_linear_forward, _FP8_LINEAR_PATCHED, True)
    WanTransformer3DModel.forward = _fp8_linear_forward
    logger.info(
        "Applied TPU patch: wan_opt FP8 DiT dense linears (skip_edge_blocks=%s, "
        "cross_kv=%s, exclude=%s, skip_high_noise=%s).",
        skip_edge_blocks,
        quantize_cross_kv,
        sorted(exclude),
        skip_high_noise,
    )
    return True
