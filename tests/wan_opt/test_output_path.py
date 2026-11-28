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
from __future__ import annotations

import pytest
import torch
from vllm_omni.diffusion import ipc
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.postprocess.media import finalize_diffusion_media
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_torchtpu.wan_opt import output_path

_ORIG_PACK = ipc._pack_tensor_if_large


@pytest.fixture
def clean_ipc(monkeypatch):
    monkeypatch.setattr(ipc, "_pack_tensor_if_large", _ORIG_PACK)
    if hasattr(ipc, "_tpu_wan_orig_pack_tensor"):
        monkeypatch.delattr(ipc, "_tpu_wan_orig_pack_tensor")
    yield ipc


def test_apply_gated_and_idempotent(clean_ipc, monkeypatch):
    monkeypatch.delenv(output_path.ENV, raising=False)
    assert output_path.apply() is False
    assert clean_ipc._pack_tensor_if_large is _ORIG_PACK

    monkeypatch.setenv(output_path.ENV, "0")
    assert output_path.apply() is False
    assert clean_ipc._pack_tensor_if_large is _ORIG_PACK

    monkeypatch.setenv(output_path.ENV, "1")
    assert output_path.apply() is True
    patched_fn = clean_ipc._pack_tensor_if_large
    assert patched_fn is not _ORIG_PACK
    assert getattr(patched_fn, output_path._OUTPUT_PATH_PATCHED, False) is True
    assert output_path.apply() is False
    assert clean_ipc._pack_tensor_if_large is patched_fn


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fast_shm_roundtrip_bit_identical(clean_ipc, monkeypatch, dtype):
    monkeypatch.setenv(output_path.ENV, "1")
    assert output_path.apply() is True

    torch.manual_seed(0)
    # Exceeds _SHM_TENSOR_THRESHOLD (1_000_000 bytes).
    video = torch.randn(1, 3, 17, 128, 128, dtype=dtype)
    expected = video.clone()

    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=video,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
        ),
        prepared_for_transport=True,
    )
    out = DiffusionOutput(media=media)
    packed = clean_ipc.pack_diffusion_output_shm(out)
    handle = packed.media["video"]["tensor"]
    assert handle["__tensor_shm__"] is True
    assert handle["borrow_on_unpack"] is True
    assert handle["nbytes"] == expected.numel() * expected.element_size()
    if dtype == torch.bfloat16:
        assert handle["numpy_dtype"] == "uint16"
        assert handle["torch_dtype_view"] is True

    unpacked = clean_ipc.unpack_diffusion_output_shm(packed)
    got_tensor = unpacked.media.video.tensor
    assert got_tensor.dtype == dtype
    assert torch.equal(got_tensor, expected)

    # Also verify finalize_diffusion_media produces bit-identical numpy frames.
    params = OmniDiffusionSamplingParams(output_type="np")
    ref_media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=expected,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
        ),
        prepared_for_transport=True,
    )
    got_np = finalize_diffusion_media(unpacked.media, sampling_params=params)[
        "payload"
    ]["video"]
    ref_np = finalize_diffusion_media(ref_media, sampling_params=params)["payload"][
        "video"
    ]
    assert got_np.dtype == ref_np.dtype
    assert (got_np == ref_np).all()


def test_small_tensor_and_non_contiguous_view(clean_ipc, monkeypatch):
    monkeypatch.setenv(output_path.ENV, "1")
    output_path.apply()

    small = torch.randn(2, 4, dtype=torch.bfloat16)
    res_small = clean_ipc._pack_tensor_if_large(small)
    assert isinstance(res_small, torch.Tensor)
    assert res_small.device.type == "cpu"
    assert torch.equal(res_small, small)

    # Non-contiguous slice larger than threshold must round-trip cleanly.
    big = torch.randn(1, 3, 17, 128, 130, dtype=torch.bfloat16)[..., :128]
    assert not big.is_contiguous()
    created: list[dict] = []
    handle = clean_ipc._pack_tensor_if_large(big, created=created)
    assert len(created) == 1
    restored = clean_ipc._tensor_from_shm(handle)
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored, big)


def test_unsupported_numpy_dtype_falls_back_to_orig_pack(clean_ipc, monkeypatch):
    fallback_calls: list[torch.Tensor] = []

    def _orig_pack(val, d2h_stream=None, created=None, *, enable_registered_shm=False):
        del d2h_stream, created, enable_registered_shm
        fallback_calls.append(val)
        return {"fallback": val.dtype}

    monkeypatch.setattr(clean_ipc, "_pack_tensor_if_large", _orig_pack)
    monkeypatch.setenv(output_path.ENV, "1")
    assert output_path.apply() is True

    fp8 = torch.empty(1024, 1024, dtype=torch.float8_e4m3fn)
    out = clean_ipc._pack_tensor_if_large(fp8)
    assert out == {"fallback": torch.float8_e4m3fn}
    assert fallback_calls == [fp8]


def test_direct_shm_unlinks_on_copy_error(clean_ipc, monkeypatch):
    from multiprocessing import shared_memory

    monkeypatch.setenv(output_path.ENV, "1")
    output_path.apply()

    created_names: list[str] = []
    orig_new_shm = clean_ipc._new_shm_array

    def _spy_new_shm(shape, dtype):
        shm, arr, handle = orig_new_shm(shape, dtype)
        created_names.append(handle["name"])
        return shm, arr, handle

    monkeypatch.setattr(clean_ipc, "_new_shm_array", _spy_new_shm)
    monkeypatch.setattr(
        torch.Tensor,
        "copy_",
        lambda self, src: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    big = torch.randn(1, 3, 17, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="boom"):
        clean_ipc._pack_tensor_if_large(big)

    assert len(created_names) == 1
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=created_names[0])


def test_engine_output_path_gated_and_bit_identical(monkeypatch):
    import numpy as np
    from vllm_omni.diffusion import diffusion_engine
    from vllm_omni.diffusion.media import (
        FloatVideoConsumer,
        VideoTransportConstraints,
    )
    from vllm_omni.diffusion.postprocess import media as media_mod

    from vllm_omni_torchtpu.wan_opt import engine_output_path

    orig_finalize = media_mod.finalize_diffusion_media
    monkeypatch.setattr(media_mod, "finalize_diffusion_media", orig_finalize)
    monkeypatch.setattr(diffusion_engine, "finalize_diffusion_media", orig_finalize)
    if hasattr(media_mod, "_tpu_wan_orig_finalize_diffusion_media"):
        monkeypatch.delattr(media_mod, "_tpu_wan_orig_finalize_diffusion_media")

    monkeypatch.delenv(engine_output_path.ENV, raising=False)
    assert engine_output_path.apply_engine() is False
    assert media_mod.finalize_diffusion_media is orig_finalize

    monkeypatch.setenv(engine_output_path.ENV, "1")
    assert engine_output_path.apply_engine() is True
    assert media_mod.finalize_diffusion_media is not orig_finalize
    assert (
        diffusion_engine.finalize_diffusion_media is media_mod.finalize_diffusion_media
    )

    torch.manual_seed(3)
    for dtype in (torch.bfloat16, torch.float32):
        for value_range in (
            VideoValueRange.NEGATIVE_ONE_TO_ONE,
            VideoValueRange.ZERO_TO_ONE,
        ):
            vid = torch.randn(1, 3, 5, 32, 32, dtype=dtype).clamp(-1.0, 1.0)
            media = DiffusionMediaOutput(
                video=VideoMediaOutput(
                    tensor=vid,
                    spec=VideoTensorSpec(
                        layout=VideoTensorLayout.BCTHW,
                        encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                        value_range=value_range,
                    ),
                ),
                prepared_for_transport=True,
            )
            params = OmniDiffusionSamplingParams(output_type="np")
            ref = orig_finalize(media, sampling_params=params)["payload"]["video"]
            got = media_mod.finalize_diffusion_media(media, sampling_params=params)[
                "payload"
            ]["video"]
            assert got.dtype == ref.dtype
            assert got.flags["C_CONTIGUOUS"]
            assert np.array_equal(got, ref)

    # Grad-enabled ZERO_TO_ONE tensor detaches cleanly before .numpy().
    grad_vid = torch.rand(1, 3, 3, 16, 16, dtype=torch.float32, requires_grad=True)
    grad_media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=grad_vid,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.ZERO_TO_ONE,
            ),
        ),
        prepared_for_transport=True,
    )
    grad_out = media_mod.finalize_diffusion_media(
        grad_media, sampling_params=OmniDiffusionSamplingParams(output_type="np")
    )["payload"]["video"]
    assert grad_out.shape == (1, 3, 16, 16, 3)

    # Non-np output_type or pending float consumers fall back to stock finalize.
    vid = torch.randn(1, 3, 3, 16, 16, dtype=torch.bfloat16)
    pil_media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=vid,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
        ),
        prepared_for_transport=True,
    )
    pil_params = OmniDiffusionSamplingParams(output_type="pil")
    pil_out = media_mod.finalize_diffusion_media(pil_media, sampling_params=pil_params)
    assert isinstance(pil_out["payload"]["video"], list)

    interp_media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=vid,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
            constraints=VideoTransportConstraints(
                pending_float_consumers=frozenset(
                    {FloatVideoConsumer.FRAME_INTERPOLATION}
                )
            ),
        ),
        prepared_for_transport=True,
    )
    with pytest.raises(AttributeError):
        # Falls back to orig_finalize (which calls RIFE with sampling_params).
        media_mod.finalize_diffusion_media(interp_media, sampling_params=None)
