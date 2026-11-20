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
"""TPU worker for autoregressive vLLM-Omni stages."""

from typing import Any

from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_torchtpu.worker.tpu_worker import TPUWorker

from vllm_omni_torchtpu.worker.base import OmniTPUWorkerBase
from vllm_omni_torchtpu.worker.tpu_ar_model_runner import TPUARModelRunner


def _require_core_model_runner_cls(worker_cls: type) -> None:
    """Fail fast if vllm-torchtpu-core predates TPUWorker.model_runner_cls."""
    if not hasattr(worker_cls, "model_runner_cls"):
        raise ImportError(
            "vllm-omni-torchtpu requires a vllm-torchtpu-core build that defines "
            "TPUWorker.model_runner_cls (vllm-project/vllm-torchtpu#2065, newer "
            "than 0.33.0). Install with `--group core-git` (see README) until "
            "vllm-torchtpu-core>=0.34.0 is on PyPI."
        )


class TPUARWorker(OmniWorkerMixin, OmniTPUWorkerBase):
    """TPU worker for autoregressive stages; the counterpart of ``GPUARWorker``.

    ``OmniTpuPlatform.get_omni_ar_worker_cls`` points vLLM-Omni here. The
    worker only swaps in the AR runner; ``TPUARModelRunner`` decides which
    stages it serves.

    The core version check runs when the worker is built, not at import,
    so the package still imports against vllm-torchtpu-core 0.33.0. Other
    stages do not use this worker.
    """

    model_runner_cls = TPUARModelRunner

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Older cores ignore model_runner_cls and would silently run the
        # default TPUModelRunner for AR stages.
        _require_core_model_runner_cls(TPUWorker)
        super().__init__(*args, **kwargs)
