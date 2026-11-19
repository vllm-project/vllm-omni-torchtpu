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

# Omni image for vllm-omni-torchtpu, layered on the upstream vllm-torchtpu
# developer image (`vllm/vllm-torchtpu:nightly` on Docker Hub, pinned by digest
# alongside `CORE_COMMIT` so CI builds stay reproducible).
#
# Build from the repository root:
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t vllm-omni-torchtpu:dev .
ARG BASE_IMAGE=vllm/vllm-torchtpu:nightly@sha256:43d090dac793f9dfa2fcf4339c915fed0d10438d618068436fc37a959df56ff9
FROM ${BASE_IMAGE} AS omni

# vllm-omni depends on opencv-python (not -headless), whose cv2 links libGL and
# glib; without them multi-rank diffusion workers crash at startup.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vllm-omni-torchtpu

# Reinstall vllm-torchtpu-core at the exact upstream commit this extract was cut
# from (`--no-deps` preserves all pre-installed base runtime pins). The upstream
# `dev` image installs core editable from `/workspace/vllm-torchtpu` and puts
# `/workspace/vllm-torchtpu/src` on PYTHONPATH, so remove that tree first to
# prevent an older checkout from shadowing the reinstalled package.
ARG CORE_COMMIT=a0275aad4f81a8befa40f8dc7a3fbeebaaff4355
RUN rm -rf /workspace/vllm-torchtpu && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.34.0.dev0 \
    uv pip install --system --no-cache --no-deps --reinstall-package vllm-torchtpu-core \
      "vllm-torchtpu-core @ git+https://github.com/vllm-project/vllm-torchtpu.git@${CORE_COMMIT}"

# Pre-install omni and test dependencies before copying the full tree so edits
# under src/ or tests_omni/ reuse this layer. `--group vllm` keeps vLLM's source
# pin in resolution so vllm-omni cannot replace it with a PyPI vllm wheel.
COPY pyproject.toml README.md LICENSE ./
ARG VLLM_COMMIT_HASH=""
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      -r pyproject.toml \
      --group vllm \
      --group test

COPY . .
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_OMNI_TORCHTPU=0.0.0 \
    uv pip install --system --no-deps -e . && \
    git config --global --add safe.directory /workspace/vllm-omni-torchtpu

# Select this package's omni and vLLM platform entry points over core's legacy
# `torchtpu` entry, plus core's custom layers and vllm-omni model registration.
ENV VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
ENV PYTHONPATH="/workspace/vllm-omni-torchtpu/src:/workspace/vllm-omni-torchtpu"
CMD ["/bin/bash"]
