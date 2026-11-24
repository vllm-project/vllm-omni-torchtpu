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
# developer image. Docker Hub publishes a dated per-commit tag for every
# nightly (`nightly-<date>-<vllm-torchtpu commit>-<vllm tag>`); the default
# pins one by digest alongside `CORE_COMMIT` so CI builds stay reproducible.
# Release tags such as `v0.33.0` are the slimmer `prod` variant without the
# test tooling (bktec, buildkite-agent, gcloud) the TPU pipeline needs, so
# override `BASE_IMAGE` (setup_docker_env.sh reads it from the environment)
# only with another `nightly-*` developer image.
#
# Build from the repository root:
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --target omni -t vllm-omni-torchtpu:dev .
#
# Stages (docker/build_image.sh --target <stage> wraps the build args):
#   omni  CI image (.buildkite/scripts/setup_docker_env.sh): core reinstalled
#         at CORE_COMMIT, test tools, this repository installed editable.
#   dev   the omni stage on an unpinned dev base (vllm/vllm-torchtpu:nightly),
#         with CORE_COMMIT set to that build's vllm-torchtpu commit and
#         VLLM_COMMIT_HASH empty, so it keeps the base image's vLLM. Published
#         to Docker Hub as vllm/vllm-omni-torchtpu:nightly.
#   prod  vllm-omni-torchtpu installed as a regular package on a vllm-torchtpu
#         release image (PROD_BASE_IMAGE, vllm/vllm-torchtpu:vX.Y.Z), without
#         test tooling. Published to Docker Hub as vllm/vllm-omni-torchtpu:vX.Y.Z.
ARG BASE_IMAGE=vllm/vllm-torchtpu:nightly-20261112-e9cf148-v0.33.0@sha256:43d090dac793f9dfa2fcf4339c915fed0d10438d618068436fc37a959df56ff9
ARG PROD_BASE_IMAGE=vllm/vllm-torchtpu:latest
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
# uv builds the commit without release tags, so setuptools-scm would stamp a
# version below the `>=0.33.0` floor and the next step would swap in the PyPI
# 0.33.0 wheel; the PRETEND version keeps this git build installed.
# The dev stage passes the commit (and version) of the vllm-torchtpu build its
# base image came from; see docker/build_image.sh.
ARG CORE_COMMIT=a0275aad4f81a8befa40f8dc7a3fbeebaaff4355
ARG CORE_VERSION=0.34.0.dev0
RUN rm -rf /workspace/vllm-torchtpu && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=${CORE_VERSION} \
    uv pip install --system --no-cache --no-deps --reinstall-package vllm-torchtpu-core \
      "vllm-torchtpu-core @ git+https://github.com/vllm-project/vllm-torchtpu.git@${CORE_COMMIT}"

# Pre-install omni and test dependencies before copying the full tree so edits
# under src/ or tests/ reuse this layer. The base image already ships vLLM, and
# nothing in this package's dependencies requires `vllm`, so a plain install
# leaves the base image's vLLM untouched. When `VLLM_COMMIT_HASH` is set (CI
# always passes it), the `vllm` group is added at that commit so vLLM's
# dependency pins constrain the resolution and a different commit is rebuilt
# from source (`VLLM_TARGET_DEVICE=empty` comes from the base image).
COPY pyproject.toml README.md LICENSE ./
ARG VLLM_COMMIT_HASH=""
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    set --; \
    if [ -n "$VLLM_COMMIT_HASH" ]; then set -- --group vllm; fi; \
    uv pip install --system \
      -r pyproject.toml \
      --group test \
      "$@"

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


# Developer image: the omni stage, built by docker/build_image.sh --target dev
# on an unpinned vllm-torchtpu dev image (nightly), and published to Docker Hub
# as vllm/vllm-omni-torchtpu:nightly by .buildkite/scripts/publish_nightly_images.sh.
FROM omni AS dev


# Production image: vllm-omni-torchtpu installed as a regular (non-editable)
# package on a vllm-torchtpu release image, without the test tooling (bktec,
# buildkite-agent, gcloud, pytest) of the omni and dev images. The release
# pipeline (.buildkite/release/) builds it from the release checkout before the
# wheel is on PyPI, stages it in GAR, and after approval copies it by digest to
# Docker Hub as vllm/vllm-omni-torchtpu:vX.Y.Z.
FROM ${PROD_BASE_IMAGE} AS prod

# vllm-omni depends on opencv-python (not -headless), whose cv2 links libGL and
# glib; without them multi-rank diffusion workers crash at startup.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Take torch, torchvision and torchaudio from the PyTorch CPU index (also set
# by the vllm-torchtpu base, and by [tool.uv.sources]): the PyPI torchaudio is
# a CUDA build that fails to load libcudart, which breaks every vLLM import.
ENV UV_TORCH_BACKEND=cpu

# The base image's vllm-torchtpu-core (the release this image pairs with) and
# vLLM stay as they are: the override pins core to whatever version the base
# reports (its source build may stamp 0.0.0, below the >=0.33.0 floor), so the
# dependency install adds only vllm-omni, torchaudio (CPU) and their
# dependencies. The package is built from a bind of the build context (writable
# for this step only), so no source tree is left in the image; the version
# check fails the build if the version stamp didn't apply.
ARG OMNI_VERSION=0.0.0
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,target=/tmp/vllm-omni-torchtpu,rw \
    cd /tmp/vllm-omni-torchtpu && \
    core="$(python3 -c 'import importlib.metadata as m; print(m.version("vllm-torchtpu-core"))')" && \
    echo "vllm-torchtpu-core==${core}" > /tmp/core-override.txt && \
    uv pip install --system -r pyproject.toml --overrides /tmp/core-override.txt && \
    rm /tmp/core-override.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_OMNI_TORCHTPU="${OMNI_VERSION}" \
    uv pip install --system --no-deps --reinstall-package vllm-omni-torchtpu . && \
    python3 -c "import importlib.metadata as m, sys; v = m.version('vllm-omni-torchtpu'); sys.exit(None if v == sys.argv[1] else f'vllm-omni-torchtpu {v} != {sys.argv[1]}')" "${OMNI_VERSION}"

WORKDIR /workspace
# Select this package's omni and vLLM platform entry points over core's legacy
# `torchtpu` entry, plus core's custom layers and vllm-omni model registration.
ENV VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models
CMD ["/bin/bash"]
