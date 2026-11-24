#!/bin/bash
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
set -euo pipefail

usage() {
  cat <<USAGE
Usage: ./docker/build_image.sh [options]

Builds a vllm-omni-torchtpu image from docker/Dockerfile with the repository
root as the docker context.

Options:
  -t, --image-tag TAG    Tag for the built image (default: vllm-omni-torchtpu-local)
      --target TARGET    Dockerfile stage: prod, dev, or omni (default: prod)
      --base-image REF   Image the stage is layered on:
                           prod  vllm/vllm-torchtpu:vX.Y.Z (default: vllm/vllm-torchtpu:latest)
                           dev   vllm/vllm-torchtpu:nightly (the default) or a dev image
                           omni  the Dockerfile's pinned nightly unless given
      --version VERSION  vllm-omni-torchtpu version stamped on the prod
                         install (default: 0.0.0)
      --core-commit REF  dev: vllm-torchtpu commit, tag or branch to reinstall
                         vllm-torchtpu-core from (default: main's HEAD)
      --core-version V   dev: version stamped on that core build (default:
                         the Dockerfile's CORE_VERSION)
  -h, --help             Show this help and exit

Targets:
  prod   vllm-omni-torchtpu installed (not editable) on a vllm-torchtpu release
         image, without test tooling. What Docker Hub serves as vX.Y.Z.
  dev    the omni stage on an unpinned vllm-torchtpu dev image, with core at
         --core-commit and the base image's vLLM. What Docker Hub serves as
         nightly.
  omni   the CI image, exactly as .buildkite/scripts/setup_docker_env.sh
         builds it (core at CORE_COMMIT, vLLM at the pyproject.toml pin).
USAGE
}

IMAGE_TAG="vllm-omni-torchtpu-local"
TARGET="prod"
BASE_IMAGE_REF=""
VERSION="0.0.0"
CORE_COMMIT=""
CORE_VERSION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --base-image)
      BASE_IMAGE_REF="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --core-commit)
      CORE_COMMIT="$2"
      shift 2
      ;;
    --core-version)
      CORE_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOCKER_ARGS=(
  --progress=plain
  -f "${SCRIPT_DIR}/Dockerfile"
  -t "${IMAGE_TAG}"
  --target "${TARGET}"
)
case "${TARGET}" in
  prod)
    DOCKER_ARGS+=(
      --build-arg "PROD_BASE_IMAGE=${BASE_IMAGE_REF:-vllm/vllm-torchtpu:latest}"
      --build-arg "OMNI_VERSION=${VERSION}"
    )
    ;;
  dev)
    # The omni stage reinstalls vllm-torchtpu-core from git at CORE_COMMIT
    # (the base dev image's own editable core is stamped 0.0.0, below this
    # package's floor). Default: vllm-torchtpu main, which the nightly base
    # was built from. Empty VLLM_COMMIT_HASH keeps the base image's vLLM.
    if [[ -z "${CORE_COMMIT}" ]]; then
      CORE_COMMIT="$(git ls-remote https://github.com/vllm-project/vllm-torchtpu.git refs/heads/main | cut -f1)"
      [[ -n "${CORE_COMMIT}" ]] || { echo "Could not resolve vllm-torchtpu main." >&2; exit 1; }
    fi
    DOCKER_ARGS+=(
      --build-arg "BASE_IMAGE=${BASE_IMAGE_REF:-vllm/vllm-torchtpu:nightly}"
      --build-arg "CORE_COMMIT=${CORE_COMMIT}"
      --build-arg "VLLM_COMMIT_HASH="
    )
    if [[ -n "${CORE_VERSION}" ]]; then
      DOCKER_ARGS+=(--build-arg "CORE_VERSION=${CORE_VERSION}")
    fi
    ;;
  omni)
    VLLM_COMMIT_HASH="$(sed -nE 's/.*vllm @ git\+https:\/\/github\.com\/vllm-project\/vllm\.git@([a-zA-Z0-9.-]+).*/\1/p' "${REPO_ROOT}/pyproject.toml" | head -n 1)"
    DOCKER_ARGS+=(--build-arg "VLLM_COMMIT_HASH=${VLLM_COMMIT_HASH}")
    if [[ -n "${BASE_IMAGE_REF}" ]]; then
      DOCKER_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE_REF}")
    fi
    ;;
  *)
    echo "Unknown target '${TARGET}'; expected prod, dev, or omni." >&2
    exit 1
    ;;
esac

echo "===> Building ${IMAGE_TAG} (target: ${TARGET})"
DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" "${REPO_ROOT}"
echo "===> Done!"
