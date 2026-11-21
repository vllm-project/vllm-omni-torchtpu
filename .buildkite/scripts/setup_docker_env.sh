#!/usr/bin/env bash
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

TARGET="${1:-omni}"
if [ "${TARGET}" != "omni" ]; then
  echo "[FATAL] Unsupported target '${TARGET}'; expected omni." >&2
  exit 1
fi
IMAGE_SUFFIX="-omni"

IMAGE_REPO="${IMAGE_REPO:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-omni-torchtpu}"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"

VLLM_COMMIT_HASH=""
if command -v buildkite-agent >/dev/null 2>&1; then
  VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "" 2>/dev/null || true)"
fi
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  VLLM_COMMIT_HASH="$(sed -nE 's/.*vllm @ git\+https:\/\/github\.com\/vllm-project\/vllm\.git@([a-zA-Z0-9.-]+).*/\1/p' "${REPO_ROOT}/pyproject.toml" | head -n 1)"
fi
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  echo "[FATAL] VLLM_COMMIT_HASH could not be resolved from Buildkite metadata or pyproject.toml." >&2
  exit 1
fi

BASE_TAG="${COMMIT_HASH}-${VLLM_COMMIT_HASH}${IMAGE_SUFFIX}"
BASE_IMAGE_TAG="${IMAGE_REPO}:${BASE_TAG}"

if [[ -n "${CI_IMAGE:-}" ]]; then
  if ! docker manifest inspect "${CI_IMAGE}" >/dev/null 2>&1; then
    echo "[FATAL] CI_IMAGE=${CI_IMAGE} is not in the registry." >&2
    exit 1
  fi
  echo "--- Image given: ${CI_IMAGE}"
  buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX}" "${CI_IMAGE}"
  buildkite-agent meta-data set "CI_IMAGE_TAG" "${CI_IMAGE}"
  echo "--- Done setup_docker_env.sh"
  exit 0
fi

if [[ "${CI_IMAGE_REBUILD:-0}" != "1" ]] \
    && docker manifest inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
  echo "--- ${BASE_IMAGE_TAG} is already in the registry; reusing it"
  buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX}" "${BASE_IMAGE_TAG}"
  buildkite-agent meta-data set "CI_IMAGE_TAG" "${BASE_IMAGE_TAG}"
  echo "--- Done setup_docker_env.sh"
  exit 0
fi

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-torchtpu:nightly-20261112-e9cf148-v0.33.0@sha256:43d090dac793f9dfa2fcf4339c915fed0d10438d618068436fc37a959df56ff9}"

echo "--- Building Docker Image: ${BASE_IMAGE_TAG} (target: ${TARGET})"
DOCKER_ARGS=(
  --progress=plain
  -f "${REPO_ROOT}/docker/Dockerfile"
  --target "${TARGET}"
  -t "${BASE_IMAGE_TAG}"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "VLLM_COMMIT_HASH=${VLLM_COMMIT_HASH}"
)
if [ -n "${CORE_COMMIT:-}" ]; then
  DOCKER_ARGS+=(--build-arg "CORE_COMMIT=${CORE_COMMIT}")
fi

DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" "${REPO_ROOT}"

echo "--- Pushing Docker Image to Registry"
docker push "${BASE_IMAGE_TAG}"

buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX}" "${BASE_IMAGE_TAG}"
buildkite-agent meta-data set "CI_IMAGE_TAG" "${BASE_IMAGE_TAG}"

echo "--- Cleaning up local built image"
docker rmi "${BASE_IMAGE_TAG}" || true

echo "--- Done setup_docker_env.sh"
