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

# Build and publish the nightly vllm-omni-torchtpu image from main. Adapted
# from vllm-project/vllm-torchtpu .buildkite/scripts/publish_nightly_images.sh.
#
# Builds docker/Dockerfile --target dev on NIGHTLY_BASE_IMAGE (default
# vllm/vllm-torchtpu:nightly, pinned by digest first so the build can't race
# the upstream nightly), then pushes:
#   GAR         <NIGHTLY_REGISTRY>/vllm-omni-torchtpu-dev:nightly-YYYYMMDD
#               and :latest
#   Docker Hub  (PUBLISH_DOCKERHUB=1, set by pipeline_nightly_image.yml)
#               <DOCKERHUB_REPO>:nightly-YYYYMMDD-<commit>-<vllm> and :nightly,
#               where <vllm> is v<vLLM version> installed in the image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"

BRANCH="${BUILDKITE_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
[[ "${BRANCH}" == "main" ]] \
    || die "nightly images are published from main, not '${BRANCH}'."
COMMIT="${BUILDKITE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
DATE="$(date -u +%Y%m%d)"
PUBLISH_DOCKERHUB="${PUBLISH_DOCKERHUB:-0}"
REGISTRY="${NIGHTLY_REGISTRY:-${RELEASE_GAR}}"
IMAGE="${REGISTRY}/vllm-omni-torchtpu-dev"
DATE_IMAGE="${IMAGE}:nightly-${DATE}"
LATEST_IMAGE="${IMAGE}:latest"

if [[ "${PUBLISH_DOCKERHUB}" == "1" ]]; then
    DOCKERHUB_REPO="${DOCKERHUB_REPO:?DOCKERHUB_REPO must be set when PUBLISH_DOCKERHUB=1}"
    DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME must be set when PUBLISH_DOCKERHUB=1}"
    # Log in to a private, temporary Docker config that only the Docker Hub
    # pushes below use: the build still pulls the base image anonymously, the
    # token never reaches the agent's shared config, and the trap logs out
    # however the script ends.
    DOCKERHUB_CONFIG="$(mktemp -d)"
    cleanup_dockerhub() {
        DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker logout >/dev/null 2>&1 || true
        rm -rf "${DOCKERHUB_CONFIG}"
    }
    trap cleanup_dockerhub EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    dockerhub_login "${DOCKERHUB_CONFIG}" "${DOCKERHUB_USERNAME}"
fi

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

BASE_IMAGE="$(pin_by_digest "${NIGHTLY_BASE_IMAGE:-docker.io/vllm/vllm-torchtpu:nightly}")"
# vllm-torchtpu-core is reinstalled from git at the vllm-torchtpu main commit
# the nightly base was cut from (approximately: main's HEAD now).
CORE_COMMIT="${CORE_COMMIT:-$(git ls-remote https://github.com/vllm-project/vllm-torchtpu.git refs/heads/main | cut -f1)}"
[[ -n "${CORE_COMMIT}" ]] || die "could not resolve vllm-torchtpu main."
echo "--- :docker: Building ${DATE_IMAGE} on ${BASE_IMAGE} (core ${CORE_COMMIT})"
"${REPO_ROOT}/docker/build_image.sh" --target dev --base-image "${BASE_IMAGE}" \
    --core-commit "${CORE_COMMIT}" -t "${DATE_IMAGE}"
docker tag "${DATE_IMAGE}" "${LATEST_IMAGE}"

echo "--- :docker: Pushing ${DATE_IMAGE} and ${LATEST_IMAGE} to Artifact Registry"
docker push "${DATE_IMAGE}"
docker push "${LATEST_IMAGE}"

if [[ "${PUBLISH_DOCKERHUB}" == "1" ]]; then
    # The vLLM the image runs, read from its metadata without importing it
    # (no TPU here). Docker tags allow [A-Za-z0-9_.-] only, so a local
    # version such as 0.33.1.dev3+gabc1234 becomes 0.33.1.dev3-gabc1234.
    VLLM_VERSION="$(docker run --rm --entrypoint python3 "${DATE_IMAGE}" \
        -c 'import importlib.metadata as m; print(m.version("vllm"))')"
    VLLM_REF="v$(printf '%s' "${VLLM_VERSION}" | tr -c 'A-Za-z0-9_.-' '-')"
    DOCKERHUB_TAG="nightly-${DATE}-${COMMIT:0:7}-${VLLM_REF}"
    DOCKERHUB_TAG="${DOCKERHUB_TAG:0:128}"
    for DOCKERHUB_IMAGE in "${DOCKERHUB_REPO}:${DOCKERHUB_TAG}" "${DOCKERHUB_REPO}:nightly"; do
        echo "--- :docker: Pushing ${DOCKERHUB_IMAGE} to Docker Hub"
        docker tag "${DATE_IMAGE}" "${DOCKERHUB_IMAGE}"
        DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker push "${DOCKERHUB_IMAGE}"
        docker rmi "${DOCKERHUB_IMAGE}" >/dev/null 2>&1 || true
    done
    buildkite-agent annotate --style success --context nightly-dockerhub \
        "Published \`${DOCKERHUB_REPO}:${DOCKERHUB_TAG}\` and \`:nightly\` (base \`${BASE_IMAGE}\`)."
fi

echo "--- Cleaning up locally built image tags"
docker rmi "${DATE_IMAGE}" "${LATEST_IMAGE}" >/dev/null 2>&1 || true

echo "--- Successfully built and published the nightly image!"
