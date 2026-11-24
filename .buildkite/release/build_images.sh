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

# Build the prod (and dev) images from the release checkout and stage them in
# GAR as <version> and <date>-<version>. Nothing goes to Docker Hub here: the
# publish step copies the staged prod image there by digest, so users get the
# exact bytes this build made.
#
#   prod  docker/Dockerfile --target prod on vllm/vllm-torchtpu:v<torchtpu>
#         (pinned by digest in preflight.sh), with this checkout installed as
#         vllm-omni-torchtpu==<version>. It installs from source, not from
#         PyPI, so it builds before the approval and the wheel upload.
#   dev   docker/Dockerfile --target dev on the vllm-torchtpu release's staged
#         dev image (<RELEASE_GAR>/vllm-torchtpu-dev:<torchtpu>), GAR only.
#         Skipped with a warning when that image isn't staged.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
TORCHTPU="$(buildkite-agent meta-data get release-torchtpu-version)"
PROD_BASE="$(buildkite-agent meta-data get release-base-image)"
DATE="$(date -u +%Y%m%d)"

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/../scripts/cleanup_docker.sh"

DEV_BASE=""
if DEV_BASE="$(pin_by_digest "${RELEASE_GAR}/vllm-torchtpu-dev:${TORCHTPU}" 2>/dev/null)"; then
    echo "dev base image: ${DEV_BASE}"
else
    DEV_BASE=""
    buildkite-agent annotate --style warning --context release-dev-image \
        "No \`${RELEASE_GAR}/vllm-torchtpu-dev:${TORCHTPU}\`; staging the prod image only."
fi

ROWS=""
for target in prod dev; do
    extra=()
    case "${target}" in
        prod) base="${PROD_BASE}" ;;
        dev)
            base="${DEV_BASE}"
            # The dev base's editable core is stamped 0.0.0; reinstall the
            # same release from its branch under its real version.
            extra=(--core-commit "releases/v${TORCHTPU}" --core-version "${TORCHTPU}")
            ;;
    esac
    [[ -n "${base}" ]] || continue
    image="${RELEASE_GAR}/vllm-omni-torchtpu-${target}"
    echo "--- :docker: Building ${image}:${VERSION} on ${base}"
    "${REPO_ROOT}/docker/build_image.sh" --target "${target}" \
        --base-image "${base}" --version "${VERSION}" -t "${image}:${VERSION}" \
        "${extra[@]}"
    docker tag "${image}:${VERSION}" "${image}:${DATE}-${VERSION}"

    echo "--- :docker: Pushing ${image}:${VERSION} and :${DATE}-${VERSION}"
    docker push "${image}:${VERSION}"
    docker push "${image}:${DATE}-${VERSION}"

    ref="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${image}:${VERSION}" \
        | grep -m 1 "^${image}@sha256:")" \
        || die "no ${image} digest after the push."
    buildkite-agent meta-data set "release-image-${target}" "${ref}"
    ROWS+="| ${target} | \`${image}:${VERSION}\` | \`$(digest_of "${ref}")\` | \`${base}\` |"$'\n'

    docker rmi "${image}:${VERSION}" "${image}:${DATE}-${VERSION}" >/dev/null 2>&1 || true
done

buildkite-agent annotate --style info --context release-images <<EOF
Staged images (GAR):

| target | tag | digest | base |
|---|---|---|---|
${ROWS}
EOF
