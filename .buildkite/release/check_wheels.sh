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

# Check that the wheel this build made installs from PyPI alone, as a user's
# `uv pip install --torch-backend cpu vllm-omni-torchtpu==<version>` would after
# the publish: resolve it from the local wheel plus pypi.org (and the PyTorch
# CPU index for torch/torchaudio), for the platform it supports (CPython 3.12,
# x86_64 Linux, glibc >= 2.31, the floor torch-tpu and tpu-sync-torch set).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
TORCHTPU="$(buildkite-agent meta-data get release-torchtpu-version)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "--- :inbox_tray: Wheel and sdist built in this build"
(cd "${WORK}" && buildkite-agent artifact download "dist/*" .)
ls -l "${WORK}/dist"

echo "--- Installing uv"
UV="$(install_uv "${WORK}/uv")"

echo "--- :mag: Resolving ${RELEASE_PYPI_PROJECT}==${VERSION} from this wheel and pypi.org"
# vllm-torchtpu-core is pinned to the release this build is layered on, so the
# check covers the pairing users get from the prod image.
echo "vllm-torchtpu-core==${TORCHTPU}" > "${WORK}/constraints.txt"
resolve_vllm_omni_torchtpu "${UV}" "${VERSION}" "${WORK}/resolved.txt" \
    --find-links "${WORK}/dist" --constraint "${WORK}/constraints.txt"

grep -E "^(vllm-omni-torchtpu|vllm-torchtpu-core|vllm-omni|vllm|torch|torchaudio|torch-tpu|tpu-sync-torch|libtpu|transformers)==" \
    "${WORK}/resolved.txt" | tee "${WORK}/pins.txt"
grep -qx "${RELEASE_PYPI_PROJECT}==${VERSION}" "${WORK}/pins.txt" \
    || die "the resolution does not pick ${RELEASE_PYPI_PROJECT}==${VERSION}."
check_no_cuda "${WORK}/resolved.txt"

NAMES=""
for f in "${WORK}"/dist/*; do NAMES+="\`$(basename "${f}")\` "; done
buildkite-agent annotate --style info --context release-wheels <<EOF
Built: ${NAMES}. \`${RELEASE_PYPI_PROJECT}==${VERSION}\` with \`vllm-torchtpu-core==${TORCHTPU}\` resolves from them plus pypi.org ($(grep -c '==' "${WORK}/resolved.txt") packages, no CUDA wheels):

\`\`\`
$(cat "${WORK}/pins.txt")
\`\`\`
EOF
