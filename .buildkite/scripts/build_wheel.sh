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
#
# ------------------------------------------------------
# Build the vllm-omni-torchtpu release wheel and sdist.
# Adapted from vllm-project/vllm-torchtpu .buildkite/scripts/build_core_wheel.sh.
#
# Version (resolved by release_version in .buildkite/release/lib.sh):
#   - Set once by preflight.sh in Buildkite meta-data (release-version), or
#     resolved on the spot from RELEASE_VERSION / BUILDKITE_BRANCH
#     (RELEASE_BRANCH_REGEX in .buildkite/release/lib.sh).
#   - Forced onto setuptools-scm, so a shallow CI checkout without tags still
#     gets it.
#
# Output:
#   dist/vllm_omni_torchtpu-<version>-py3-none-any.whl and
#   dist/vllm_omni_torchtpu-<version>.tar.gz, uploaded as Buildkite artifacts.
#   Build meta-data WHEEL_NAME for later steps.
#
# CI only: runs inside a vllm-omni-torchtpu-release build.
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WHEEL_UTILS="${SCRIPT_DIR}/wheel_utils.py"
# Provides release_version, check_not_published and die.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"

command -v buildkite-agent >/dev/null 2>&1 \
    || die "buildkite-agent not found: this script only runs in Buildkite."

# --- Resolve the release version ---------------------------------------------
echo "--- :label: Resolving release version"
VERSION="$(release_version)"
echo "Release version: ${VERSION}"

# Stop before building if this version is already on PyPI (versions can't be
# reused). PUBLISH_PYPI comes from the step's env in .buildkite/release/pipeline.yml.
[[ -n "${PUBLISH_PYPI:-}" ]] || die "PyPI project (PUBLISH_PYPI) is not set."
check_not_published "${PUBLISH_PYPI}" "${VERSION}"

# --- Private build venv ------------------------------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/omni-build.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

# CI agents lack python3-venv (no ensurepip), so create the venv without pip
# and bootstrap pip with PyPA's get-pip.py.
"${PYTHON}" -m venv --without-pip "${WORK_DIR}/venv" \
    || die "could not create a venv with ${PYTHON}."
PY="${WORK_DIR}/venv/bin/python"
curl -fsSL --retry 3 -o "${WORK_DIR}/get-pip.py" \
    https://bootstrap.pypa.io/get-pip.py \
    || die "could not download get-pip.py from bootstrap.pypa.io."
"$PY" "${WORK_DIR}/get-pip.py" --quiet
"$PY" -m pip install --quiet build twine packaging \
    "setuptools>=77.0.0" wheel "setuptools_scm[toml]>=8.0"

# --- Build -------------------------------------------------------------------
echo "--- :package: Building wheel and sdist"
cd "${REPO_ROOT}"
rm -rf dist build src/*.egg-info
SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}" "$PY" -m build --sdist --wheel --no-isolation

shopt -s nullglob
WHEELS=(dist/*.whl)
SDISTS=(dist/*.tar.gz)
shopt -u nullglob
[[ "${#WHEELS[@]}" -eq 1 ]] \
    || die "expected exactly one wheel in dist/, found ${#WHEELS[@]}."
[[ "${#SDISTS[@]}" -eq 1 ]] \
    || die "expected exactly one sdist in dist/, found ${#SDISTS[@]}."
WHEEL="${WHEELS[0]}"
SDIST="${SDISTS[0]}"

# The file names must carry exactly the release version; anything else means
# the version override did not take effect.
"$PY" "$WHEEL_UTILS" verify_built_wheel "$WHEEL" "$VERSION"
"$PY" "$WHEEL_UTILS" verify_wheel_contents "$WHEEL"
"$PY" "$WHEEL_UTILS" verify_publish_dist "$WHEEL" "$PUBLISH_PYPI" "$VERSION"
"$PY" "$WHEEL_UTILS" verify_publish_dist "$SDIST" "$PUBLISH_PYPI" "$VERSION"

echo "--- :mag: Checking metadata"
"$PY" -m twine check --strict "$WHEEL" "$SDIST"

# --- Hand off to later steps -------------------------------------------------
buildkite-agent artifact upload "$WHEEL"
buildkite-agent artifact upload "$SDIST"
buildkite-agent meta-data set "WHEEL_NAME" "$(basename "$WHEEL")"

echo "Built ${WHEEL} and ${SDIST}"
