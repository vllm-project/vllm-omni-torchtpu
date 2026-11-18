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

# Sourced by .buildkite/kubernetes/run.sh; leaves WORKLOAD_IMAGE exported and
# the launcher's --env flags in `env_args`.

if [[ -z "${WORKLOAD_IMAGE:-}" ]]; then
  WORKLOAD_IMAGE=$(buildkite-agent meta-data get "${CI_IMAGE_TAG_KEY:-CI_IMAGE_TAG${DOCKER_IMAGE_SUFFIX:-}}" \
    --default "" 2>/dev/null || true)
fi
if [[ -z "${WORKLOAD_IMAGE:-}" ]]; then
  echo "$0: no CI_IMAGE_TAG${DOCKER_IMAGE_SUFFIX:-} metadata and no WORKLOAD_IMAGE." \
       "The image build step has to run before this one, or the step has to" \
       "name an image." >&2
  exit 2
fi
export WORKLOAD_IMAGE

export FORCE_COLOR="1"
export TQDM_MININTERVAL="30"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE="0.33.0"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_OMNI_TORCHTPU="0.0.0"
export VLLM_PLUGINS="${VLLM_PLUGINS:-torchtpu_omni,torchtpu_layers,vllm_omni_register_models}"
export UV_NO_CACHE="1"
export CI_RUNNER="kube"
export SKIP_SPANNER_UPLOAD="1"

FORWARD=(
  JAX_PLATFORMS
  FORCE_COLOR
  TQDM_MININTERVAL
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_OMNI_TORCHTPU
  VLLM_PLUGINS
  UV_NO_CACHE
  CI_RUNNER
  SKIP_SPANNER_UPLOAD
  CAPTURE_PROFILE
  TMPDIR
  RAY_TMPDIR
  VLLM_CACHE_ROOT
  VLLM_XLA_CACHE_PATH
  SERVED_MODEL_NAME
  RUN_ROOT
  VLLM_ENGINE_READY_TIMEOUT_S
  TPU_ACCELERATOR_TYPE
  TPU_MULTIHOST_BACKEND
  NIGHTLY
  CREATED_BY
  SHAPE
  WORKLOAD_IMAGE
  PYTEST_ADDOPTS
  BUILDKITE_ANALYTICS_TOKEN
  INGEST_BEARER_TOKEN
  ARTIFACTS_DIR
)

unset BUILDKITE_ANALYTICS_TOKEN BUILDKITE_TEST_ENGINE_API_ACCESS_TOKEN
export BUILDKITE_AGENT_OIDC_REQUEST_TOKEN_SKIP_TOKEN_REDACTION=true

env_args=()
for _name in "${FORWARD[@]}"; do
  env_args+=(--env "$_name")
done
unset _name
