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

# Runs one command in a single pod on the Kueue fleet:
#   SHAPE=<machine-type>/<topology> .buildkite/kubernetes/run.sh <command> [args...]
#   SHAPE=cpu                       .buildkite/kubernetes/run.sh <command> [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: SHAPE=<machine-type>/<topology>|cpu $0 <command> [args...]" >&2
  exit 2
fi

shape="${SHAPE:-}"
if [[ "$shape" == "cpu" ]]; then
  hardware=(--cpu)
else
  machine_type="${shape%%/*}"
  topology="${shape#*/}"
  if [[ -z "$shape" || "$machine_type" == "$shape" || -z "$topology" ]]; then
    echo "$0: SHAPE must be <machine-type>/<topology> or cpu, got '${shape}'" >&2
    exit 2
  fi
  hardware=(--machine-type "$machine_type" --topology "$topology")
fi

# shellcheck source=.buildkite/kubernetes/common.sh
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# shellcheck disable=SC2016
IN_POD='
  set -o pipefail
  [ -n "${ARTIFACTS_DIR:-}" ] && mkdir -p "$ARTIFACTS_DIR"
  "$@"
  rc=$?
  if [ -n "${ARTIFACTS_DIR:-}" ] && [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ]; then
    find "$ARTIFACTS_DIR" -type l ! -exec test -e {} \; -delete 2>/dev/null || true
    if ! buildkite-agent artifact upload "$ARTIFACTS_DIR/**/*"; then
      echo "ERROR: artifacts were produced but could not be uploaded" >&2
      [ "$rc" -eq 0 ] && rc=1
    fi
  fi
  exit $rc
'

# shellcheck disable=SC2154
exec /opt/launcher/launch \
  "${hardware[@]}" \
  "${env_args[@]}" \
  -- bash -c "$IN_POD" -- "$@"
