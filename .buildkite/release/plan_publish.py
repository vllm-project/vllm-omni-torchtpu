#!/usr/bin/env python3
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
"""Print the second half of a release build as a Buildkite pipeline (JSON).

    plan_publish.py | buildkite-agent pipeline upload --no-interpolation

Adapted from vllm-project/vllm-torchtpu .buildkite/release/plan_publish.py.

Runs once the images are staged and the wheel is checked: the approval block,
the Docker Hub and PyPI publishes, and the check that what users get is what
was built. Uploaded rather than written into pipeline.yml so the approval
prompt can name the exact version and prod digest, and so the team restriction
is added only when it is set. --no-interpolation: nothing in it is for
Buildkite to interpolate.

Optional build env:
  RELEASE_APPROVER_TEAMS=a,b   Buildkite team slugs allowed to approve
  RELEASE_QUEUE=<queue>        agent queue (default release-vllm-torchtpu)
  RELEASE_DOCKERHUB_REPO=<r>   Docker Hub repository (vllm/vllm-omni-torchtpu)
"""

from __future__ import annotations

import json
import os
import subprocess

DEFAULT_QUEUE = "release-vllm-torchtpu"
DEFAULT_DOCKERHUB_REPO = "vllm/vllm-omni-torchtpu"
DOCKERHUB_SECRET = "DOCKERHUB_TOKEN_VLLM_OMNI_TORCHTPU"
PYPI_PROJECT = "vllm-omni-torchtpu"


def meta(key: str) -> str:
    return subprocess.run(
        ["buildkite-agent", "meta-data", "get", key],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def command_step(
    label: str,
    key: str,
    depends_on: list[str],
    commands: list[str],
    queue: str,
    **extra: object,
) -> dict:
    step = {
        "label": label,
        "key": key,
        "depends_on": depends_on,
        "agents": {"queue": queue},
        "commands": commands,
    }
    step.update(extra)
    return step


def build_steps(
    *,
    version: str,
    prod: str,
    prerelease: bool,
    branch: str,
    torchtpu_version: str,
    dockerhub_repo: str = DEFAULT_DOCKERHUB_REPO,
    queue: str = DEFAULT_QUEUE,
    teams: list[str] | None = None,
) -> list[dict]:
    """The approval block and the publish steps that wait on it."""
    latest_options = [{"label": "No", "value": "no"}]
    if not prerelease:
        latest_options.insert(0, {"label": "Yes", "value": "yes"})
    block: dict = {
        "block": f":rocket: Publish {PYPI_PROJECT} {version}?",
        "key": "approve_publish",
        "depends_on": ["plan_publish"],
        "prompt": (
            f"Publish {PYPI_PROJECT} {version} from {branch}, built on "
            f"vllm-torchtpu {torchtpu_version}: the staged prod image {prod} "
            f"to Docker Hub as {dockerhub_repo}:v{version}, and the "
            f"{PYPI_PROJECT} {version} wheel and sdist to PyPI. "
            "PyPI releases can't be replaced."
        ),
        "fields": [
            {
                "select": "Move latest on Docker Hub to this version?",
                "key": "dockerhub_move_latest",
                "required": True,
                "default": latest_options[0]["value"],
                "options": latest_options,
                "hint": "No for a patch to an older line. "
                "A pre-release never moves latest.",
            },
            {
                "select": f"Overwrite v{version} if it is already on Docker Hub?",
                "key": "dockerhub_overwrite",
                "required": True,
                "default": "no",
                "options": [
                    {"label": "No", "value": "no"},
                    {"label": "Yes", "value": "yes"},
                ],
                "hint": "Released tags should not change; "
                "Yes only to replace a broken image.",
            },
        ],
    }
    if teams:
        block["allowed_teams"] = teams

    return [
        block,
        command_step(
            ":docker: Publish the prod image to Docker Hub",
            "publish_images",
            ["approve_publish"],
            ["bash .buildkite/release/publish_images.sh"],
            queue,
            secrets=[DOCKERHUB_SECRET],
            concurrency=1,
            concurrency_group="vllm-omni-torchtpu-dockerhub-publish",
        ),
        command_step(
            f":package: Publish {PYPI_PROJECT} to PyPI",
            "publish_wheel",
            ["approve_publish"],
            ["bash .buildkite/scripts/publish_wheel.sh"],
            queue,
            env={"PUBLISH_PYPI": PYPI_PROJECT},
            concurrency=1,
            concurrency_group="vllm-omni-torchtpu-publish",
        ),
        command_step(
            ":white_check_mark: Check what users get",
            "verify_published",
            ["publish_images", "publish_wheel"],
            ["bash .buildkite/release/verify_published.sh"],
            queue,
        ),
    ]


def main() -> None:
    teams = [t for t in os.environ.get("RELEASE_APPROVER_TEAMS", "").split(",") if t]
    steps = build_steps(
        version=meta("release-version"),
        prod=meta("release-image-prod"),
        prerelease=meta("release-prerelease") == "yes",
        branch=os.environ["BUILDKITE_BRANCH"],
        torchtpu_version=meta("release-torchtpu-version"),
        dockerhub_repo=os.environ.get("RELEASE_DOCKERHUB_REPO", DEFAULT_DOCKERHUB_REPO),
        queue=os.environ.get("RELEASE_QUEUE", DEFAULT_QUEUE),
        teams=teams,
    )
    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
