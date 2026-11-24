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
"""CPU unit tests for the release tooling under .buildkite/.

Covers ci_gate.py, plan_publish.py (.buildkite/release/) and wheel_utils.py
(.buildkite/scripts/). They need only the standard library, pytest and
packaging, so they also run with ``--noconftest`` on a host without vLLM.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import pathlib
import sys
import zipfile
from types import ModuleType

import pytest

_BUILDKITE = pathlib.Path(__file__).resolve().parents[1] / ".buildkite"


def _load(relpath: str) -> ModuleType:
    path = _BUILDKITE / relpath
    name = f"_release_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ci_gate = _load("release/ci_gate.py")
plan_publish = _load("release/plan_publish.py")
wheel_utils = _load("scripts/wheel_utils.py")


# The tests/conftest.py autouse fixtures touch the TPU and vllm_torchtpu;
# these tests are pure CPU, so override them with no-ops.
@pytest.fixture(autouse=True)
def _sync_tpu() -> None:
    """Release tooling tests run on CPU without a TPU accelerator."""


@pytest.fixture(autouse=True)
def _omni_model_patches() -> None:
    """No model patches needed."""


@pytest.fixture(autouse=True)
def _torchtpu_eager_mode() -> None:
    """No torch_tpu eager mode needed."""


# --- ci_gate.py ---------------------------------------------------------------

BUILD_URL = "https://buildkite.com/vllm/vllm-omni-torchtpu-ci/builds/42"
CTX = ci_gate.CONTEXT


def _status(context: str, state: str, url: str = BUILD_URL) -> dict:
    return {"context": context, "state": state, "target_url": url}


def _nightly_statuses(state: str = "success", url: str = BUILD_URL) -> list[dict]:
    return [
        _status(f"{CTX}/unit-tests-omni-nightly", state, f"{url}#1"),
        _status(f"{CTX}/unit-tests-omni-multichip-nightly", state, f"{url}#2"),
    ]


def test_ci_gate_targets_this_repo() -> None:
    assert ci_gate.REPO == "vllm-project/vllm-omni-torchtpu"
    assert ci_gate.CONTEXT == "buildkite/vllm-omni-torchtpu-ci"


def test_ci_gate_required_steps_match_pipeline() -> None:
    pipeline = (_BUILDKITE / "pipeline.yml").read_text(encoding="utf-8")
    for step in ci_gate.REQUIRED_NIGHTLY_STEPS:
        assert f'key: "{step}"' in pipeline


@pytest.mark.parametrize(
    "context",
    [
        f"{CTX}/unit-tests-omni-nightly",
        f"{CTX}/unit_tests_omni_nightly",
        f"{CTX}/Unit Tests - Omni (Nightly)",
    ],
)
def test_ci_gate_step_name_normalizes_label_and_key(context: str) -> None:
    assert ci_gate.step_name(context) == "unit_tests_omni_nightly"


def test_ci_gate_passes_full_nightly_build() -> None:
    statuses = [_status(CTX, "success"), *_nightly_statuses()]
    url, reason = ci_gate.check("abc", statuses)
    assert reason is None
    assert url == BUILD_URL


def test_ci_gate_accepts_step_keys() -> None:
    statuses = [
        _status(CTX, "success"),
        _status(f"{CTX}/unit_tests_omni_nightly", "success", f"{BUILD_URL}#1"),
        _status(
            f"{CTX}/unit_tests_omni_multichip_nightly", "success", f"{BUILD_URL}#2"
        ),
    ]
    assert ci_gate.check("abc", statuses) == (BUILD_URL, None)


def test_ci_gate_fails_without_build() -> None:
    url, reason = ci_gate.check("abc", [])
    assert url is None
    assert "no buildkite/vllm-omni-torchtpu-ci build" in reason


def test_ci_gate_fails_on_failed_build() -> None:
    statuses = [_status(CTX, "failure"), *_nightly_statuses()]
    url, reason = ci_gate.check("abc", statuses)
    assert url is None
    assert "failure, not success" in reason


def test_ci_gate_fails_on_pr_lane_build() -> None:
    statuses = [
        _status(CTX, "success"),
        _status(f"{CTX}/unit-tests-omni-pr", "success", f"{BUILD_URL}#1"),
    ]
    url, reason = ci_gate.check("abc", statuses)
    assert url is None
    assert "unit_tests_omni_nightly" in reason
    assert "unit_tests_omni_multichip_nightly" in reason


def test_ci_gate_requires_every_nightly_step() -> None:
    statuses = [
        _status(CTX, "success"),
        _status(f"{CTX}/unit-tests-omni-nightly", "success", f"{BUILD_URL}#1"),
        _status(
            f"{CTX}/unit-tests-omni-multichip-nightly", "failure", f"{BUILD_URL}#2"
        ),
    ]
    url, reason = ci_gate.check("abc", statuses)
    assert url is None
    assert "unit_tests_omni_multichip_nightly" in reason
    assert "unit_tests_omni_nightly," not in reason


def test_ci_gate_ignores_steps_of_older_builds() -> None:
    older = "https://buildkite.com/vllm/vllm-omni-torchtpu-ci/builds/41"
    statuses = [_status(CTX, "success"), *_nightly_statuses(url=older)]
    url, reason = ci_gate.check("abc", statuses)
    assert url is None
    assert reason is not None


def test_ci_gate_newest_step_status_wins() -> None:
    # Newest first: a retried step that failed after an earlier success.
    statuses = [
        _status(CTX, "success"),
        *_nightly_statuses("failure"),
        *_nightly_statuses("success"),
    ]
    url, _ = ci_gate.check("abc", statuses)
    assert url is None


# --- plan_publish.py ----------------------------------------------------------

PROD = "us-central1-docker.pkg.dev/p/r/vllm-omni-torchtpu-prod@sha256:" + "a" * 64


def _steps(**overrides: object) -> list[dict]:
    kwargs: dict = {
        "version": "0.34.0",
        "prod": PROD,
        "prerelease": False,
        "branch": "releases/v0.34.0",
        "torchtpu_version": "0.34.0",
    }
    kwargs.update(overrides)
    return plan_publish.build_steps(**kwargs)


def test_plan_publish_steps_and_dependencies() -> None:
    steps = _steps()
    assert [s["key"] for s in steps] == [
        "approve_publish",
        "publish_images",
        "publish_wheel",
        "verify_published",
    ]
    by_key = {s["key"]: s for s in steps}
    assert by_key["approve_publish"]["depends_on"] == ["plan_publish"]
    assert by_key["publish_images"]["depends_on"] == ["approve_publish"]
    assert by_key["publish_wheel"]["depends_on"] == ["approve_publish"]
    assert by_key["verify_published"]["depends_on"] == [
        "publish_images",
        "publish_wheel",
    ]
    for key in ("publish_images", "publish_wheel", "verify_published"):
        assert by_key[key]["agents"] == {"queue": "release-vllm-torchtpu"}
    assert by_key["publish_images"]["secrets"] == ["DOCKERHUB_TOKEN_VLLM_OMNI_TORCHTPU"]
    assert by_key["publish_wheel"]["env"] == {"PUBLISH_PYPI": "vllm-omni-torchtpu"}
    assert by_key["publish_wheel"]["commands"] == [
        "bash .buildkite/scripts/publish_wheel.sh"
    ]
    # Only the Docker Hub step may read the Docker Hub token.
    assert "secrets" not in by_key["publish_wheel"]
    assert "secrets" not in by_key["verify_published"]


def test_plan_publish_prompt_names_version_digest_and_repo() -> None:
    prompt = _steps()[0]["prompt"]
    assert "vllm-omni-torchtpu 0.34.0" in prompt
    assert PROD in prompt
    assert "vllm/vllm-omni-torchtpu:v0.34.0" in prompt
    assert "vllm-torchtpu 0.34.0" in prompt


def test_plan_publish_release_offers_latest() -> None:
    latest = _steps()[0]["fields"][0]
    assert latest["key"] == "dockerhub_move_latest"
    assert [o["value"] for o in latest["options"]] == ["yes", "no"]
    assert latest["default"] == "yes"


def test_plan_publish_prerelease_never_moves_latest() -> None:
    latest = _steps(version="0.34.0rc1", prerelease=True)[0]["fields"][0]
    assert [o["value"] for o in latest["options"]] == ["no"]
    assert latest["default"] == "no"


def test_plan_publish_overwrite_defaults_to_no() -> None:
    overwrite = _steps()[0]["fields"][1]
    assert overwrite["key"] == "dockerhub_overwrite"
    assert overwrite["default"] == "no"


def test_plan_publish_teams_and_queue() -> None:
    block = _steps()[0]
    assert "allowed_teams" not in block
    steps = _steps(teams=["release-approvers"], queue="release-omni")
    assert steps[0]["allowed_teams"] == ["release-approvers"]
    assert steps[1]["agents"] == {"queue": "release-omni"}


def test_plan_publish_main_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    meta = {
        "release-version": "0.34.0rc1",
        "release-image-prod": PROD,
        "release-prerelease": "yes",
        "release-torchtpu-version": "0.34.0",
    }
    monkeypatch.setattr(plan_publish, "meta", meta.__getitem__)
    monkeypatch.setenv("BUILDKITE_BRANCH", "releases/v0.34.0")
    monkeypatch.setenv("RELEASE_APPROVER_TEAMS", "a,,b")
    monkeypatch.delenv("RELEASE_QUEUE", raising=False)
    monkeypatch.delenv("RELEASE_DOCKERHUB_REPO", raising=False)
    plan_publish.main()
    steps = json.loads(capsys.readouterr().out)["steps"]
    assert steps[0]["allowed_teams"] == ["a", "b"]
    assert "0.34.0rc1" in steps[0]["block"]
    assert [o["value"] for o in steps[0]["fields"][0]["options"]] == ["no"]


# --- wheel_utils.py -----------------------------------------------------------


def _wheel(tmp_path: pathlib.Path, name: str, members: list[str]) -> str:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member in members:
            zf.writestr(member, "")
    return str(path)


GOOD_MEMBERS = [
    "vllm_omni_torchtpu/__init__.py",
    "vllm_omni_torchtpu/worker/base.py",
    "vllm_omni_torchtpu-0.34.0.dist-info/METADATA",
    "vllm_omni_torchtpu-0.34.0.dist-info/entry_points.txt",
]


def test_verify_built_wheel_accepts_release(capsys: pytest.CaptureFixture) -> None:
    wheel_utils.verify_built_wheel(
        "dist/vllm_omni_torchtpu-0.34.0rc1-py3-none-any.whl", "0.34.0rc1"
    )
    assert "0.34.0rc1" in capsys.readouterr().out


def test_verify_built_wheel_rejects_wrong_version() -> None:
    with pytest.raises(SystemExit, match="!= release version"):
        wheel_utils.verify_built_wheel(
            "vllm_omni_torchtpu-0.0.0-py3-none-any.whl", "0.34.0"
        )


def test_verify_built_wheel_rejects_platform_wheel() -> None:
    with pytest.raises(SystemExit, match="not py3-none-any"):
        wheel_utils.verify_built_wheel(
            "vllm_omni_torchtpu-0.34.0-cp312-cp312-manylinux_2_28_x86_64.whl",
            "0.34.0",
        )


def test_verify_wheel_contents(tmp_path: pathlib.Path) -> None:
    good = _wheel(tmp_path, "good.whl", GOOD_MEMBERS)
    wheel_utils.verify_wheel_contents(good)

    for stray in ("tests/test_packaging.py", ".buildkite/pipeline.yml", "docs/x.md"):
        bad = _wheel(tmp_path, "bad.whl", [*GOOD_MEMBERS, stray])
        with pytest.raises(SystemExit, match="stray files"):
            wheel_utils.verify_wheel_contents(bad)

    empty = _wheel(tmp_path, "empty.whl", GOOD_MEMBERS[2:])
    with pytest.raises(SystemExit, match="no vllm_omni_torchtpu modules"):
        wheel_utils.verify_wheel_contents(empty)


@pytest.mark.parametrize(
    "filename",
    [
        "vllm_omni_torchtpu-0.34.0-py3-none-any.whl",
        "vllm_omni_torchtpu-0.34.0.tar.gz",
    ],
)
def test_verify_publish_dist_accepts(filename: str) -> None:
    wheel_utils.verify_publish_dist(filename, "vllm-omni-torchtpu", "0.34.0")


@pytest.mark.parametrize(
    ("filename", "match"),
    [
        ("vllm_torchtpu_core-0.34.0-py3-none-any.whl", "PUBLISH_PYPI"),
        ("vllm_omni_torchtpu-0.34.1.tar.gz", "release version"),
    ],
)
def test_verify_publish_dist_rejects(filename: str, match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        wheel_utils.verify_publish_dist(filename, "vllm-omni-torchtpu", "0.34.0")


def _jwt(claims: dict) -> str:
    def enc(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{enc({'alg': 'RS256'})}.{enc(claims)}.sig"


def test_gcp_oidc_claims() -> None:
    good = {"iss": "https://accounts.google.com", "aud": "pypi"}
    claims = wheel_utils.decode_jwt_claims(_jwt(good))
    assert claims == good
    assert wheel_utils.check_gcp_oidc_claims(claims, "pypi") is None
    assert "not Google Cloud" in wheel_utils.check_gcp_oidc_claims(
        {**good, "iss": "https://token.actions.githubusercontent.com"}, "pypi"
    )
    assert "audience" in wheel_utils.check_gcp_oidc_claims(
        {**good, "aud": "sigstore"}, "pypi"
    )


def test_decode_jwt_claims_rejects_non_jwt() -> None:
    with pytest.raises(SystemExit, match="not a JWT"):
        wheel_utils.decode_jwt_claims("not-a-token")


def test_wheel_utils_cli_rejects_unknown_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["wheel_utils.py", "nope"])
    with pytest.raises(SystemExit, match="unknown command"):
        wheel_utils.main()
