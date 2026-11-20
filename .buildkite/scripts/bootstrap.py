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
#
# Derived from vllm-project/vllm-torchtpu `.buildkite/scripts/bootstrap.py` at
# commit a0275aad4f81a8befa40f8dc7a3fbeebaaff4355, trimmed to upload the single
# `.buildkite/pipeline.yml` without multi-stage, merge-queue, dev, perf,
# pre-commit, or release flows.
"""Bootstrap the Buildkite CI pipeline for `vllm-omni-torchtpu`.

Compatible with Python 3.10+ stdlib on `queue=cpu` (invoked by `bootstrap.sh`).
"""

from __future__ import annotations

import http.client
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import cap_pr_tpu_concurrency

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_PIPELINE_FILE = ".buildkite/pipeline.yml"
_CREDENTIAL_HELPER = pathlib.Path("/etc/buildkite-agent/git-credential-github-app")
_GITHUB_API_URL = "https://api.github.com"
_DEFAULT_GITHUB_REPO = "vllm-project/vllm-omni-torchtpu"
_GITHUB_REPO_RE = re.compile(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?/?$")

# Max TPU jobs one author's PR builds may have in flight on kube (~2 PRs at ~19
# jobs); extra jobs wait in Buildkite as `limited` and take no Kueue slot, while
# nightly, post-merge, and release builds run uncapped.
PR_TPU_JOBS_PER_AUTHOR = 40

# History depths tried, in order, until a PR's merge-base with its base is found.
_PR_FETCH_DEPTHS = (20, 200, 1000)
_PR_SUITE_LABELS = ("skip-tpu-ci", "run-full-ci")


class BootstrapError(Exception):
    """Fatal configuration or git/pin lookup error that exits bootstrap with 1."""


def _env_or(env: Mapping[str, str], key: str, default: str) -> str:
    """Match bash `${VAR:-default}`: treat both unset and empty string as default."""
    val = env.get(key, "")
    return val if val else default


def _split_labels(raw: str) -> tuple[str, ...]:
    return tuple(item for item in raw.split(",") if item)


def _resolve_github_repo(repo_url: str) -> str:
    match = _GITHUB_REPO_RE.search(repo_url.strip())
    return match.group(1) if match else _DEFAULT_GITHUB_REPO


@dataclass(frozen=True)
class BuildContext:
    """Immutable snapshot of the build environment read at startup."""

    branch: str
    commit: str
    message: str
    source: str
    github_repo: str
    pull_request: str | None
    pr_base_branch: str
    labels: tuple[str, ...]
    author_email: str
    nightly_env: bool
    torchtpu_nightly_env: bool
    vllm_commit_override: str
    pr_tpu_concurrency_group_env: str

    @property
    def is_pr(self) -> bool:
        return self.pull_request is not None

    @property
    def is_post_merge(self) -> bool:
        return not self.is_pr and self.branch == "main" and self.source == "webhook"

    @property
    def is_oncall_fix(self) -> bool:
        return self.is_pr and "oncall-fix" in self.labels

    @property
    def is_full_ci(self) -> bool:
        return self.is_pr and "run-full-ci" in self.labels

    @property
    def runs_tpu(self) -> bool:
        return "skip-tpu-ci" not in self.labels

    def check_suite_labels(self) -> None:
        """Raise `BootstrapError` if mutually exclusive PR suite labels are set."""
        if not self.is_pr:
            return
        matches = [label for label in _PR_SUITE_LABELS if label in self.labels]
        if len(matches) > 1:
            joined = ", ".join(matches)
            raise BootstrapError(
                f"ERROR: PR labels {joined} are mutually exclusive. "
                "Remove all but one of skip-tpu-ci / run-full-ci in the "
                "**Labels** sidebar on the GitHub PR page, then retry the "
                "**Bootstrap Pipeline** step."
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BuildContext:
        if _env_or(env, "SKIP_TPU_CI", "0").strip() != "0":
            raise BootstrapError(
                "ERROR: SKIP_TPU_CI was removed; use the skip-tpu-ci PR label"
            )
        pr_raw = _env_or(env, "BUILDKITE_PULL_REQUEST", "false")
        pull_request = None if pr_raw == "false" else pr_raw
        labels = (
            _split_labels(env.get("BUILDKITE_PULL_REQUEST_LABELS", ""))
            if pull_request is not None
            else ()
        )
        author = _env_or(
            env,
            "BUILDKITE_BUILD_AUTHOR_EMAIL",
            _env_or(env, "BUILDKITE_BUILD_CREATOR_EMAIL", ""),
        )
        return cls(
            branch=env.get("BUILDKITE_BRANCH", ""),
            commit=_env_or(env, "BUILDKITE_COMMIT", "HEAD"),
            message=env.get("BUILDKITE_MESSAGE", ""),
            source=env.get("BUILDKITE_SOURCE", ""),
            github_repo=_resolve_github_repo(env.get("BUILDKITE_REPO", "")),
            pull_request=pull_request,
            pr_base_branch=_env_or(env, "BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main"),
            labels=labels,
            author_email=author,
            nightly_env=_env_or(env, "NIGHTLY", "0") == "1",
            torchtpu_nightly_env=_env_or(env, "TORCHTPU_NIGHTLY", "0") == "1",
            vllm_commit_override=env.get("VLLM_COMMIT_HASH_OVERRIDE", ""),
            pr_tpu_concurrency_group_env=env.get("PR_TPU_CONCURRENCY_GROUP", ""),
        )


# Build kind -> (Buildkite job priority, Kueue workload priority class).
# Nightly builds are unclassed on kube, level with the tpu-inference nightlies.
PRIORITIES: dict[str, tuple[int, str | None]] = {
    "oncall-fix": (12, "oncall-fix"),
    "post-merge": (11, "post-merge"),
    "pre-merge": (6, "pre-merge"),
    "other": (2, None),
    "nightly": (1, None),
}


def build_kind(ctx: BuildContext) -> str:
    """Classify the build into one of the `PRIORITIES` keys."""
    if ctx.nightly_env or ctx.torchtpu_nightly_env:
        return "nightly"
    if ctx.is_oncall_fix:
        return "oncall-fix"
    if ctx.is_pr:
        return "pre-merge"
    if ctx.branch == "main":
        return "post-merge"
    return "other"


# Commit-message, changed-file, and vLLM-pin patterns (adapted from upstream by
# dropping `^\.pyrefly-baseline\.json$`, which does not exist in this repo).
_SKIP_CI_RE = re.compile(r"\[(skip\sci|ci\sskip)\]")
_DOCS_ONLY_RE = re.compile(
    r"(\.md$|\.ico$|\.png$|^README$|^docs/|^\.github/CODEOWNERS$"
    r"|^LICENSE$|^\.git-blame-ignore-revs$)"
)
_MERGED_PR_RE = re.compile(r"\(#([0-9]+)\)$")
_VLLM_PIN_RE = re.compile(
    r".*vllm @ git\+https://github\.com/vllm-project/vllm\.git@([a-zA-Z0-9.-]+).*"
)
_MISSING_VLLM_PIN_ERROR = (
    "ERROR: vLLM commit hash is missing from pyproject.toml. "
    "Cannot proceed without a pinned hash."
)


def has_skip_ci_marker(message: str) -> bool:
    """Return True if the commit message contains `[skip ci]` or `[ci skip]`."""
    return bool(_SKIP_CI_RE.search(message))


def only_docs_changed(files: Sequence[str]) -> bool:
    """Return True when `files` is non-empty and every path is non-code/docs."""
    return bool(files) and all(bool(_DOCS_ONLY_RE.search(path)) for path in files)


def parse_merged_pr_number(message: str) -> str | None:
    """Extract trailing `(#123)` PR number from the first line of a squash commit."""
    first_line = message.splitlines()[0] if message else ""
    match = _MERGED_PR_RE.search(first_line)
    return match.group(1) if match else None


def parse_pinned_vllm_commit(pyproject_text: str) -> str:
    """Extract the pinned upstream vLLM ref from `pyproject.toml`."""
    for line in pyproject_text.splitlines():
        match = _VLLM_PIN_RE.match(line)
        if match:
            return match.group(1)
    raise BootstrapError(_MISSING_VLLM_PIN_ERROR)


def read_pinned_vllm_commit(repo_root: pathlib.Path = _REPO_ROOT) -> str:
    """Read `pyproject.toml` under `repo_root` and return the pinned vLLM ref."""
    pyproject_path = repo_root / "pyproject.toml"
    try:
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapError(f"ERROR: could not read {pyproject_path}: {exc}") from exc
    return parse_pinned_vllm_commit(pyproject_text)


def prepare_upload_yaml(raw_yaml: str, *, job_priority: int, nightly: bool) -> str:
    """Prepend `priority:` and substitute `build.env("NIGHTLY")` when needed."""
    # build.env("NIGHTLY") is evaluated by Buildkite against the build's initial
    # environment, so a NIGHTLY=1 set by bootstrap (for the run-full-ci label)
    # never reaches it unless substituted before upload.
    body = raw_yaml.replace('build.env("NIGHTLY")', '"1"') if nightly else raw_yaml
    return f"priority: {job_priority}\n{body}"


def _strip_tpu_steps(steps: list[Any]) -> list[Any]:
    """Remove TPU steps (`holds_chips`) and any groups left empty."""
    kept: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            kept.append(step)
            continue
        if isinstance(step.get("steps"), list):
            sub = _strip_tpu_steps(step["steps"])
            if sub:
                kept.append({**step, "steps": sub})
        elif not cap_pr_tpu_concurrency.holds_chips(step):
            kept.append(step)
    return kept


# ---------------------------------------------------------------------------
# Subprocess & GitHub helpers (monkeypatched in tests)
# ---------------------------------------------------------------------------


def _agent(
    *args: str,
    input_text: str | None = None,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: pathlib.Path = _REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run `buildkite-agent` with the given arguments."""
    return subprocess.run(
        ["buildkite-agent", *args],
        cwd=cwd,
        env=env,
        input=input_text,
        stdout=subprocess.PIPE if "--dry-run" in args else None,
        text=True,
        check=check,
    )


def _git(
    *args: str,
    check: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: pathlib.Path = _REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run a `git` command and capture stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if args[:1] in (("diff",), ("merge-base",)) else None,
        text=True,
        check=check,
    )


def _diff_names(
    *diff_specs: Sequence[str], env: Mapping[str, str], cwd: pathlib.Path = _REPO_ROOT
) -> list[str]:
    for args in diff_specs:
        diff = _git("diff", "--name-only", *args, env=env, cwd=cwd)
        if diff.returncode == 0 and (
            files := [line for line in diff.stdout.splitlines() if line]
        ):
            return files
    return []


def _changed_files(
    ctx: BuildContext,
    *,
    env: Mapping[str, str],
    cwd: pathlib.Path = _REPO_ROOT,
) -> list[str]:
    """Return files changed by a PR or non-scheduled commit build."""
    if ctx.is_pr:
        print(f"PR detected. Target branch: {ctx.pr_base_branch}")
        base_ref = f"origin/{ctx.pr_base_branch}"
        # A shallow fetch misses the merge-base of a PR that is many commits
        # behind its base, and `base...commit` then fails. Fetch deeper until
        # git finds it. Without a merge-base, run everything: diffing only the
        # head commit would treat a multi-commit PR whose last commit is
        # docs-only as a docs-only change.
        for depth in _PR_FETCH_DEPTHS:
            if (
                _git(
                    "fetch",
                    "origin",
                    ctx.pr_base_branch,
                    f"--depth={depth}",
                    "--quiet",
                    env=env,
                    cwd=cwd,
                ).returncode
                != 0
            ):
                print("Base fetch failed")
            _git(
                "fetch",
                "origin",
                ctx.commit,
                f"--depth={depth}",
                "--quiet",
                env=env,
                cwd=cwd,
            )
            mb = _git("merge-base", base_ref, ctx.commit, env=env, cwd=cwd)
            if mb.returncode == 0:
                break
            print(f"No merge-base with {base_ref} at depth {depth}")
        else:
            print(f"No merge-base with {base_ref}; running the full build")
            return []
        if files := _diff_names([f"{base_ref}...{ctx.commit}"], env=env, cwd=cwd):
            return files
        tree = _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-m",
            ctx.commit,
            check=True,
            env=env,
            cwd=cwd,
        )
        return [line for line in tree.stdout.splitlines() if line]

    if ctx.source != "schedule":
        # Any other build is of a commit, which on main is the squash merge of
        # one PR, so its diff against its parent is the whole change. Not a
        # scheduled build: the nightly tests main as it stands, whatever its
        # last commit touched.
        _git("fetch", "origin", ctx.commit, "--depth=2", "--quiet", env=env, cwd=cwd)
        return _diff_names([f"{ctx.commit}^", ctx.commit], env=env, cwd=cwd)

    return []


def _github_app_token(*, env: Mapping[str, str] | None = None) -> str | None:
    if not _CREDENTIAL_HELPER.is_file():
        return None
    try:
        proc = subprocess.run(
            [str(_CREDENTIAL_HELPER), "get"],
            cwd=_REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            token = line.removeprefix("password=").strip()
            if token:
                return token
    return None


def _github_get(path: str, *, repo: str, token: str | None) -> str:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"{_GITHUB_API_URL.rstrip('/')}/repos/{repo}/{path.lstrip('/')}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=10
    ) as resp:
        return resp.read().decode("utf-8")


def _github_labels(
    pr_number: str,
    *,
    repo: str = _DEFAULT_GITHUB_REPO,
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Read labels of a merged PR from GitHub; return `()` on any failure."""
    try:
        items = json.loads(
            _github_get(
                f"issues/{pr_number}/labels",
                repo=repo,
                token=_github_app_token(env=env),
            )
        )
    except (
        OSError,
        urllib.error.URLError,
        http.client.HTTPException,
        ValueError,
    ):
        return ()
    if not isinstance(items, list):
        return ()
    return tuple(
        item["name"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    )


def upload_with_priority(
    yaml_file: str,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
    job_priority: int,
    nightly: bool,
    runs_tpu: bool = True,
    pr_tpu_concurrency_group: str | None,
    pr_tpu_jobs_per_author: int = PR_TPU_JOBS_PER_AUTHOR,
    env: Mapping[str, str],
) -> None:
    """Upload one pipeline YAML file with priority, TPU filtering, and PR capping."""
    print(f"--- :pipeline: Uploading {yaml_file} with priority {job_priority}")
    raw_yaml = (repo_root / yaml_file).read_text(encoding="utf-8")
    payload = prepare_upload_yaml(raw_yaml, job_priority=job_priority, nightly=nightly)

    if not runs_tpu or pr_tpu_concurrency_group:
        dry_run = _agent(
            "pipeline",
            "upload",
            "--dry-run",
            "--format",
            "json",
            input_text=payload,
            check=False,
            env=env,
            cwd=repo_root,
        )
        if dry_run.returncode == 0 and dry_run.stdout is not None:
            try:
                pipeline = json.loads(dry_run.stdout)
                if not isinstance(pipeline, dict):
                    raise TypeError(
                        "expected JSON object from --dry-run, "
                        f"got {type(pipeline).__name__}"
                    )
                if not runs_tpu:
                    pipeline["steps"] = _strip_tpu_steps(pipeline.get("steps") or [])
                elif pr_tpu_concurrency_group:
                    cap_pr_tpu_concurrency.cap(
                        pipeline.get("steps") or [],
                        pr_tpu_concurrency_group,
                        pr_tpu_jobs_per_author,
                    )
                rendered = json.dumps(pipeline) + "\n"
            except Exception as exc:
                if not runs_tpu:
                    raise BootstrapError(
                        f"ERROR: failed to filter TPU steps for {yaml_file}: {exc}"
                    ) from exc
                print(
                    f"WARNING: failed to cap TPU concurrency for {yaml_file}: {exc}",
                    file=sys.stderr,
                )
            else:
                _agent(
                    "pipeline",
                    "upload",
                    "--no-interpolation",
                    input_text=rendered,
                    env=env,
                    cwd=repo_root,
                )
                return
        if not runs_tpu:
            raise BootstrapError(
                f"ERROR: buildkite-agent --dry-run failed while filtering TPU steps "
                f"for {yaml_file} (exit {dry_run.returncode})"
            )
        print(
            f"WARNING: could not put {yaml_file} in {pr_tpu_concurrency_group}; "
            "uploading it uncapped",
            file=sys.stderr,
        )

    _agent("pipeline", "upload", input_text=payload, env=env, cwd=repo_root)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_bootstrap(
    env: Mapping[str, str] | None = None,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
) -> int:
    """Execute the Buildkite bootstrap flow and return an exit status."""
    child_env = dict(os.environ if env is None else env)
    ctx = BuildContext.from_env(child_env)
    # A post-merge build honors the merged PR's skip-tpu-ci label.
    if ctx.is_post_merge and (pr := parse_merged_pr_number(ctx.message)):
        ctx = replace(
            ctx, labels=_github_labels(pr, repo=ctx.github_repo, env=child_env)
        )
    try:
        ctx.check_suite_labels()
    except BootstrapError as exc:
        _agent(
            "annotate",
            "--style",
            "error",
            "--context",
            "ci-labels",
            str(exc),
            env=child_env,
            cwd=repo_root,
        )
        raise

    # 1. Kind and priority (metadata).
    print("--- Determining job priority")
    kind = build_kind(ctx)
    job_priority, workload_priority = PRIORITIES[kind]
    print(f"Build type: {kind} - Priority: {job_priority}")
    child_env["JOB_PRIORITY"] = str(job_priority)
    _agent(
        "meta-data",
        "set",
        "JOB_PRIORITY",
        str(job_priority),
        env=child_env,
        cwd=repo_root,
    )
    # The kube launcher submits every workload of the build at this Kueue
    # priority class (ci-infra launch.py). None leaves them unclassed, at 0.
    if workload_priority is not None:
        _agent(
            "meta-data",
            "set",
            "WORKLOAD_PRIORITY",
            workload_priority,
            env=child_env,
            cwd=repo_root,
        )
    if kind == "oncall-fix":
        _agent(
            "annotate",
            "--style",
            "warning",
            "--context",
            "oncall-fix",
            "Runs at on-call priority (`oncall-fix` label): "
            "ahead of every other build, including main.",
            env=child_env,
            cwd=repo_root,
        )

    # 2. Skip checks ([skip ci], docs-only).
    print("--- :git: Checking if CI build should be skipped")
    if has_skip_ci_marker(ctx.message):
        print("Commit message contains [skip ci]. Skipping build.")
        return 0
    changed_files = _changed_files(ctx, env=child_env, cwd=repo_root)
    if changed_files:
        print("Files changed:")
        print("\n".join(changed_files))
        if only_docs_changed(changed_files):
            print("Only documentation/non-code files changed. Skipping CI build.")
            return 0
        print("Code files changed. Proceeding with pipeline upload.")

    # 3. vLLM commit.
    print("--- Starting Buildkite Bootstrap")
    base_vllm_hash = read_pinned_vllm_commit(repo_root)
    vllm_commit_hash = ctx.vllm_commit_override or base_vllm_hash
    _agent(
        "meta-data",
        "set",
        "VLLM_COMMIT_HASH",
        vllm_commit_hash,
        env=child_env,
        cwd=repo_root,
    )
    print(f"Using vLLM commit: {vllm_commit_hash}")

    # 4. Suite policy, NIGHTLY export, and concurrency group.
    nightly = ctx.nightly_env or ctx.is_full_ci
    if not ctx.runs_tpu:
        suite_name = "cpu-only"
        suite_about = "TPU steps and Omni Docker image skipped (skip-tpu-ci)"
        child_env["OMNI_IMAGE_SKIP"] = "skip-tpu-ci: no TPU step uses the -omni image"
    elif nightly:
        suite_name = "pr+nightly-suite" if ctx.is_pr else "full"
        suite_about = "Omni Docker image build and full nightly TPU test suite"
    else:
        suite_name = "pr" if ctx.is_pr else "default"
        suite_about = "Omni Docker image build and PR TPU test suite"

    nightly_tag = " [NIGHTLY=1]" if nightly else ""
    suite_note = f"CI suite: {suite_name}{nightly_tag}: {suite_about}"
    print(f"--- :pipeline: Suite {suite_name}{nightly_tag} (priority {job_priority})")
    print(suite_note)
    _agent(
        "annotate",
        "--style",
        "info",
        "--context",
        "ci-suite",
        suite_note,
        env=child_env,
        cwd=repo_root,
    )

    if nightly:
        child_env["NIGHTLY"] = "1"

    # Key the concurrency group on the commit author's email: every webhook
    # build carries it, whereas BUILDKITE_BUILD_CREATOR_EMAIL is missing when
    # the GitHub account is not linked to a Buildkite user.
    if ctx.is_pr and ctx.author_email and ctx.runs_tpu:
        concurrency_group: str | None = f"tpu-pr/{ctx.author_email.lower()}"
    else:
        concurrency_group = (
            (ctx.pr_tpu_concurrency_group_env or None) if ctx.runs_tpu else None
        )
    if concurrency_group is not None:
        child_env["PR_TPU_CONCURRENCY_GROUP"] = concurrency_group

    # 5. Upload .buildkite/pipeline.yml (TPU steps depend on build_docker_omni).
    upload_with_priority(
        _PIPELINE_FILE,
        repo_root=repo_root,
        job_priority=job_priority,
        nightly=nightly,
        runs_tpu=ctx.runs_tpu,
        pr_tpu_concurrency_group=concurrency_group,
        env=child_env,
    )
    print("--- Buildkite Bootstrap Finished")
    return 0


_USAGE = "Usage: bootstrap.py"


def main(argv: Sequence[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(line_buffering=True)

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args:
            raise BootstrapError(_USAGE)
        rc = run_bootstrap()
    except BootstrapError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
