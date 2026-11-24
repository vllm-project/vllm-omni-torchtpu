# Agent Instructions for vLLM-Omni TPU

`vllm-omni-torchtpu` is the Google Cloud TPU platform plugin for `vllm-omni`.

## Repository standards

Every change must follow these documents. Reviewers check PRs against them.

- `docs/developers_guide/coding_guidance.md` (`R-1.x` through `R-8.x` coding
  style, architecture, diff discipline, and test hygiene)
- `CONTRIBUTING.md` (Omni package layout, upstream-first repo boundaries,
  runtime patch rules, `tests/` structure, PR lifecycle, and AI-assisted code
  policy)
- `docs/testing.md` and `.buildkite/README.md` (test markers, CPU checks, and
  Buildkite TPU lanes)

## Agent behavior and must-follow rules

- **Hard constraints (`coding_guidance.md`)**: Treat every `MUST` and `NEVER`
  rule (`R-1.x`–`R-8.x`) as a hard constraint, and every `SHOULD` as the default
  that requires a stated reason to deviate from.
- **Fail closed (`R-7.10`)**: If a task turns out to be a duplicate (`R-7.8`),
  trivial busywork (`R-7.9`), or blocked by a gated-area conflict (`R-2.7`),
  stop and report what is missing instead of forcing a PR through.
- **Maintainer approval before publishing**: Never push branches or open a PR
  without explicit maintainer approval.
- **Human accountability (`R-7.7`)**: Never submit unreviewed code. The human
  contributor must review, understand, and be able to defend every changed line,
  disclose non-trivial AI assistance in the PR description, and include
  `Co-authored-by:` commit trailers alongside the DCO `Signed-off-by:` trailer.
- **No internal references (`R-7.12`)**: Never include `go/` links, `b/` or
  `cl/` IDs, internal source paths, internal hosts, or personal buckets in
  code, comments, YAML, commit messages, or PR text.
- **Verify against pinned versions (`R-2.5`)**: Write against the `vllm-omni`
  and `vllm-torchtpu-core` versions pinned in `pyproject.toml`. Never add
  `hasattr` or `try-import` compatibility shims for newer APIs.

## Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm-omni-torchtpu --comments
gh pr list --repo vllm-project/vllm-omni-torchtpu --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm-omni-torchtpu --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

## Small, single-purpose PRs

Follow [Google's Small CLs guide](https://google.github.io/eng-practices/review/developer/small-cls.html)
and `R-7.1`–`R-7.3` in `docs/developers_guide/coding_guidance.md`. Small,
focused PRs are faster to review thoroughly, make bugs easier to catch, cause
fewer merge conflicts, and are safer to revert.

Before opening a PR, inspect the diff size and scope against the
`vllm-project/vllm-omni-torchtpu` `main` branch (`origin/main` on a direct
clone, or `upstream/main` on a fork where `origin` is the fork):

```bash
git diff --stat origin/main...HEAD
```

- **Size and cohesion (`R-7.1`–`R-7.3`)**: Aim for one self-contained logical
  change within `R-7.1`'s soft target (**<= 300 changed lines**, excluding
  lockfiles).
  - Separate mechanical or behavior-preserving refactors from behavioral
    changes (`R-7.2`), and omit unrelated cleanups or drive-by edits (`R-7.3`).
  - Break larger end-to-end work into independently testable, reviewable steps
    (for example, landing a self-contained building block with its tests before
    broader integration), unless `R-7.1`'s carve-outs apply. When splitting into
    a sequence of PRs, follow the
    [Stacked PRs workflow (reference)](#stacked-prs-workflow-reference).

## Migrating PRs from vllm-torchtpu

When porting an Omni PR from `vllm-project/vllm-torchtpu`:

1. **Read the upstream PR.** Fetch the PR details from
   `vllm-project/vllm-torchtpu`. If it is stacked, port only its own delta
   unless it hard-depends on the parent.
2. **Map paths.**
   - `src/vllm_torchtpu/omni/patches.py` -> `src/vllm_omni_torchtpu/patches/__init__.py`
   - `src/vllm_torchtpu/omni/<mod>.py` -> `src/vllm_omni_torchtpu/<mod>.py`
   - `omni/README.md` -> `docs/omni_plugin.md`; `omni/ROADMAP.md` -> `docs/ROADMAP.md`
   - `tests_omni/<path>` -> `tests/<path>`
3. **Rename.** `vllm_torchtpu.omni` -> `vllm_omni_torchtpu` everywhere (imports,
   logs, asserts, monkeypatch targets). Drop `pytest.mark.omni`; use only markers
   declared in `pyproject.toml`.
4. **Leave core changes in core.** Drop hunks to `vllm_torchtpu/envs.py`,
   `compilation/tpu_compiler.py`, `docs/reference/env_vars.md`, and
   `tests/test_envs_builders.py`. Any core helper you import must exist at both
   `v0.33.0` and the `core-git` pin in `pyproject.toml`.
5. **Env knobs.** Rename `VLLM_TPU_OMNI_*` to `VLLM_OMNI_*` (the only prefix
   `TpuPlatform.validate_environ` exempts from the unread-variable warning).
   Read them with `vllm_torchtpu.envs.env_bool(name, default)()` and document
   them in `docs/omni_plugin.md`. Never add to `envs.environment_variables`,
   because `tpu_compiler.py` hashes every entry into the compile key.
6. **Commit.** One squashed, DCO-signed commit off `origin/main` (or
   `upstream/main`), with the upstream title as subject and
   `vllm-project/vllm-torchtpu#<N>` cited in the body.
7. **Verify.**
   - Run the CI checks in `.github/workflows/ci.yml`.
   - No stale references: grep for `vllm_torchtpu.omni`, `\btests_omni\b`
     (anchored so it doesn't match `unit_tests_omni_*` in Buildkite),
     `mark.omni`, `envs.VLLM_TPU_OMNI`.
   - CPU tests: run `pytest tests/test_packaging.py` and
     `pytest tests --collect-only -q`. For touched tests on CPU, run with
     `JAX_PLATFORMS=cpu` and `--noconftest` (which skips conftest's
     `VLLM_PLUGINS` default, so set
     `VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models`).
   - TPU: follow the "Unit Tests - Omni (PR)" Buildkite step; stage goldens
     with `.buildkite/scripts/stage_model_snapshot.sh` first.
8. **Never push or open a PR without maintainer approval.**

## Development workflow

### Running pre-commit

Install both hook types in the active project virtual environment:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Run the relevant checks before proposing a PR:

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

### DCO sign-off

Every commit must include a Developer Certificate of Origin sign-off (see `DCO`).
Create signed commits with `git commit --signoff` (or `git commit -s`) and
verify that the commit message contains this trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```

## Stacked PRs workflow (reference)

Use this workflow only when a change needs to be split into a stack of
dependent PRs. Because contributions here are primarily fork-based (`origin` is
your fork, `upstream` is `vllm-project/vllm-omni-torchtpu`), GitHub requires
`--base` to be a branch in `vllm-project/vllm-omni-torchtpu`. Every PR in a
stack must build and pass tests on its own without relying on code added in
later PRs:

- **PR 1** targets `main` (`--base main`) off `upstream/main`.
- **PR 2+ (stacked follow-ups)** also target `main` (`--base main`) so CI runs,
  MUST be opened as **Draft** (`--draft`), and MUST list the stack dependency
  at the top of the PR description (for example, `Depends on #<PR 1>`) until
  every preceding PR merges. Its diff will include earlier PRs' commits until
  they merge into `main`.

```bash
# PR 1 (base change off upstream/main)
git checkout -b feat/part-1 upstream/main
git commit -s -m "..."
git push -u origin feat/part-1
gh pr create --repo vllm-project/vllm-omni-torchtpu \
  --base main --head <your-fork>:feat/part-1

# PR 2 (stacked on PR 1: target main so CI runs, open as Draft)
git checkout -b feat/part-2 feat/part-1
git commit -s -m "..."
git push -u origin feat/part-2
gh pr create --repo vllm-project/vllm-omni-torchtpu \
  --draft --base main --head <your-fork>:feat/part-2
```

### Updating a fork stack

```bash
# 1. When PR 1 gains new commits: rebase PR 2 onto PR 1
git checkout feat/part-2
git rebase feat/part-1
git push --force-with-lease origin feat/part-2

# 2. When rebasing an unmerged stack onto upstream/main (from top branch;
#    --update-refs rewrites both feat/part-1 and feat/part-2)
git fetch upstream main
git checkout feat/part-2
git rebase --update-refs upstream/main
git push --force-with-lease origin feat/part-1 feat/part-2

# 3. After PR 1 is squash-merged into main: drop PR 1's commits, push,
#    and mark PR 2 ready for review once previous PRs are merged
git fetch upstream main
git rebase --onto upstream/main feat/part-1 feat/part-2
git push --force-with-lease origin feat/part-2
gh pr ready <pr2_number> --repo vllm-project/vllm-omni-torchtpu
```

*(If working directly on branches in `vllm-project/vllm-omni-torchtpu`, you may
alternatively target `--base feat/part-1` and retarget with
`gh pr edit <pr2_number> --base main` after PR 1 merges.)*
