# vllm-omni-torchtpu Coding Guidance

Engineering standard for code in `vllm-project/vllm-omni-torchtpu`. It is
normative for two audiences:

- **Human authors** — apply it while writing code; run the author checklist
  (at the end of this doc) before sending a PR.
- **Coding agents** — treat every MUST/NEVER as a hard constraint and every
  SHOULD as the default that requires a stated reason to deviate from.

Reviewers cite these rule IDs (`R-<section>.<n>`) in their findings so feedback
and instructions can point at them precisely.

Two meta-principles organize this document:

1. **Reuse upstream.** `vllm-omni`, `vllm`, and `vllm-torchtpu-core` are the
   first places to look for any functionality. This repo exists to integrate
   `vllm-omni` with TPU, not to re-implement `vllm-omni` or core TPU kernels.
2. **Derive, don't declare.** Any value recoverable from tensor shapes, dtypes,
   hardware APIs, sharding specs, or config metadata is derived at the point of
   use — never duplicated into a config knob, env var, magic number, or
   redundant member variable. (Rationale: see the
   [Appendix](#appendix-rationale-for-derive-dont-declare).)

---

## 1. Python style

- **R-1.1** Follow the
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  for everything the repo tooling does not decide: naming, docstrings
  (Google-style `Args:`/`Returns:`/`Raises:`), comments, exception style,
  default-argument rules, comprehension complexity limits. Never use
  PyTorch-style trailing underscores (`*_`) on function names unless avoiding a
  Python keyword collision.
- **R-1.2** Formatting is decided by the repo's `.pre-commit-config.yaml` and
  `pyproject.toml`, where `ruff` both formats code and sorts imports, and the
  GitHub Actions CI check (`.github/workflows/ci.yml`) gates every PR. If
  `pre-commit install` was run in the clone, hooks fire automatically on each
  commit — but only on staged files, and fresh clones/worktrees may lack the
  install. Before every push, run `pre-commit run --all-files` (or at minimum
  `ruff format --check .` and `ruff check .`). Never use `--no-verify` to push
  code the formatters haven't actually been run on; if hooks can't run locally,
  run `ruff` manually first.
- **R-1.3** `ruff format` wraps code at 88 columns. When a multi-line call or
  literal keeps getting re-wrapped, add a trailing comma after the last element
  so ruff keeps one element per line. `ruff check` fails any line longer than
  88 (`E501`), and the formatter does not split comments or strings, so wrap
  those by hand. End a line with `# noqa: E501` only when it cannot be split,
  such as a table row or a long URL.
- **R-1.4** Explicit imports only — no `from x import *`. Import modules or
  names directly; no aliasing to obscure single letters.
- **R-1.5** Use modern typing: PEP 604 unions (`int | None`, not
  `Optional[int]`), builtin generics (`list[int]`), `StrEnum` for string enums.
  Type-annotate public function signatures.
- **R-1.6** Every new source file carries the full 14-line Apache 2.0 license
  header (copy from a sibling file) and no `SPDX-License-Identifier` tag: the
  open-source release scanner reads the tag instead of the header text. A file
  ported from upstream keeps its `SPDX-FileCopyrightText` notice above the
  license text.
- **R-1.7** No `print` in library code under `src/` — use
  `vllm.logger.init_logger`. Log lines are self-contained and AI-debuggable: tag
  the subsystem, include the relevant shapes/values, and name the failure mode.
  A reader with only the log (no source) should be able to identify root cause
  and the file to inspect. Use `warning_once`/`info_once` for per-process
  one-shot messages (e.g. unexpected fallback paths); log only the branch the
  user didn't expect.
- **R-1.8** **Minimize comments — prefer self-documenting code.** Eliminate
  comments that restate what the code already says; rename or restructure
  instead. Write a comment only for what the code cannot show: a constraint, an
  invariant, or a non-obvious why. Keep docstrings and comments brief and
  direct, and assume the reader is familiar with `vllm-omni` and TPU.

## 2. Architecture ground rules

`vllm-omni-torchtpu` is a **`vllm-omni` platform plugin, not a fork**.
`vllm-omni` and `vllm-torchtpu-core` install as dependencies; this repo injects
`vllm-omni` TPU behavior through entry points (`vllm_omni.platform_plugins`, plus
the `vllm.platform_plugins` coexistence alias documented in `pyproject.toml`).
Core TPU kernels, `TpuPlatform`, and `vllm_torchtpu.envs` remain in
`vllm-torchtpu-core`.

- **R-2.1** Directory responsibilities (all under `src/vllm_omni_torchtpu/`):

  | Directory / Module | Role | Allowed dependencies |
  |---|---|---|
  | `platform.py` | `OmniTpuPlatform` integration surface, backend selection, device setup | `vllm_omni`, `vllm_torchtpu`, `torch`, `torch_tpu` — thin translation only |
  | `worker/` | `OmniTPUModelRunner`, `OmniTPUWorkerBase`, `TPUARWorker`, `TPUARModelRunner`, `TPUGenerationModelRunner` | `vllm_omni`, `vllm`, `vllm_torchtpu`, `torch` — thin translation only |
  | `patches/` | Runtime monkey-patches for `vllm-omni` modules that lack upstream hooks | Targeted `vllm_omni` modules; idempotent, documented with root cause + removal condition |
  | `attention.py`, `rope.py`, `profiler.py` | Shared TPU diffusion attention backends, RoPE helpers, profiler wrapper | Organized by functionality, not by model |

- **R-2.2** **Interface files stay thin.** `platform.py` and `worker/` only
  translate between `vllm-omni`/`torch` and TPU execution. Reusable compute
  belongs in shared subsystem modules (`attention.py`, `rope.py`) or core
  `vllm-torchtpu-core`. No model-specific logic in model-agnostic files.
- **R-2.3** **Core TPU kernels live in `vllm-torchtpu-core`.** Low-level Pallas
  or JAX kernels belong in `vllm-torchtpu-core`, not inside
  `vllm-omni-torchtpu` model or patch modules. Shared helpers take settings as
  arguments rather than reading global env vars inside jitted/compiled regions
  (`R-4.3`).
- **R-2.4** Name modules by **functionality, not by model** (`attention.py`,
  `rope.py`, `profiler.py`). A functional name states what it does, when it is
  useful, and what can be deleted without collateral damage. A model-named file
  (such as `patches/qwen3omni_thinker.py`) is only acceptable for genuinely
  model-specific patches, and it must not become the home for reusable pieces —
  factor those out.
- **R-2.5** **Write against the pinned versions.** The `vllm-torchtpu-core` and
  `vllm-omni` pins in `pyproject.toml` are load-bearing. Any core helper imported
  from `vllm_torchtpu` must exist at both the `vllm-torchtpu-core` floor and the
  `core-git` pin. No `hasattr`/`try-import` compatibility shims for APIs newer
  than the pin — call the pinned API directly. Existing back-compat branches are
  removed when the pin advances, in a dedicated PR, auditing and deleting
  patches that upstream fixed.
- **R-2.6** **Environment mutation belongs in explicit worker setup hooks.**
  Never scatter `os.environ[...] =` or `LIBTPU_INIT_ARGS` edits through model or
  patch code. Per-worker TPU setup belongs in `OmniTpuPlatform.set_device`
  (using `set_libtpu_flag_default` from `vllm_torchtpu.env_override` rather than
  raw `LIBTPU_INIT_ARGS` edits). Code that runs in the `vllm-omni` orchestrator
  process must never open the TPU or query JAX/`torch.accelerator` device state
  (use `torch_tpu._internal.utils.hardware.get_hbm_bytes_per_device()` where HBM
  size is needed without opening a device).
- **R-2.7** **Read the governing section before touching a gated area.** Do not
  modify code in these areas without first reading and following the listed
  rules; if they conflict with the requested change, stop and explain rather
  than proceeding:

  | Area | Read first |
  |---|---|
  | `platform.py`, `worker/` | R-2.2, R-2.6, and [CONTRIBUTING.md §6](../../CONTRIBUTING.md#6-package-layout-and-upstream-api-contracts) (interface files stay thin, orchestrator off TPU, upstream base contracts) |
  | `patches/` | R-3.4, R-3.5, and [CONTRIBUTING.md §7](../../CONTRIBUTING.md#7-runtime-patches-and-plugin-entry-points) (upstream-first, idempotency, 3-part patch comments) |
  | `pyproject.toml` pins and entry points | R-2.5 and `pyproject.toml` coexistence comments (pins are load-bearing) |

## 3. Reuse upstream first

- **R-3.1** **Search order for any new functionality: `vllm-omni` / `vllm`
  upstream -> `vllm-torchtpu-core` -> `vllm-omni-torchtpu` -> write it.**
  Subclass the deepest applicable upstream base class (`OmniPlatform`,
  `TpuPlatform`, `OmniTPUModelRunner`, `OmniTPUWorkerBase`) and override only
  what differs on TPU. Leveraging existing upstream implementations instead of
  copying them is what keeps this plugin maintainable against fast-moving
  upstreams.
- **R-3.2** **No per-model forks of shared layers.** New models extend or
  subclass the common attention/RoPE/worker implementation — never carry private
  copies. If the common path cannot express what a model needs, extend the
  common path.
- **R-3.3** Shared logic has a designated home (`attention.py`, `rope.py`,
  `worker/base.py`, `tests/omni_test_utils.py`). When flagging duplication, name
  the exact target file, not just "dedupe".
- **R-3.4** **When the defect is upstream, fix it upstream** (`vllm-omni`,
  `vllm`, `torch-tpu`, `vllm-torchtpu-core`). A temporary local patch is
  acceptable only with a `TODO` referencing a public upstream issue/PR and
  stating its removal condition.
- **R-3.5** Use framework extension points, not private APIs: `vllm-omni`
  platform plugin hooks, `get_diffusion_attn_backend_cls`, `get_profiler_cls`,
  and documented worker classes. Never gate shared patches on hardcoded model or
  pipeline name allowlists; patch the shared class cleanly with a fallback to
  the original, or key on model/config capabilities.

## 4. Derive, don't declare

- **R-4.1** **No string-based or attribute-probing dispatch.** No model-name
  matching, no `hasattr`/`getattr`-with-default on config objects, no
  dtype-by-string. Use `isinstance`, enums, and direct attribute access that
  fails loudly. Key behavior off capabilities, not names:

  ```python
  # Bad: if upstream renames the field, the default is returned forever —
  # no error, just silently wrong parallelism. And name matching breaks on
  # every finetune/renamed checkpoint.
  pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
  use_mrope = "qwen3" in model_config.model.lower()

  # Good: fails loudly on an upstream rename; keyed on the capability.
  assert isinstance(parallel_config, ParallelConfig)
  pp_size = parallel_config.pipeline_parallel_size
  use_mrope = model_config.uses_mrope
  ```

- **R-4.2** **Derive from argument shapes, not object state.** Logic that
  determines a value purely from tensor shapes/dtypes is robust to upstream
  changes. Question every argument that can be derived from another argument.
- **R-4.3** **No env reads inside jitted/compiled functions.** Closed-over
  constants are a JAX/torch.compile footgun. Read envs in a thin wrapper and
  pass values as arguments, so one glance at the wrapper shows every env that
  affects behavior.
- **R-4.4** **No hard-coded hardware constants.** HBM capacity, cores-per-chip,
  device kind: use runtime queries (`hardware.get_hbm_bytes_per_device()`,
  `get_num_chips()`), never hard-coded sizes or `if "v6e" in name`. Prefer
  self-explanatory constants: `jnp.iinfo(jnp.int16).max`, not `32767`.
- **R-4.5** **No magic mesh-axis strings.** Define and use a shared axis-name
  constant; a hard-coded string breaks silently everywhere when the mesh layout
  changes.
- **R-4.6** **Config flows through `vllm` / `vllm-omni` typed config**, never
  `additional_config` dict-digging wrapped in try/except. If a TPU-only field is
  unavoidable, extend the type rather than stuffing loose attributes onto
  `VllmConfig` or `OmniDiffusionConfig`.
- **R-4.7** **Typed containers over dicts/bools/sentinels.** Dataclass or
  NamedTuple over dict returns; one enum selector over N boolean flags;
  `int | None` over magic sentinel values; mutually-exclusive bools become an
  enum; no `*args/**kwargs` pass-through in internal APIs.

## 5. Fail fast — no silent failure

- **R-5.1** **No defensive try/except, no bare excepts, no `except: pass`.**
  Catch a specific exception only where the pass is provably safe.
  `gc.collect()` and `atexit` hacks are band-aids indicating something else is
  broken — find it.
- **R-5.2** **Assert untested assumptions**, especially around tensor layouts
  and shapes, where the failure mode is silently wrong numbers rather than a
  crash. An assert that fires is infinitely cheaper than a model that quietly
  produces garbage.
- **R-5.3** **Raise `NotImplementedError` + TODO for incomplete features — do not
  silently do the wrong thing.** Use `NotImplementedError` for unsupported
  options, not generic `RuntimeError`. An automatic fallback is acceptable only
  when it preserves correctness at a known perf cost and logs `warning_once`
  (`R-1.7`); when the result would be wrong or the config is unsupported, raise
  an actionable error instead — never fall back.
- **R-5.4** **Write error messages that tell the user the fix**: `"TPUARWorker
  requires vllm-torchtpu-core>=0.34.0 (install --group core-git)"`, not
  `"import error"`.
- **R-5.5** **Do not hide a dtype mismatch with a cast.** Adding
  `v = v.astype(k.dtype)` to make an error go away hides the real bug — find
  and fix the source of the dtype difference first.
- **R-5.6** **Guard recompilation with official APIs**, never override private
  JAX/PyTorch internals without a documented patch comment and removal
  condition.

## 6. Minimize surface area

- **R-6.1** **Question every new argument, flag, and env var.** If there is no
  case where a bool argument takes the other value, remove the argument. Prefer
  implicit enablement from the use case over a user-facing knob.
- **R-6.2** **Add a new env var/flag only as** one of: (a) a backend selector
  during a migration window — one enum value, not N bools, with its deletion
  condition stated; (b) a debug escape hatch that doesn't change defaults; (c) a
  documented quality/perf tradeoff dial whose exact mode stays available. All
  omni env knobs use the `VLLM_OMNI_*` prefix, read locally via
  `vllm_torchtpu.envs.env_bool(name, default)()` and documented in
  `docs/omni_plugin.md`. Never add entries to
  `vllm_torchtpu.envs.environment_variables`, because `tpu_compiler.py` hashes
  every entry into the compile key.
- **R-6.3** **Collapse special cases into existing arguments** when semantics
  allow.
- **R-6.4** **Unify parallelism paths.** One sharding formulation across
  parallelism sizes rather than branching everywhere.
- **R-6.5** **Generalize from the mechanism, not the observed failure.** Gate a
  fix on the variable that actually drives the behavior, not on the one
  model/checkpoint where the bug was seen.
- **R-6.6** **Design for arbitrary checkpoints.** Branch on declared config/model
  metadata, never "works for the official checkpoint only".

## 7. Diff discipline

- **R-7.1** **Small, single-purpose PRs**
  ([Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)).
  Small, focused PRs get faster, more thorough reviews, make bugs easier to spot,
  conflict less, and are safer to revert. Aim for a soft target of <= 300
  changed lines (excluding lockfiles) doing one self-contained logical change.
  Carve-outs: pure deletions, atomic renames/moves with their caller updates,
  tiny dependent follow-ups, and larger cohesive diffs where splitting would be
  artificial may stay in one PR; reviewers may still ask to split smaller
  multi-purpose diffs. Very large designs (>500 lines of core logic) need an
  RFC/design issue first. Each PR in a stack must pass tests on its own, target
  `main` so CI runs, stay **Draft**, and cite its stack dependency (e.g.
  `Depends on #<PR>`) until preceding PRs merge.
- **R-7.2** **Refactor first, feature second.** Land the enabling refactor or
  cleanup as its own behavior-preserving PR before the feature/optimization PR
  that needs it.
- **R-7.3** **Do not include changes unrelated to the PR's purpose.** If a hunk
  is not needed for what the PR is about — reformatting of lines the change
  didn't touch, a rename done in passing, an unrelated typo fix — remove it or
  move it to its own PR. Good-but-off-topic improvements get re-landed
  separately.
- **R-7.4** **Every line must be justified.** The standard probe — for the author
  to preempt and the reviewer to ask — is: *"what happens if this line does not
  exist?"*
- **R-7.5** **No dead code.** Unused dependencies, skipped tests, commented-out
  debug code: delete. Every deliberately-kept alternate path documents why it
  exists and when to use it.
- **R-7.6** Update the PR branch by **rebase, never merge** — a merge commit can
  skip pre-merge CI. WIP goes up as a Draft.
- **R-7.7** **A human reviews every changed line before the PR goes up.**
  However the code was produced, the human submitter must have read the full
  diff, be able to explain and defend each change end-to-end, and have run the
  relevant tests. Never send a diff containing lines nobody can account for.
- **R-7.8** **Check for duplicate work before opening a PR.** Search open
  issues and PRs for the same fix or feature first:

  ```bash
  gh issue view <issue_number> --repo vllm-project/vllm-omni-torchtpu --comments
  gh pr list --repo vllm-project/vllm-omni-torchtpu --state open \
      --search "<short area keywords>"
  ```

  If an open PR already addresses it, don't open another; if your approach is
  materially different, explain the difference on the issue first.

- **R-7.9** **No low-value busywork PRs.** Don't open one-off PRs for a single
  typo, an isolated style fix, or one mutable default. A mechanical cleanup
  lands only when it belongs to the same purpose as substantive work in the same
  PR (`R-7.3`); a cleanup that is neither same-purpose nor substantive enough to
  stand alone is simply not done — drop it (`R-7.10`). This does not ban
  standalone refactor PRs: an enabling refactor per `R-7.2`, or a cleanup PR
  that meaningfully reduces duplication or dead code, is substantive work, not
  busywork.
- **R-7.10** **Fail closed.** If the work turns out to be a duplicate, trivial
  busywork, or blocked by a gated-area conflict (`R-2.7`), stop and report what
  is missing instead of opening the PR anyway. This applies doubly to coding
  agents: returning a short explanation is the correct outcome, not a failure.
- **R-7.11** **PR descriptions name the mechanism, not the trigger.** The title
  states the general capability unlocked, not only the model that prompted it
  (unless it is model-specific); the body is a terse bullet list of mechanisms;
  fix PRs state the original bug and the fix mechanism; perf PRs carry per-mode
  speedups; every PR carries an honest Limitations section when limitations
  exist.
- **R-7.12** **No internal-only references anywhere that lands in the public
  repo** — code, comments, YAML, scripts, commit messages, PR text alike: no
  `go/` links, no `b/` or `cl/` numbers, no internal source paths, no personal
  GCS buckets in scripts or CI, no links to private notes. When code needs to
  cite the reasoning behind a design, inline the reasoning in the comment or
  cite a public source (vLLM/vLLM-Omni source path, HF model card, published
  paper).

## 8. Tests

- **R-8.1** **Every production change under `src/vllm_omni_torchtpu/` needs a
  test that exercises the real behavior** — the forward pass or patch behavior,
  not just imports or AST inspection; error paths too; real `vllm-omni`
  initialization (or `object.__new__(cls)` + `monkeypatch`) over fake module
  trees. Never add `mock_omni` or `_Fake*` classes.
- **R-8.2** **A regression test must fail without the fix.** If reverting the
  fix doesn't turn the test red, the test isn't testing the fix. For patch
  regression tests, exercise the real failing path (`xfail` on the unpatched bug
  rather than canary tests that fail when upstream fixes a bug).
- **R-8.3** **Parametrize, don't clone.** Factor setup/inference/assertion
  helpers out of giant tests; keep forward passes directly in the test body
  (never hidden inside an assertion helper); verify parametrization actually
  flows into the code under test. Use one `assert` per metric/condition with an
  actionable failure message (no compound `assert a and b`).
- **R-8.4** **Verify tests are wired into CI.** Green CI != code exercised —
  check `.buildkite/pipeline.yml` and markers declared in `pyproject.toml`
  (`tpu_test`, `multichip`, `nightly`, `spawns_tpu_workers`). Mark tests
  accurately; do not use `pytest.mark.omni` (that marker belonged to the
  monorepo split).
- **R-8.5** **Tests must run on the smallest hardware that exercises the
  behavior.** Coverage that only exists on multichip/nightly configurations
  doesn't count as PR coverage if a smaller test can verify it.
- **R-8.6** Accuracy and parity tests are generic harnesses: model, config, and
  threshold are arguments, not hard-coded per model. Compare against real
  CPU/fp32 references or saved goldens across all channels with tight, named
  limits — never against TPU tensors or uncalibrated distribution heuristics
  alone.
- **R-8.7** For MagicMock-heavy tests (only where unavoidable): explicitly set
  every attribute the code under test reads. MagicMock auto-creates missing
  attributes, and arithmetic/iteration over a Mock fails silently instead of
  raising.
- **R-8.8** **Design the test before writing it.** Answer four questions first:
  what is the module for, what is its I/O contract, what failure am I guarding
  against, and what is the cheapest level that catches it (unit over integration
  over e2e)? Then: extend existing test files, fixtures (`tests/conftest.py`),
  and helpers (`tests/omni_test_utils.py`) before creating a new file; assert
  observable behavior through public APIs and state intent in the test name or
  docstring; one behavior per test with the smallest setup that triggers it.
- **R-8.9** **No one-off kernel benchmarks in `tests/`.** Correctness belongs in
  `pytest`; perf measurement belongs in benchmarks. A timing-based assert in a
  unit test is a flake generator.
- **R-8.10** **Run evals/parity proactively for model-affecting changes.** Any
  change that can affect output, accuracy, or serving behavior ships with
  before/after parity or eval results (including visual/audio comparisons where
  applicable) in the PR description — don't wait for a reviewer to ask.

---

## Author checklist (run before sending a PR)

1. **Scope** — duplicate-work check done, not busywork (`R-7.8`, `R-7.9`);
   single purpose; refactor pre-landed; stacked base cited; no drive-by
   reformatting (`R-7.x`).
2. **Reuse & placement** — nothing reimplements a `vllm-omni` / `vllm` /
   `vllm-torchtpu-core` utility; code sits in the right module per `R-2.1`;
   interface files thin; no model-specific logic in shared files; dead and
   superseded code deleted (`R-2.x`, `R-3.x`).
3. **Description** — title names the mechanism/capability; mechanism bullet
   list; fix PRs state the bug and the fix mechanism; parity/eval results for
   output-changing PRs; honest Limitations section (`R-7.11`, `R-8.10`).
4. **No new knobs** unless one of `R-6.2`'s three cases applies; omni env knobs
   use `VLLM_OMNI_*` via `vllm_torchtpu.envs.env_bool` and are documented in
   `docs/omni_plugin.md`.
5. **Derived values** — nothing hard-coded that hardware queries, shapes,
   axis-name constants, or config metadata can provide (`R-4.x`).
6. **Fail-fast** — asserts on untested configs; `NotImplementedError` + TODO for
   gaps; actionable errors; no bare excepts (`R-5.x`).
7. **Tests** — exercise real behavior/forward pass; fail without the fix;
   parametrized; one assert per metric; wired into CI with valid markers;
   runnable on small hardware (`R-8.x`).
8. **Hygiene** — `pre-commit run --all-files` clean; `ruff check .` and
   `ruff format --check .` clean; 88 columns; full 14-line Apache 2.0 license
   headers; explicit imports; no internal links (`R-1.x`, `R-7.12`).
9. **Lifecycle** — WIP as Draft; branch updated by rebase, never merge; DCO
   `Signed-off-by:` present; AI assistance disclosed with `Co-authored-by:`
   where applicable; every changed line read and defensible by the human
   submitter (`R-7.6`, `R-7.7`).

---

## Appendix — rationale for "Derive, don't declare"

Every **declared** copy of a derivable value — a config knob, env var, magic
number, or cached member variable — creates an invariant that must be maintained
by hand: the declared value has to stay consistent with the tensor shape,
checkpoint layout, hardware, or mesh it mirrors. Nothing enforces that
consistency; humans and agents do, on every future change. Derived values don't
have this failure mode:

- **Upstream churn.** This repo sits on fast-moving `vllm-omni` and `vllm`. When
  upstream changes a shape, a field, or a layout, code that derives the value at
  the point of use keeps working; a declared copy goes stale with no error.
- **Staleness is silent.** In an inference stack, a stale declared value usually
  doesn't crash — it produces silently wrong numerics or a perf cliff, the most
  expensive class of bug to detect (the same failure mode `R-5.2` guards
  against).
- **Surface area.** Every knob must be documented, tested across its settings,
  and correctly set by every user; a derived value costs none of that (`R-6.1`).
- **Reviewability.** A derivation shows *why* the value is what it is; a literal
  `128` shows nothing, and the reviewer must reconstruct the reasoning to check
  it.

The practical test, for authors and reviewers alike: for any constant, argument,
or config field, ask *"can this be computed from something already in scope — a
shape, a dtype, a hardware query, the mesh, or the config?"* If yes, compute it
there (`R-4.2`–`R-4.7`).
