# AGENTS.md

`vllm-omni-torchtpu` is the Google Cloud TPU platform plugin for `vllm-omni`.

## Migrating PRs from vllm-torchtpu

1. **Read the upstream PR.** Fetch the PR details from `vllm-project/vllm-torchtpu`.
   If it is stacked, port only its own delta unless it hard-depends on the parent.
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
   `TpuPlatform.validate_environ` exempts from the unread-variable warning). Read
   them with `vllm_torchtpu.envs.env_bool(name, default)()` and document them in
   `docs/omni_plugin.md`. Never add to `envs.environment_variables`, because
   `tpu_compiler.py` hashes every entry into the compile key.
6. **Commit.** One squashed, DCO-signed commit off `origin/main`, with the upstream
   title as subject and `vllm-project/vllm-torchtpu#<N>` cited in the body.
7. **Verify.**
   - Run the CI checks in `.github/workflows/ci.yml`.
   - No stale references: grep for `vllm_torchtpu.omni`, `\btests_omni\b`
     (anchored so it doesn't match `unit_tests_omni_*` in Buildkite), `mark.omni`,
     `envs.VLLM_TPU_OMNI`.
   - CPU tests: run touched tests with `JAX_PLATFORMS=cpu` and `--noconftest`.
     That also skips conftest's `VLLM_PLUGINS` default, so set
     `VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models`.
   - TPU: follow the "Unit Tests - Omni (PR)" Buildkite step; stage goldens with
     `.buildkite/scripts/stage_model_snapshot.sh` first.
8. **Never push or open a PR without maintainer approval.**
