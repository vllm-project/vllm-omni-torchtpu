# Buildkite TPU Pipeline

`scripts/bootstrap.sh` (`scripts/bootstrap.py`) uploads [`pipeline.yml`](pipeline.yml)
to run `vllm-omni-torchtpu` image builds and TPU tests in the `vllm` Buildkite
organization.

## Buildkite Pipeline Configuration

Configure the Buildkite pipeline for `vllm-project/vllm-omni-torchtpu` with one
bootstrap step on the `cpu` queue:

```yaml
steps:
  - label: ":pipeline: Bootstrap Pipeline"
    key: "bootstrap"
    agents:
      queue: "cpu"
    commands:
      - bash .buildkite/scripts/bootstrap.sh
```

`bootstrap.py` skips builds when a commit message contains `[skip ci]` or
`[ci skip]`, or when a change touches only documentation files.

## Pipeline Steps (`pipeline.yml`)

Maintainers ported [`pipeline.yml`](pipeline.yml) from
`vllm-project/vllm-torchtpu` (`.buildkite/pipeline_build.yml` and
`.buildkite/pipeline_tests.yml`). Each TPU test step depends on
`build_docker_omni`.

| Step key | Condition | Shape and queue | Marker filter and timeout |
| :--- | :--- | :--- | :--- |
| `build_docker_omni` | Always | `cpu_64_core` | Builds `docker/Dockerfile` (`--target omni`) and publishes `CI_IMAGE_TAG-omni`. |
| `unit_tests_omni_pr` | `NIGHTLY != "1"` | `tpu7x-standard-1t/1x1x1` (`kube`) | `tests/**/{test_*,*_test}.py`, `-m 'not nightly'`, `--timeout=1200`. |
| `unit_tests_omni_nightly` | `NIGHTLY == "1"` | `tpu7x-standard-1t/1x1x1` (`kube`) | `tests/**/{test_*,*_test}.py`, `-m 'not multichip'`, `--timeout=1800`. |
| `unit_tests_omni_multichip_nightly` | `NIGHTLY == "1"` | `tpu7x-standard-4t/2x2x1` (`kube`) | `tests/**/{test_*,*_test}.py`, `-m 'multichip'`, `--timeout=1800`. |

## Shared Capacity with `vllm-torchtpu` (`vllm` Buildkite Org)

This pipeline shares infrastructure and scheduling rules with
`vllm-project/vllm-torchtpu` in the `vllm` Buildkite organization:

- **Queues and shapes**: `bootstrap.sh` runs on `cpu`, `build_docker_omni` runs
  on `cpu_64_core`, and TPU steps run on `kube` (`podTemplate: tpu-launcher`)
  using `tpu7x-standard-1t/1x1x1` and `tpu7x-standard-4t/2x2x1`.
- **Priority classes**: `bootstrap.py` sets `priority:` (`JOB_PRIORITY`) and
  Kueue `WORKLOAD_PRIORITY` to `oncall-fix` (`12`), `post-merge` (`11`),
  `pre-merge` (`6`), `other` (`2`, unclassed), or `nightly` (`1`, unclassed).
- **Per-author PR cap**: `cap_pr_tpu_concurrency.py` puts PR TPU steps in
  `tpu-pr/<author-email>` (`concurrency: 40`, `eager`), sharing one 40-job cap
  across `vllm-torchtpu` and `vllm-omni-torchtpu` PRs.
- **PR labels**: `oncall-fix` selects the `oncall-fix` priority class, and
  `run-full-ci` substitutes `NIGHTLY=1` to run the nightly suite on the PR.
  This pipeline has no `skip-tpu-ci` label because every step builds or uses
  the TPU image. To skip Buildkite entirely, put `[skip ci]` in the commit
  message; GitHub Actions still runs the CPU `lint` and `wheel` checks.
- **Artifact Registry and GCS**: `setup_docker_env.sh` pushes to
  `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-omni-torchtpu`
  (`IMAGE_REPO`), and TPU pods stage golden files from
  `gs://tpu-commons-ci/golden/`.
- **Test Engine suite**: `pipeline.yml` reports to `tpu-ci`, the same suite as
  `vllm-torchtpu`. See [Test Engine Suite](#test-engine-suite).

## Test Engine Suite

`pipeline.yml` hardcodes `suite-slug: tpu-ci`, matching `vllm-torchtpu`. The
omni tests already reported into `tpu-ci` when they lived in `vllm-torchtpu`,
and no file under `tests/` shares a path with a `vllm-torchtpu` test, so test
timings and flaky-test history do not collide.

Before this pipeline can report results, a CI admin must:

1. Allow the `vllm-omni-torchtpu-ci` pipeline in the OIDC policy of the
   `tpu-ci` suite. The `tests` plugin requests a suite OIDC token.
2. Make `/opt/launcher/launch` inject the `tpu-ci` tokens
   (`BUILDKITE_ANALYTICS_TOKEN` and `INGEST_BEARER_TOKEN`) from Secret Manager
   for `vllm-omni-torchtpu-ci` jobs. `kubernetes/common.sh` unsets the agent's
   tokens, so the TPU pod uses only the injected values.

To move to a separate suite later, make three changes together: create the
new Test Engine suite, store its token as a new Secret Manager secret, and
wire that secret into `/opt/launcher/launch` for this pipeline. Changing only
`suite-slug` would still inject the `tpu-ci` tokens inside the TPU pod.

## Helper Scripts

- `scripts/bootstrap.sh` and `scripts/bootstrap.py`: Classify builds, skip
  `[skip ci]` and docs-only changes, set priorities and `VLLM_COMMIT_HASH`, and
  upload `pipeline.yml`.
- `scripts/cap_pr_tpu_concurrency.py`: Assigns `kube` TPU steps on PR builds to
  the author's `tpu-pr/<email>` concurrency group.
- `scripts/setup_docker_env.sh`: Builds and pushes the omni CI image from
  `docker/Dockerfile`, then records `CI_IMAGE_TAG-omni` in Buildkite metadata
  after the push succeeds.
- `scripts/cleanup_docker.sh`: Removes stale `vllm-omni-torchtpu` containers and
  images on shared Docker builder agents before a build.
- `scripts/stage_model_snapshot.sh`: Syncs golden reference outputs from
  `gs://tpu-commons-ci/golden/` into `$HF_HOME/golden` inside the TPU pod.
- `kubernetes/run.sh` and `kubernetes/common.sh`: Submit single-pod TPU jobs
  through `/opt/launcher/launch` and forward `VLLM_PLUGINS` and Buildkite Test
  Engine (`bktec`) settings.
