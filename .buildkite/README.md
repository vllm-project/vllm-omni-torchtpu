# Buildkite TPU Pipeline (Inert Until Attached)

**Status: Inert.** No Buildkite pipeline or TPU agent queue is attached to this
repository yet. These files do not run on pull requests or pushes until
maintainers connect a Buildkite pipeline backed by the `cpu_64_core` image
builder queue and the Kueue TPU (`kube`) fleet.

## Pipeline Steps (`pipeline.yml`)

Maintainers ported [`pipeline.yml`](pipeline.yml) as a single self-contained
pipeline from `vllm-project/vllm-torchtpu` (`.buildkite/pipeline_build.yml` and
`.buildkite/pipeline_tests.yml`). All YAML anchors live at the top of
`pipeline.yml`.

| Step key | Condition | Shape / queue | Marker filter and timeout |
| :--- | :--- | :--- | :--- |
| `build_docker_omni` | Always (unless `OMNI_IMAGE_SKIP` is set) | `cpu_64_core` | Builds `docker/Dockerfile` (`--target omni`) and publishes `CI_IMAGE_TAG-omni`. |
| `unit_tests_omni_pr` | `NIGHTLY != "1"` | `tpu7x-standard-1t/1x1x1` (`kube`) | `tests_omni/**/{test_*,*_test}.py`, `-m 'not nightly'`, `--timeout=1200`. |
| `unit_tests_omni_nightly` | `NIGHTLY == "1"` | `tpu7x-standard-1t/1x1x1` (`kube`) | `tests_omni/**/{test_*,*_test}.py`, `-m 'not multichip'`, `--timeout=1800`. |
| `unit_tests_omni_multichip_nightly` | `NIGHTLY == "1"` | `tpu7x-standard-4t/2x2x1` (`kube`) | `tests_omni/**/{test_*,*_test}.py`, `-m 'multichip'`, `--timeout=1800`. |

## Helper Scripts

- `scripts/setup_docker_env.sh`: Builds and pushes the omni CI image from
  `docker/Dockerfile` and records `CI_IMAGE_TAG-omni` in Buildkite metadata.
- `scripts/cleanup_docker.sh`: Cleans up stale containers and images on shared
  Docker builder agents before a new build.
- `scripts/stage_model_snapshot.sh`: Syncs golden reference outputs from
  `gs://tpu-commons-ci/golden/` into `$HF_HOME/golden` inside the TPU pod.
- `kubernetes/run.sh` and `kubernetes/common.sh`: Submit single-pod TPU jobs
  through `/opt/launcher/launch` on the Kueue `kube` queue and forward
  `VLLM_PLUGINS` and Buildkite Test Engine (`bktec`) settings.

## Adapting This Pipeline to a New Organization

The default values preserve parity with `vllm-project/vllm-torchtpu`. Update
these four settings before attaching a Buildkite pipeline to another repository:

- **`IMAGE_REPO`** (`scripts/setup_docker_env.sh`): Point to an Artifact Registry
  or container repository where your builder queue has push access.
- **Agent queues** (`pipeline.yml`): Replace `cpu_64_core` and `kube` (plus
  `KUBE_SHAPE_SINGLE` and `KUBE_SHAPE_MULTI` if needed) with your Buildkite
  builder and TPU runner queues.
- **`suite-slug`** (`pipeline.yml`): Update `tpu-ci` to match your Buildkite
  Test Engine suite slug (or replace `bktec run` with a direct `pytest` call).
- **Golden reference bucket** (`pipeline.yml`): Replace
  `gs://tpu-commons-ci/golden/` with a GCS bucket that your TPU pods can read.
