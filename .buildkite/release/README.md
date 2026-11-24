# vllm-omni-torchtpu-release

The `vllm-omni-torchtpu-release` Buildkite pipeline publishes everything that
this repository ships:

- the nightly developer image, `vllm/vllm-omni-torchtpu:nightly` on Docker Hub
  (see [Nightly images](#nightly-images));
- each release's production image, `vllm/vllm-omni-torchtpu:vX.Y.Z` on Docker
  Hub;
- each release's wheel and sdist, `vllm-omni-torchtpu` on PyPI.

The pipeline mirrors `vllm-project/vllm-torchtpu`'s `vllm-torchtpu-release`
pipeline. It runs only in the **TPU Release** Buildkite cluster, on the
`release-vllm-torchtpu` queue. That queue's VM runs as its own service account
(`ci-release`) and refuses jobs from pipelines that are not on its allowlist.
Only that VM can read the Docker Hub token or mint a PyPI token. No CI agent
can publish.

The pipeline settings upload [`pipeline.yml`](pipeline.yml) from the build's
own checkout, and only for builds of `main` and `releases/v*`. A release branch
therefore runs the release tooling that it was cut with. To fix that tooling
for a release, merge the fix to `main` and cherry-pick it to the branch.

## Cut a release

1. Get a full CI run of the release commit. Start a **vllm-omni-torchtpu-ci**
   build of `releases/vX.Y.Z` with `NIGHTLY=1`, or add the `run-full-ci` label
   to a PR. The run has to pass, because the preflight step checks it.
2. Start a **New Build** of **vllm-omni-torchtpu-release** on
   `releases/vX.Y.Z`, at that commit. For a pre-release, set
   `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then runs these steps without anyone's help:
   - **preflight** ([`preflight.sh`](preflight.sh)) settles the version, from
     the branch or `RELEASE_VERSION`. It checks that `vllm-omni-torchtpu-ci`
     passed both nightly TPU steps on the commit
     ([`ci_gate.py`](ci_gate.py)), and that PyPI does not have the version
     yet. It also checks that the matching `vllm-torchtpu` release is out
     (explained in [Pairing with vllm-torchtpu](#pairing-with-vllm-torchtpu)).
   - **build** ([`build_images.sh`](build_images.sh) and
     [`../scripts/build_wheel.sh`](../scripts/build_wheel.sh)) builds the
     `prod` image from this checkout, before anything is on PyPI. It stages
     the image in GAR as `vllm-omni-torchtpu-prod:<version>` and
     `:<date>-<version>`. It also builds a GAR-only `dev` image, and the
     pure-Python wheel and sdist.
   - **check** ([`check_wheels.sh`](check_wheels.sh)) resolves the wheel
     together with `pypi.org` for CPython 3.12 on x86_64 Linux. The check
     fails if the resolution pulls any CUDA wheels.
4. **Approve** the publish block. The block asks whether to move `latest`
   (never for a pre-release) and whether an existing Docker Hub tag may be
   replaced.
5. The build publishes what it built and checked:
   - [`publish_images.sh`](publish_images.sh) copies the staged prod image
     by digest to Docker Hub as `vllm/vllm-omni-torchtpu:vX.Y.Z`, and to
     `latest` if the approver chose it.
   - [`../scripts/publish_wheel.sh`](../scripts/publish_wheel.sh) uploads the
     wheel and sdist to PyPI with Trusted Publishing (Google Cloud OIDC,
     no stored token) and PEP 740 attestations.
   - [`verify_published.sh`](verify_published.sh) then checks, without
     credentials, that an anonymous pull of the tag returns the staged digest.
     It also checks that `vllm-omni-torchtpu==X.Y.Z` resolves from `pypi.org`.

Release engineers can set these build environment variables when something is
broken and they know why:

| Env | Effect |
|---|---|
| `RELEASE_VERSION=vX.Y.Zrc1` | Publish a pre-release under that version. |
| `TORCHTPU_VERSION=X.Y.Z` | Build on this `vllm-torchtpu` release instead of the default. |
| `RELEASE_SKIP_CI_GATE=1` | Skip the `vllm-omni-torchtpu-ci` check (annotated as a warning). |
| `RELEASE_APPROVER_TEAMS=a,b` | Allow only these Buildkite teams to approve. |

## Pairing with vllm-torchtpu

Each `vllm-omni-torchtpu` release builds on one `vllm-torchtpu` release. The
default pairing uses the same `X.Y.Z`, so `vllm-omni-torchtpu` 0.34.0rc1 pairs
with `vllm-torchtpu` 0.34.0. `TORCHTPU_VERSION` overrides the pairing.
Preflight fails unless both of these exist:

- `vllm-torchtpu-core==<torchtpu version>` on PyPI;
- `vllm/vllm-torchtpu:v<torchtpu version>` on Docker Hub.

Preflight pins that image by digest. The `prod` stage of
[`docker/Dockerfile`](../../docker/Dockerfile) installs this package on top of
it. That stage keeps the base image's `vllm-torchtpu-core` and vLLM, takes
`torchaudio` from the PyTorch CPU index, and adds `libgl1` and `libglib2.0-0`
for `opencv-python`. It has none of the CI tooling (`bktec`,
`buildkite-agent`, `gcloud`, `pytest`).

## Nightly images

A daily schedule on `main` runs the `nightly_images_upload` step. That step
uploads [`../pipeline_nightly_image.yml`](../pipeline_nightly_image.yml), which
runs [`../scripts/publish_nightly_images.sh`](../scripts/publish_nightly_images.sh).
The script builds the `dev` stage of `docker/Dockerfile` on
`vllm/vllm-torchtpu:nightly`, pinned by digest. It reinstalls
`vllm-torchtpu-core` at the `vllm-torchtpu` `main` commit and keeps the base
image's vLLM. It then pushes these tags:

- GAR: `vllm-omni-torchtpu-dev:nightly-YYYYMMDD` and `:latest`;
- Docker Hub: `vllm/vllm-omni-torchtpu:nightly-YYYYMMDD-<commit>-<vllm>` and
  `:nightly`.

To try the nightly against scratch targets, set `NIGHTLY_REGISTRY` and
`DOCKERHUB_REPO` on the schedule, or set `NIGHTLY_BASE_IMAGE` to build on
another base image.

## Admin setup

A Buildkite, PyPI, Docker Hub or ci-infra admin must finish these steps before
the first nightly or release. Each item names the system that the admin
changes.

### Buildkite: the release pipeline

1. Create the pipeline **vllm-omni-torchtpu-release** in the `vllm`
   organization, in the **TPU Release** cluster.
2. Set the repository to
   `git@github.com:vllm-project/vllm-omni-torchtpu.git`. The VM checks out
   with a cluster deploy key, so add a read-only deploy key to the GitHub
   repository and store its private half as the cluster secret
   `VLLM_OMNI_TORCHTPU_DEPLOY_KEY`. Alternatively, use the public HTTPS URL.
3. In the GitHub settings of the pipeline, turn off builds on push and on pull
   request. The pipeline runs only from manual builds and the schedule.
4. Set the pipeline steps to:

   ```yaml
   steps:
     - label: ":pipeline: Upload the release pipeline"
       branches: "main releases/v*"
       agents:
         queue: release-vllm-torchtpu
       command: buildkite-agent pipeline upload .buildkite/release/pipeline.yml
   ```

5. Add a schedule on `main`, daily (for example `0 9 * * *` UTC, after the
   `vllm-torchtpu` nightly), for the nightly image.
6. Optional: create a Buildkite team of release approvers, and set
   `RELEASE_APPROVER_TEAMS` in the pipeline environment.

### Buildkite: the CI pipeline

1. On **vllm-omni-torchtpu-ci**, turn on GitHub commit statuses for each job
   ("Update commit statuses" with per-step statuses). `ci_gate.py` reads the
   `buildkite/vllm-omni-torchtpu-ci/<step>` statuses of the nightly steps.
2. Allow builds of `releases/v*` branches, so a release engineer can start a
   `NIGHTLY=1` build of the release commit.

### ci-infra: the release VM

1. Add the ID of `vllm-omni-torchtpu-release` to `allowed_pipeline_ids` of
   `ci_release_vllm_torchtpu` (`release_agents.tf`), so the release VM
   accepts its jobs.
2. Check that the `ci-release` service account can push to
   `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu`.
   The images use the new names `vllm-omni-torchtpu-prod` and
   `vllm-omni-torchtpu-dev`.

### Docker Hub

1. Create the public repository `vllm/vllm-omni-torchtpu` in the `vllm`
   organization.
2. Create an access token with read and write scope on that repository, for
   the `vllm` user.
3. Store the token in the TPU Release cluster as the secret
   `DOCKERHUB_TOKEN_VLLM_OMNI_TORCHTPU`. Make the secret readable by
   `vllm-omni-torchtpu-release` only.

### PyPI

1. Add a Trusted Publisher for the `vllm-omni-torchtpu` project: **Publishing**,
   then **Add a new publisher**, then **Google Cloud**, with the email
   `ci-release@cloud-ullm-inference-ci-cd.iam.gserviceaccount.com`.
2. If the project does not exist yet, add a **pending publisher** under the
   account's publishing page instead. The first upload then creates the
   project.
3. Add the `vllm-omni-torchtpu` maintainers as project owners after the first
   upload.

### Before the first release (v0.34.0)

1. Wait until `vllm-torchtpu` 0.34.0 is on PyPI and on Docker Hub as
   `vllm/vllm-torchtpu:v0.34.0`.
2. In `pyproject.toml`, raise the floor to `vllm-torchtpu-core>=0.34.0`.
   Then drop the `core-git` group and the `CORE_COMMIT` reinstall from CI, if
   the CI image no longer needs them.
3. Pin `vllm-omni` to a release that is on PyPI.
4. Cut `releases/v0.34.0` and follow [Cut a release](#cut-a-release). For the
   first run, consider `RELEASE_VERSION=v0.34.0rc1`.
