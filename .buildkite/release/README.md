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
