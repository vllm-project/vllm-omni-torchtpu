#!/usr/bin/env bash
#
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
# ==============================================================================
# VBench GPU evaluation of videos written by generate_videos.py.
#
# Usage:
#   bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=<bucket> RUN_NAME=<run> [KEY=VALUE ...]
#   bash benchmarks/vbench/run_gpu_eval.sh --ssh GPU_NAME=<gpu-vm> GCS_BUCKET=<bucket> RUN_NAME=<run> [KEY=VALUE ...]
# ==============================================================================

set -euo pipefail

GCS_BUCKET="${GCS_BUCKET:-}"
RUN_NAME="${RUN_NAME:-}"
VIDEO_DIR="${VIDEO_DIR:-}"
GCS_VIDEO_DIR="${GCS_VIDEO_DIR:-}"
GCS_RESULTS_DIR="${GCS_RESULTS_DIR:-}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-true}"
ALLOW_MISSING="${ALLOW_MISSING:-false}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-}"
WORK_DIR_FROM_ARG=""

VBENCH_REPO="${VBENCH_REPO:-https://github.com/Vchitect/VBench.git}"
VBENCH_BRANCH="${VBENCH_BRANCH:-master}"
BENCHMARK_JSON="${BENCHMARK_JSON:-VBench_full_info_sub110.json}"
BENCHMARK_JSON_PATH="${BENCHMARK_JSON_PATH:-}"
VBENCH_UTIL_PATH="${VBENCH_UTIL_PATH:-}"

GPU_NAME="${GPU_NAME:-}"
GPU_ZONE="${GPU_ZONE:-}"
GPU_PROJECT="${GPU_PROJECT:-}"
INTERNAL_IP="${INTERNAL_IP:-false}"
TUNNEL_THROUGH_IAP="${TUNNEL_THROUGH_IAP:-false}"
SSH_MODE=false

DIMENSIONS_TEXT="${DIMENSIONS:-}"
DIMENSIONS=()
if [[ -n "${DIMENSIONS_TEXT}" ]]; then
  # shellcheck disable=SC2206
  DIMENSIONS=(${DIMENSIONS_TEXT})
fi

usage() {
  cat <<EOF
Usage: $0 [--ssh] GCS_BUCKET=<bucket> RUN_NAME=<run> [KEY=VALUE ...]

Required:
  RUN_NAME            Run name passed to generate_videos.py.
  GCS_BUCKET          GCS bucket holding the generated videos. Not needed with
                      VIDEO_DIR and UPLOAD_RESULTS=false.

Common options:
  VIDEO_DIR           Local directory of generated videos; skips the GCS download.
  BENCHMARK_JSON      VBench metadata file name (default: ${BENCHMARK_JSON})
  BENCHMARK_JSON_PATH Path of the metadata file (default: next to this script,
                      else gs://<bucket>/<run>/<BENCHMARK_JSON>, else the copy
                      in the VBench checkout).
  DIMENSIONS          Space-separated VBench dimensions (default: all in the JSON)
  BASELINE_RUN_NAME   Earlier run in WORK_DIR to compare the scores against.
  ALLOW_MISSING       Score an incomplete set of videos (default: false)
  GCS_VIDEO_DIR       GCS video prefix (default: \${RUN_NAME}/videos)
  GCS_RESULTS_DIR     GCS results prefix (default: \${RUN_NAME}/vbench_results)
  UPLOAD_RESULTS      Upload results to GCS (default: true; SSH mode uploads from
                      the caller machine)
  WORK_DIR            Working directory on the GPU host (default: \$HOME/vbench_evaluation)
  VBENCH_REPO         VBench repository (default: ${VBENCH_REPO})
  VBENCH_BRANCH       VBench branch (default: ${VBENCH_BRANCH})

SSH options:
  --ssh               Copy this script to a GPU VM and run it there.
  GPU_NAME            GPU VM name.
  GPU_ZONE            GPU VM zone.
  GPU_PROJECT         GCP project of the GPU VM.
  INTERNAL_IP         Connect using the internal IP (default: false)
  TUNNEL_THROUGH_IAP  Connect through IAP (default: false)
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

step() {
  echo "==> $*"
}

parse_args() {
  local arg key value
  for arg in "$@"; do
    case "${arg}" in
      --ssh)
        SSH_MODE=true
        ;;
      --help | -h)
        usage
        exit 0
        ;;
      WORK_DIR=*)
        WORK_DIR_FROM_ARG="${arg#*=}"
        ;;
      DIMENSIONS=*)
        value="${arg#*=}"
        # shellcheck disable=SC2206
        DIMENSIONS=(${value})
        ;;
      *=*)
        key="${arg%%=*}"
        value="${arg#*=}"
        [[ "${key}" =~ ^[A-Z_]+$ ]] || die "Unknown option: ${key} (see --help)"
        printf -v "${key}" '%s' "${value}"
        ;;
      *)
        die "Unexpected argument: ${arg} (expected KEY=VALUE; see --help)"
        ;;
    esac
  done
}

absolute_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "File not found: ${path}"
  local dir base
  dir="$(cd "$(dirname "${path}")" && pwd)"
  base="$(basename "${path}")"
  printf '%s/%s\n' "${dir}" "${base}"
}

require_bool() {
  [[ "${!1}" == "true" || "${!1}" == "false" ]] || die "$1 must be true or false."
}

normalize_config() {
  [[ -n "${RUN_NAME}" ]] || die "RUN_NAME is required. See --help."
  require_bool UPLOAD_RESULTS
  require_bool ALLOW_MISSING
  if [[ -z "${GCS_BUCKET}" ]]; then
    [[ -n "${VIDEO_DIR}" && "${UPLOAD_RESULTS}" == "false" ]] ||
      die "GCS_BUCKET is required unless VIDEO_DIR is set and UPLOAD_RESULTS=false."
  fi

  GCS_BUCKET="${GCS_BUCKET#gs://}"
  GCS_BUCKET="${GCS_BUCKET%/}"
  GCS_VIDEO_DIR="${GCS_VIDEO_DIR:-${RUN_NAME}/videos}"
  GCS_RESULTS_DIR="${GCS_RESULTS_DIR:-${RUN_NAME}/vbench_results}"
  WORK_DIR="${WORK_DIR_FROM_ARG:-${WORK_DIR:-$HOME/vbench_evaluation}}"
  # A quoted or SSH-forwarded WORK_DIR=~/dir arrives with a literal tilde.
  WORK_DIR="${WORK_DIR/#\~/$HOME}"
  # run_local changes into WORK_DIR, so paths below it must be absolute.
  [[ "${WORK_DIR}" == /* ]] || WORK_DIR="${PWD}/${WORK_DIR}"

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  UTIL_SCRIPT="$(absolute_file "${VBENCH_UTIL_PATH:-${SCRIPT_DIR}/gpu_eval_utils.py}")"
  if [[ -z "${BENCHMARK_JSON_PATH}" && -f "${SCRIPT_DIR}/${BENCHMARK_JSON}" ]]; then
    BENCHMARK_JSON_PATH="${SCRIPT_DIR}/${BENCHMARK_JSON}"
  fi
  if [[ -n "${BENCHMARK_JSON_PATH}" ]]; then
    BENCHMARK_JSON_PATH="$(absolute_file "${BENCHMARK_JSON_PATH}")"
  fi

  # The venv and the VBench checkout are shared; everything else is per run,
  # so the results of an earlier run stay around to compare against.
  RUN_DIR="${WORK_DIR}/runs/${RUN_NAME}"
  JSON_DEST="${RUN_DIR}/${BENCHMARK_JSON}"
  # The entries of JSON_DEST that have videos; what VBench evaluates.
  EVAL_JSON="${RUN_DIR}/eval_${BENCHMARK_JSON}"
  DOWNLOAD_DIR="${RUN_DIR}/downloaded_videos"
  VBENCH_VIDEO_DIR="${RUN_DIR}/vbench_videos"
  RESULTS_DIR="${RUN_DIR}/evaluation_results"
}

print_config() {
  echo "=========================================================================="
  echo "Starting VBench evaluation"
  echo "  Run Name:        ${RUN_NAME}"
  if [[ -n "${VIDEO_DIR}" ]]; then
    echo "  Video Source:    ${VIDEO_DIR}"
  else
    echo "  Video Source:    gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
  fi
  if [[ "${UPLOAD_RESULTS}" == "true" ]]; then
    echo "  Results Target:  gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}"
  fi
  echo "  Run Dir:         ${RUN_DIR}"
  echo "  Benchmark JSON:  ${BENCHMARK_JSON}"
  echo "  Dimensions:      ${DIMENSIONS[*]:-(all in ${BENCHMARK_JSON})}"
  echo "=========================================================================="
}

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

quote_command() {
  local command="$1"
  shift
  local arg
  for arg in "$@"; do
    command="${command} $(shell_quote "${arg}")"
  done
  printf '%s\n' "${command}"
}

run_over_ssh() {
  [[ -n "${GPU_NAME}" ]] || die "GPU_NAME must be specified when using --ssh mode."
  [[ -z "${VIDEO_DIR}" ]] || die "VIDEO_DIR is a path on the GPU host; run without --ssh there."
  require_bool INTERNAL_IP
  require_bool TUNNEL_THROUGH_IAP

  local source_script remote_script remote_util remote_json remote_command
  local gcloud_args=()
  [[ -n "${GPU_ZONE}" ]] && gcloud_args+=("--zone=${GPU_ZONE}")
  [[ -n "${GPU_PROJECT}" ]] && gcloud_args+=("--project=${GPU_PROJECT}")
  [[ "${INTERNAL_IP}" == "true" ]] && gcloud_args+=("--internal-ip")
  [[ "${TUNNEL_THROUGH_IAP}" == "true" ]] && gcloud_args+=("--tunnel-through-iap")
  source_script="$(absolute_file "${BASH_SOURCE[0]}")"
  remote_script="/tmp/run_gpu_eval_${USER:-user}_$$.sh"
  remote_util="/tmp/gpu_eval_utils_${USER:-user}_$$.py"
  remote_json="/tmp/vbench_${USER:-user}_$$_${BENCHMARK_JSON}"

  echo "=========================================================================="
  echo "Executing VBench evaluation remotely"
  echo "  GPU Name:    ${GPU_NAME}"
  echo "  Run Name:    ${RUN_NAME}"
  echo "  Video Dir:   gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
  echo "  Results Dir: gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}"
  echo "=========================================================================="

  gcloud compute scp "${source_script}" "${GPU_NAME}:${remote_script}" "${gcloud_args[@]}"
  gcloud compute scp "${UTIL_SCRIPT}" "${GPU_NAME}:${remote_util}" "${gcloud_args[@]}"

  # The results are uploaded from this machine, with the caller's credentials,
  # so the GPU VM's service account only needs to read the bucket.
  local remote_args=(
    "GCS_BUCKET=${GCS_BUCKET}"
    "RUN_NAME=${RUN_NAME}"
    "GCS_VIDEO_DIR=${GCS_VIDEO_DIR}"
    "GCS_RESULTS_DIR=${GCS_RESULTS_DIR}"
    "VBENCH_REPO=${VBENCH_REPO}"
    "VBENCH_BRANCH=${VBENCH_BRANCH}"
    "BENCHMARK_JSON=${BENCHMARK_JSON}"
    "BASELINE_RUN_NAME=${BASELINE_RUN_NAME}"
    "ALLOW_MISSING=${ALLOW_MISSING}"
    "VBENCH_UTIL_PATH=${remote_util}"
    "UPLOAD_RESULTS=false"
  )
  local cleanup_files=("${remote_script}" "${remote_util}")
  if [[ -n "${WORK_DIR_FROM_ARG}" ]]; then
    remote_args+=("WORK_DIR=${WORK_DIR_FROM_ARG}")
  fi
  if [[ ${#DIMENSIONS[@]} -gt 0 ]]; then
    remote_args+=("DIMENSIONS=${DIMENSIONS[*]}")
  fi
  if [[ -n "${BENCHMARK_JSON_PATH}" ]]; then
    gcloud compute scp "${BENCHMARK_JSON_PATH}" "${GPU_NAME}:${remote_json}" "${gcloud_args[@]}"
    remote_args+=("BENCHMARK_JSON_PATH=${remote_json}")
    cleanup_files+=("${remote_json}")
  fi

  remote_command="trap 'rm -f ${cleanup_files[*]}' EXIT; $(quote_command "bash $(shell_quote "${remote_script}")" "${remote_args[@]}")"
  gcloud compute ssh "${GPU_NAME}" "${gcloud_args[@]}" "--command=${remote_command}"

  if [[ "${UPLOAD_RESULTS}" == "true" ]]; then
    # scp resolves a relative remote path against the remote home directory.
    local remote_results_dir="${WORK_DIR_FROM_ARG:-vbench_evaluation}"
    remote_results_dir="${remote_results_dir#\~/}"
    remote_results_dir="${remote_results_dir%/}/runs/${RUN_NAME}/evaluation_results"

    local local_results_parent
    local_results_parent="$(mktemp -d "${TMPDIR:-/tmp}/vbench_results.XXXXXX")"
    step "Copying evaluation results from ${GPU_NAME}:${remote_results_dir}..."
    gcloud compute scp --recurse "${GPU_NAME}:${remote_results_dir}" "${local_results_parent}/" "${gcloud_args[@]}"
    RESULTS_DIR="${local_results_parent}/evaluation_results"
    upload_results
    rm -rf "${local_results_parent}"
  fi
}

python_supports_vbench() {
  "$1" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13)' >/dev/null 2>&1
}

install_system_packages() {
  step "Ensuring build, git, and video codec system packages..."
  local packages=(git curl wget unzip zip python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0)

  if have sudo; then
    { sudo apt-get update -y && sudo apt-get install -y "${packages[@]}"; } || true
  elif have apt-get; then
    { apt-get update -y && apt-get install -y "${packages[@]}"; } || true
  fi
}

ensure_uv() {
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  if ! have uv; then
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || python3 -m pip install --user uv 2>/dev/null || true
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  fi
}

create_venv() {
  step "Setting up Python virtual environment (Python 3.10-3.12)..."
  ensure_uv

  if [[ -d venv ]] && ! python_supports_vbench venv/bin/python3; then
    echo "Recreating venv with compatible Python version."
    rm -rf venv
  fi

  if [[ ! -d venv ]]; then
    local candidate
    for candidate in python3 python3.12 python3.11 python3.10; do
      if have "${candidate}" && python_supports_vbench "${candidate}"; then
        "${candidate}" -m venv venv
        break
      fi
    done
  fi

  if [[ ! -d venv ]] && have uv; then
    uv venv venv --python 3.11 --seed || uv venv venv --python 3.12 --seed || uv venv venv --python 3.10 --seed
  fi

  [[ -d venv ]] || python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  python3 -m pip install --upgrade "pip<25" "setuptools<71" wheel packaging
}

sync_vbench_repo() {
  step "Syncing VBench repository..."
  if [[ ! -d VBench/.git ]]; then
    git clone -b "${VBENCH_BRANCH}" "${VBENCH_REPO}" VBench
  else
    # Drop our patches so the pull applies cleanly; they are reapplied below.
    git -C VBench checkout -- . >/dev/null 2>&1 || true
    git -C VBench fetch origin "${VBENCH_BRANCH}" || true
    git -C VBench checkout "${VBENCH_BRANCH}" || true
    git -C VBench pull origin "${VBENCH_BRANCH}" || true
  fi
}

show_gpu_status() {
  step "Checking NVIDIA GPU and driver status..."
  if have nvidia-smi; then
    nvidia-smi || true
  else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed on GPU VMs."
  fi
}

install_cuda_torch() {
  step "Installing PyTorch with CUDA support..."
  if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    python3 -m pip uninstall -y torch torchvision 2>/dev/null || true
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple ||
      python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple ||
      python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
  fi
  python3 -m pip install "numpy<2"
}

install_vbench_dependencies() {
  step "Installing VBench and evaluation dependencies..."
  (cd VBench && python3 -m pip install --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/cu121)
  python3 -m pip install --no-build-isolation git+https://github.com/openai/CLIP.git --extra-index-url https://download.pytorch.org/whl/cu121
  python3 -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' --extra-index-url https://download.pytorch.org/whl/cu121 || true
  python3 -m pip install pandas tabulate google-cloud-storage tqdm opencv-python decord

  if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null && have nvidia-smi; then
    echo "Reinstalling CUDA-enabled PyTorch after dependency resolution..."
    python3 -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
  fi

  python3 - <<'PY'
import torch
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"==> PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {device_count}")
PY
}

# Builds ${WORK_DIR}/venv and ${WORK_DIR}/VBench once. Later runs only
# re-activate the venv; delete ${WORK_DIR}/venv to rebuild or update VBench.
setup_environment() {
  local ready_marker="venv/.vbench_ready"
  if [[ -f "${ready_marker}" && -d VBench/.git ]]; then
    step "Reusing the VBench environment in ${WORK_DIR}."
    # shellcheck disable=SC1091
    source venv/bin/activate
  else
    install_system_packages
    create_venv
    sync_vbench_repo
    show_gpu_status
    install_cuda_torch
    python3 "${UTIL_SCRIPT}" patch-vbench VBench
    install_vbench_dependencies
    touch "${ready_marker}"
  fi
  # Idempotent, and it fails loudly if upstream VBench no longer matches.
  python3 "${UTIL_SCRIPT}" patch-vbench VBench
}

gcs_cp() {
  if have gcloud && gcloud storage cp "$@"; then
    return 0
  fi
  if have gsutil && gsutil -m cp "$@"; then
    return 0
  fi
  return 1
}

fetch_benchmark_json() {
  step "Locating ${BENCHMARK_JSON}..."
  local vbench_json="${WORK_DIR}/VBench/vbench/${BENCHMARK_JSON}"
  if [[ -n "${BENCHMARK_JSON_PATH}" ]]; then
    cp "${BENCHMARK_JSON_PATH}" "${JSON_DEST}"
  elif [[ -n "${GCS_BUCKET}" ]] && gcs_cp "gs://${GCS_BUCKET}/${RUN_NAME}/${BENCHMARK_JSON}" "${JSON_DEST}" 2>/dev/null; then
    echo "Using the copy that generate_videos.py uploaded with the run."
  elif [[ -f "${vbench_json}" ]]; then
    # VBench ships its complete prompt set, VBench_full_info.json.
    echo "Using ${vbench_json}."
    cp "${vbench_json}" "${JSON_DEST}"
  else
    die "${BENCHMARK_JSON} is not next to this script, in the run's GCS folder or in VBench; pass BENCHMARK_JSON_PATH."
  fi
}

extract_dimensions() {
  if [[ ${#DIMENSIONS[@]} -eq 0 ]]; then
    step "Extracting dimensions from ${EVAL_JSON}..."
    local extracted
    extracted="$(python3 "${UTIL_SCRIPT}" dimensions "${EVAL_JSON}")"
    # shellcheck disable=SC2206
    DIMENSIONS=(${extracted})
  fi

  [[ ${#DIMENSIONS[@]} -gt 0 ]] || die "No VBench dimensions found. Pass DIMENSIONS=\"...\" or check ${EVAL_JSON}."
  echo "Dimensions to evaluate: ${DIMENSIONS[*]}"
}

download_videos() {
  if [[ -n "${VIDEO_DIR}" ]]; then
    [[ -d "${VIDEO_DIR}" ]] || die "VIDEO_DIR=${VIDEO_DIR} is not a directory."
    DOWNLOAD_DIR="$(cd "${VIDEO_DIR}" && pwd)"
    return
  fi
  step "Downloading generated videos from gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/..."
  rm -rf "${DOWNLOAD_DIR}"
  mkdir -p "${DOWNLOAD_DIR}"
  gcs_cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" "${DOWNLOAD_DIR}/" || die "Failed to download videos from GCS."
}

prepare_videos() {
  step "Aligning video filenames for VBench..."
  local args=("${JSON_DEST}" "${DOWNLOAD_DIR}" "${VBENCH_VIDEO_DIR}" "${EVAL_JSON}")
  [[ "${ALLOW_MISSING}" == "true" ]] && args+=("--allow-missing")
  python3 "${UTIL_SCRIPT}" prepare-videos "${args[@]}"
}

run_vbench() {
  step "Running VBench evaluate.py..."
  rm -rf "${RESULTS_DIR}"
  mkdir -p "${RESULTS_DIR}"
  (
    cd "${WORK_DIR}/VBench"
    # One process per GPU, but never more than the smallest dimension has
    # prompts: VBench crashes in a process that gets no prompt.
    local num_gpus max_processes
    num_gpus="$(python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 1)"
    max_processes="$(python3 "${UTIL_SCRIPT}" max-processes "${EVAL_JSON}" "${DIMENSIONS[@]}")"
    local num_processes=$((num_gpus < max_processes ? num_gpus : max_processes))
    local launcher=(python3)
    if [[ "${num_processes}" -gt 1 ]]; then
      launcher=(torchrun "--nproc_per_node=${num_processes}")
    fi
    echo "Evaluating with ${num_processes} process(es) on ${num_gpus} GPU(s)."
    # All processes must agree on the name of the results files (see patch_vbench).
    VBENCH_RUN_TIMESTAMP="$(date +%Y-%m-%d-%H:%M:%S)"
    export VBENCH_RUN_TIMESTAMP
    "${launcher[@]}" evaluate.py \
      --videos_path "${VBENCH_VIDEO_DIR}" \
      --full_json_dir "${EVAL_JSON}" \
      --output_path "${RESULTS_DIR}" \
      --dimension "${DIMENSIONS[@]}" \
      --mode vbench_standard
  )
}

summarize_results() {
  step "VBench scores for ${RUN_NAME}:"
  local args=("${RESULTS_DIR}")
  if [[ -n "${BASELINE_RUN_NAME}" ]]; then
    args+=("--baseline" "${WORK_DIR}/runs/${BASELINE_RUN_NAME}/evaluation_results")
  fi
  python3 "${UTIL_SCRIPT}" summarize "${args[@]}" | tee "${RESULTS_DIR}/summary.md"
}

upload_results() {
  step "Publishing results to gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/..."
  shopt -s nullglob
  local result_files=("${RESULTS_DIR}"/*)
  shopt -u nullglob

  [[ ${#result_files[@]} -gt 0 ]] || die "No result files found in ${RESULTS_DIR}."
  gcs_cp -r "${result_files[@]}" "gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/" || die "Failed to upload evaluation results to GCS."
}

run_local() {
  print_config
  mkdir -p "${RUN_DIR}"
  cd "${WORK_DIR}"

  setup_environment
  fetch_benchmark_json
  download_videos
  prepare_videos
  extract_dimensions
  run_vbench
  summarize_results
  if [[ "${UPLOAD_RESULTS}" == "true" ]]; then
    upload_results
  fi

  echo ""
  echo "=========================================================================="
  echo "VBench evaluation finished successfully!"
  if [[ "${UPLOAD_RESULTS}" == "true" ]]; then
    echo "Results published to: gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/"
  fi
  echo "Local results stored in: ${RESULTS_DIR}/"
  echo "=========================================================================="
}

parse_args "$@"
normalize_config

if [[ "${SSH_MODE}" == "true" ]]; then
  run_over_ssh
else
  run_local
fi
