#!/usr/bin/env bash
# Sync contxt to yompute under /data/contxt and run the full KV experiment matrix.
set -euo pipefail

YOMPUTE_HOST="${YOMPUTE_HOST:-root@192.168.4.51}"
REMOTE_CONTXT="${REMOTE_CONTXT:-/data/contxt}"
REMOTE_CHECKOUT="${REMOTE_CONTXT}/checkout"
REMOTE_RUNS="${REMOTE_CONTXT}/kv-cache-runs"
HF_CACHE_HOST_PATH="${HF_CACHE_HOST_PATH:-/data/diffusion-ontop/datasets/hf_cache}"
HF_OFFLINE="${HF_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Repo:       ${REPO_ROOT}"
echo "Remote:     ${YOMPUTE_HOST}:${REMOTE_CONTXT}"

ssh "${YOMPUTE_HOST}" "mkdir -p '${REMOTE_CHECKOUT}' '${REMOTE_RUNS}' && printf '%s\n' '# contxt' > '${REMOTE_CONTXT}/README.md'"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude 'outputs/' \
  --exclude '.cursor/' \
  "${REPO_ROOT}/" "${YOMPUTE_HOST}:${REMOTE_CHECKOUT}/"

OFFLINE_ARGS=()
if [[ "${HF_OFFLINE}" == "1" ]]; then
  OFFLINE_ARGS+=( -e "TRANSFORMERS_OFFLINE=1" )
fi

MOUNT_HF=()
if ssh "${YOMPUTE_HOST}" "test -d '${HF_CACHE_HOST_PATH}'"; then
  MOUNT_HF+=( -v "${HF_CACHE_HOST_PATH}:/workspace/data/hf_cache" )
  OFFLINE_ARGS+=( -e "HF_HOME=/workspace/data/hf_cache" )
  echo "HF cache mount: ${HF_CACHE_HOST_PATH} -> /workspace/data/hf_cache"
else
  echo "WARN: HF cache dir missing on host: ${HF_CACHE_HOST_PATH} (models may download from network)"
fi

echo "Starting docker matrix on ${YOMPUTE_HOST} …"
REMOTE_LOG="${REMOTE_RUNS}/matrix_$(date +%Y%m%d_%H%M%S).log"

MATRIX_ENV=()
if [[ -n "${MATRIX_MODELS:-}" ]]; then
  MATRIX_ENV+=( -e "MATRIX_MODELS=${MATRIX_MODELS}" )
fi

if [[ "${QUEUE_BACKGROUND:-0}" == "1" ]]; then
  echo "Background mode: log -> ${REMOTE_LOG}"
  ssh "${YOMPUTE_HOST}" "nohup docker run --rm --gpus all \
    -v '${REMOTE_CHECKOUT}:/workspace' \
    -v '${REMOTE_RUNS}:${REMOTE_RUNS}' \
    ${MOUNT_HF[@]+"${MOUNT_HF[@]}"} \
    -e CONTXT_KV_ROOT='${REMOTE_RUNS}' \
    ${MATRIX_ENV[@]+"${MATRIX_ENV[@]}"} \
    ${OFFLINE_ARGS[@]+"${OFFLINE_ARGS[@]}"} \
    -w /workspace \
    yompute/pytorch-gpu \
    bash -lc 'set -euo pipefail
      python3 -m pip install --quiet transformers==4.44.2 sentencepiece
      python3 experiments/kv-cache-shortcuts/run_kv_matrix.py' \
    >'${REMOTE_LOG}' 2>&1 &"
  echo "Queued. Tail: ssh ${YOMPUTE_HOST} 'tail -f ${REMOTE_LOG}'"
  exit 0
fi

ssh "${YOMPUTE_HOST}" "docker run --rm --gpus all \
  -v '${REMOTE_CHECKOUT}:/workspace' \
  -v '${REMOTE_RUNS}:${REMOTE_RUNS}' \
  ${MOUNT_HF[@]+"${MOUNT_HF[@]}"} \
  -e CONTXT_KV_ROOT='${REMOTE_RUNS}' \
  ${MATRIX_ENV[@]+"${MATRIX_ENV[@]}"} \
  ${OFFLINE_ARGS[@]+"${OFFLINE_ARGS[@]}"} \
  -w /workspace \
  yompute/pytorch-gpu \
  bash -lc 'set -euo pipefail
    python3 -m pip install --quiet transformers==4.44.2 sentencepiece
    python3 experiments/kv-cache-shortcuts/run_kv_matrix.py'"

echo "Done. Outputs under ${YOMPUTE_HOST}:${REMOTE_RUNS}/"
