#!/usr/bin/env bash
# Copy HF hub model caches from legacy path into /data2/LLMs/hf_cache (idempotent).
set -euo pipefail

YOMPUTE_HOST="${YOMPUTE_HOST:-root@192.168.4.51}"
LEGACY_HF="${LEGACY_HF:-/data/diffusion-ontop/datasets/hf_cache}"
DATA2_LLMS="${DATA2_LLMS:-/data2/LLMs}"
DEST="${DATA2_LLMS}/hf_cache"

echo "Sync HF models: ${YOMPUTE_HOST}:${LEGACY_HF}/hub -> ${DEST}/hub"

ssh "${YOMPUTE_HOST}" "LEGACY_HF='${LEGACY_HF}' DATA2_LLMS='${DATA2_LLMS}' bash -s" <<'REMOTE'
set -euo pipefail
DEST="${DATA2_LLMS}/hf_cache"
mkdir -p "${DEST}/hub"
if [[ ! -d "${LEGACY_HF}/hub" ]]; then
  echo "No legacy hub at ${LEGACY_HF}/hub" >&2
  exit 1
fi
for model_dir in "${LEGACY_HF}/hub"/models--*; do
  [[ -d "${model_dir}" ]] || continue
  name="$(basename "${model_dir}")"
  echo "  rsync ${name} ..."
  rsync -a "${model_dir}/" "${DEST}/hub/${name}/"
done
# Shared hub metadata (locks, version) — merge, do not delete dest-only files.
rsync -a "${LEGACY_HF}/hub/.locks/" "${DEST}/hub/.locks/" 2>/dev/null || true
[[ -f "${LEGACY_HF}/hub/version.txt" ]] && cp -f "${LEGACY_HF}/hub/version.txt" "${DEST}/hub/"

manifest="${DATA2_LLMS}/manifest.txt"
{
  echo "# HF hub models under ${DEST}/hub (updated $(date -Iseconds))"
  for d in "${DEST}/hub"/models--*; do
    [[ -d "${d}" ]] || continue
    du -sh "${d}" | awk -v n="$(basename "${d}")" '{print $1 "\t" n}'
  done
} > "${manifest}"
echo "Wrote ${manifest}"
REMOTE

echo "Done. Future runs: HF_CACHE_HOST_PATH=${DATA2_LLMS}/hf_cache"
