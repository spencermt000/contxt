# contxt

Experiments and utilities for context / KV-cache / model-behavior work.

## Layout

- `experiments/shared/` — shared loaders, eval helpers, dataset prep
- `experiments/kv-cache-shortcuts/` — KV cache quantization, layer analysis, sliding quantization, and `run_kv_matrix.py`
- `scripts/yompute/` — sync + queue runs on a yompute-style GPU host (see `scripts/yompute/README.md`)

## Yompute

Push changes to **GitHub**, then from the repo root:

```bash
./scripts/yompute/queue_contxt_kv_matrix.sh
```

Yompute pulls **`main`** from this repo into `/data/contxt/checkout` by default (see `scripts/yompute/README.md`). Artifacts go under `/data/contxt/kv-cache-runs/`.

## Requirements

See `requirements.txt`. Use a local venv for development.
