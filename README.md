# contxt

Experiments and utilities for context / KV-cache / model-behavior work.

## Layout

- `experiments/shared/` — shared loaders, eval helpers, dataset prep
- `experiments/kv-cache-shortcuts/` — KV cache quantization, layer analysis, sliding quantization, and `run_kv_matrix.py`
- `scripts/yompute/` — sync + queue runs on a yompute-style GPU host (see `scripts/yompute/README.md`)

## Yompute

From the repo root:

```bash
./scripts/yompute/queue_contxt_kv_matrix.sh
```

Artifacts on the GPU host default to `/data/contxt/` (see yompute README).

## Requirements

See `requirements.txt`. Use a local venv for development.
