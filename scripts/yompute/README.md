# Yompute helpers (contxt)

## Project layout on the GPU host

Default root for this repo’s artifacts:

| Path | Purpose |
|------|---------|
| `/data/contxt/checkout/` | Synced copy of the **contxt** repo (code) |
| `/data/contxt/kv-cache-runs/` | Per-model KV experiment outputs |

Per model (example `Qwen/Qwen2.5-0.5B-Instruct` → slug `Qwen_Qwen2.5-0.5B-Instruct`):

```
/data/contxt/kv-cache-runs/Qwen_Qwen2.5-0.5B-Instruct/
  quantization/
  layer/
  sliding_quantization/
```

## Code sync: GitHub → yompute (default)

The queue script **clones or pulls** `contxt` on the GPU host into `/data/contxt/checkout` from GitHub (shallow `main` by default). Push from your Mac first, then run the queue script.

| Variable | Default | Meaning |
|----------|---------|---------|
| `SYNC_MODE` | `git` | `git` = clone/pull from `CONTXT_GIT_URL`; `rsync` = copy from your Mac (legacy) |
| `CONTXT_GIT_URL` | `https://github.com/spencermt000/contxt.git` | Repo to clone on yompute |
| `CONTXT_GIT_REF` | `main` | Branch to checkout |

## Queue KV matrix (all models)

From your Mac, in the **contxt** repo root (after `git push`):

```bash
./scripts/yompute/queue_contxt_kv_matrix.sh
```

Optional env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `YOMPUTE_HOST` | `root@192.168.4.51` | SSH target |
| `HF_CACHE_HOST_PATH` | `/data/diffusion-ontop/datasets/hf_cache` | Mount as `HF_HOME` for offline HF hub |
| `HF_OFFLINE` | `0` | Set `1` to use `TRANSFORMERS_OFFLINE=1` (needs all models in cache) |
| `QUEUE_BACKGROUND` | `0` | Set `1` to `nohup` the docker run on the host (log under `kv-cache-runs/matrix_*.log`) |
| `MATRIX_MODELS` | (built-in list in `run_kv_matrix.py`) | Comma-separated HF ids; export before `queue_contxt_kv_matrix.sh` to override the default matrix |

The script updates `/data/contxt/checkout` from GitHub, then runs `run_kv_matrix.py` inside `yompute/pytorch-gpu` with GPU enabled.

## DeepSeek instruct (optional)

The matrix uses `deepseek-ai/deepseek-coder-1.3b-base` by default (matches cached hub name on yompute).

To use the shared instruct tree instead:

```bash
ssh root@192.168.4.51 \
  'docker run --rm --gpus all \
    -v /data/contxt/checkout:/workspace \
    -v /data/contxt/kv-cache-runs:/data/contxt/kv-cache-runs \
    -e MODEL_NAME=/data/shared/models/deepseek-coder-1.3b \
    -e CONTXT_KV_ROOT=/data/contxt/kv-cache-runs \
    -w /workspace yompute/pytorch-gpu \
    bash -lc "python3 experiments/kv-cache-shortcuts/run_kv_matrix.py --only /data/shared/models/deepseek-coder-1.3b"'
```

(Adjust `--only` / matrix in `run_kv_matrix.py` if you make that the canonical id.)
