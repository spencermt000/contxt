# Experiment Design

Tests layer-wise non-uniform KV-cache quantization strategies, where different layers
receive different bit widths rather than a single global precision. Strategies range from
purely positional (tapered, reverse-tapered) to content-driven (importance-weighted by
value-norm score) to a hybrid that combines quantization with redundancy-based token pruning.

Experiment entry point: `experiments/kv-cache-shortcuts/sliding_quantization.py`
Matrix runner: `experiments/kv-cache-shortcuts/run_kv_matrix.py`
Cross-model analysis: `experiments/kv-cache-shortcuts/analyze_matrix.py`

# Methodology

**Strategies** (`_build_strategy_maps`): four bit-assignment maps are built from the
baseline KV cache, each keyed by layer index:

| Strategy | Description |
|---|---|
| `tapered` | early layers 2-bit, middle 4-bit, late 8-bit |
| `reverse_tapered` | early layers 8-bit, middle 4-bit, late 2-bit |
| `importance_weighted` | layers sorted by avg V-norm; lowest-norm → 2-bit, mid → 4-bit, high → 8-bit |
| `uniform_4bit` | flat 4-bit across all layers (baseline comparison for non-uniform strategies) |

A fifth strategy, `tapered_plus_redundancy_prune`, applies the tapered bit map and then
zeros the trailing 50% of sequence positions in any layer whose avg pairwise K cosine
exceeds 0.9 (high intra-layer redundancy). Its memory estimate is computed as 50% of the
tapered strategy's size.

**Progressive quantization** (`progressive_quantize_kv_cache`): applies per-layer uniform
min-max quantization using the bit width from `layer_bits_map`, defaulting to 8-bit for any
unspecified layer.

**Baseline capture**: reuses `_capture_baseline` from `quantization.py` directly. The
baseline is identical to the quantization experiment's baseline for a given model.

**Comparison** (`_compare_from_disk`): loads strategy results from disk, computes KL
divergence, token match rate, and perplexity vs the full-precision baseline, then writes
`comparison.json`. Memory savings are read from `strategy_plan.json`.

**Metrics**:
- `kl_div` — average per-step KL divergence from baseline rollout
- `token_match` — greedy token match rate
- `memory_mb` / `memory_savings_pct` — estimated compressed size and savings vs fp16 cache
- `perplexity` — teacher-forced perplexity on held-out continuation

**Model loading** (`experiments/shared/model_loader.py`):
- `LOAD_4BIT=1` / `LOAD_8BIT=1` force bitsandbytes quantization at model-weight load time
- Models ≥3B parameters on CUDA auto-enable 8-bit weight loading (`auto_8bit`)
- `nuke_vram()` takes no arguments; callers null their own refs before calling

**Output layout** (via `checkpoint_io.run_dir`):
- `KV_MODEL_RUN_ROOT` set → `{KV_MODEL_RUN_ROOT}/sliding_quantization/`
- Otherwise → `outputs/kv-cache-shortcuts/sliding_quantization/{sanitized_model_name}/`

Artifacts per run:
```
baseline/
  kv_cache.pt, rollout_logits.pt, teacher_logits.pt, rollout_tokens.json, meta.json
strategy_plan.json
tapered/
  kv_cache.pt, rollout_logits.pt, teacher_logits.pt, rollout_tokens.json, meta.json
reverse_tapered/   (same structure)
importance_weighted/
uniform_4bit/
tapered_plus_redundancy_prune/
comparison.json
```

**Matrix runner** (`run_kv_matrix.py`): runs all four experiments per model sequentially.
Sliding experiment can be skipped with `--skip sliding`. Model list overridable via
`MATRIX_MODELS` env var or `--only <model>`.

**Cross-model analysis** (`analyze_matrix.py`):
Reads `comparison.json` from each model's `sliding_quantization/` directory and prints:
- Best strategy per model by lowest KL divergence
- Best strategy per model among those achieving ≥70% memory savings
- Cross-model average/min/max KL per strategy (to identify which strategies generalize)

No GPU required. Usage:
```
python3 experiments/kv-cache-shortcuts/analyze_matrix.py --kv-root /data/contxt/kv-cache-runs
```

**Related experiment** (`experiments/context-management/compressed_attention.py`):
A token-merging approach that reduces KV sequence length via attention-score-based merging,
rather than reducing per-element precision. This is also included in the `run_kv_matrix.py`
matrix run as a fourth experiment type.

# Results

### Qwen_Qwen2.5-0.5B-Instruct

Since 8-bit is lossless on this model, tapered strategies (heavy early-layer compression)
are expected to be viable. The `importance_weighted` strategy may closely track `tapered` if
lower-norm layers happen to be early layers. Full per-strategy breakdown pending.

### Qwen_Qwen2.5-Coder-1.5B

Given that flat 4-bit causes catastrophic degradation (KL 14.4) on this model, strategies
that assign 4-bit or lower to any layer should be evaluated carefully. `reverse_tapered`
(8-bit early, 2-bit late) is the highest-risk profile. `importance_weighted` may be able to
preserve quality by keeping high-norm (high-importance) layers at 8-bit. Full results pending.

### bigcode_starcoder2-3b

Pending. Model will use auto 8-bit weight loading on CUDA.

### deepseek-ai_deepseek-coder-1.3b-base

Pending.

### Qwen_Qwen2.5-3B-Instruct

Pending. Model will use auto 8-bit weight loading on CUDA.

# Discussion

The Coder-1.5B result (4-bit → KL 14.4) establishes that flat 4-bit is a hard floor for
at least some models. Non-uniform strategies that protect the most sensitive layers while
compressing redundant ones are the primary motivation for this experiment.

The `tapered_plus_redundancy_prune` strategy is the most aggressive in terms of memory
reduction — it combines bit-width reduction with sequence-position truncation for
high-cosine layers. Whether its quality holds is the key open question.

Cross-model `analyze_matrix.py` output will show which strategies consistently minimize KL
across architectures, informing a recommended default for production use.
