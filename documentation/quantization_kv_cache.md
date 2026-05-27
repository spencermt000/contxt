# Experiment Design

Measures the effect of uniform post-hoc KV-cache quantization on output quality and memory.
After a full-precision forward pass captures the KV cache, the stored tensors are quantized
to 8/4/2/1 bits using symmetric min-max uniform quantization, then the model decodes from
the modified cache. Rollout and teacher-forcing logits are compared against the fp16 baseline.

Experiment entry point: `experiments/kv-cache-shortcuts/quantization.py`
Matrix runner: `experiments/kv-cache-shortcuts/run_kv_matrix.py`
Cross-model analysis: `experiments/kv-cache-shortcuts/analyze_matrix.py`

# Methodology

**Quantization scheme** (`_uniform_quantize_dequantize`): per-tensor min-max scaling to N-bit
integer levels, dequantized back to the original dtype. Applied independently to every K and V
tensor across all layers (`quantize_kv_cache`).

**Capture phase** (`_capture_baseline`): runs a single forward pass with `use_cache=True`,
saves the full-precision KV cache and the first-token id, then calls `nuke_vram()` before
releasing the model. Callers null their own references before the call; `nuke_vram` only
clears caches and runs GC.

**Variant phase** (`_run_variant_from_disk`): loads quantized cache from disk, loads model
fresh (freeing it via `nuke_vram()` after each variant), runs `rollout_from_cache` and
`teacher_forced_logits`. This keeps peak VRAM bounded to one model at a time.

**Comparison phase** (`_compare_from_disk`): all metrics computed CPU-side from saved
artifacts; results written to `comparison.json` alongside the other run artifacts.

**Metrics**:
- `kl_div_from_baseline` — average per-step KL divergence (rollout logits vs baseline)
- `token_match_rate` — fraction of greedily decoded tokens that match the baseline
- `perplexity` / `perplexity_change` — teacher-forced perplexity on the held-out continuation
- `memory_mb` — theoretical compressed size (elems × bits / 8)

**Model loading** (`experiments/shared/model_loader.py`):
- `LOAD_4BIT=1` / `LOAD_8BIT=1` env flags force bitsandbytes quantization at load time
- If neither flag is set and the model is ≥3B parameters on CUDA, 8-bit loading is enabled
  automatically (`auto_8bit`). This is a model-weight quantization distinct from KV-cache
  quantization; it reduces the model's VRAM footprint during the capture/variant phases.
- On CPU or MPS, quantized loading is silently skipped.
- `DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"`

**Output layout** (via `checkpoint_io.run_dir`):
- `KV_MODEL_RUN_ROOT` set → `{KV_MODEL_RUN_ROOT}/quantization/`
- Otherwise → `outputs/kv-cache-shortcuts/quantization/{sanitized_model_name}/`

**Matrix runner** (`run_kv_matrix.py`):
Iterates over a fixed model list and runs quantization, layer, and sliding experiments
sequentially per model, setting `KV_MODEL_RUN_ROOT` for each. Model list is overridable via
`MATRIX_MODELS` env var (comma-separated) or `--only <model>`. Individual experiments can be
skipped with `--skip quant,layer,sliding,compressed`.

**Cross-model analysis** (`analyze_matrix.py`):
Reads `comparison.json` artifacts from a completed or partial matrix run directory and prints
aggregated tables without requiring a GPU. Covers quantization quality, sliding strategy
rankings, and layer redundancy across all models. Usage:

```
python3 experiments/kv-cache-shortcuts/analyze_matrix.py --kv-root /data/contxt/kv-cache-runs
python3 experiments/kv-cache-shortcuts/analyze_matrix.py --json summary.json
```

**Related experiment** (`experiments/context-management/compressed_attention.py`):
A complementary token-merging approach (compressed attention) that reduces context length
rather than quantizing cache values. Also included in the `run_kv_matrix.py` matrix run.

# Results

### Qwen_Qwen2.5-0.5B-Instruct

8-bit: KL ~0.000x, 100% token match — nearly lossless.
4-bit: KL ~0.000x, 100% token match — lossless on this model size.
Lower bit widths not yet fully analyzed.

### Qwen_Qwen2.5-Coder-1.5B

8-bit: KL < 0.001, 100% token match — lossless.
4-bit: KL 14.4, token match 1.5% — catastrophic degradation. Coder-1.5B is significantly
more sensitive to 4-bit KV quantization than the 0.5B base model.

### bigcode_starcoder2-3b

Pending.

### deepseek-ai_deepseek-coder-1.3b-base

Pending.

### Qwen_Qwen2.5-3B-Instruct

Pending (will use auto 8-bit model loading on CUDA due to ≥3B parameter count).

# Discussion

8-bit KV quantization is safe across both completed models. 4-bit is model-dependent: the
0.5B generalist model tolerates it, but the 1.5B coder model collapses. This suggests that
coding-specialized models encode more information in fine-grained cache precision. Lower
bit widths (2-bit, 1-bit) are expected to degrade further.

The asymmetry between model sizes and domains motivates the sliding-quantization approach
(see `sliding_kv_cache.md`), which applies different bit widths per layer rather than
uniformly across the cache.
