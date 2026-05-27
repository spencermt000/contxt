# Experiment Design

Identifies structurally redundant transformer layers by analyzing pairwise cosine similarity
of key vectors within each layer's KV cache, then ablates the most- and least-redundant
layers (zeroing their cache entries) to measure output impact. The hypothesis is that layers
with high intra-layer key cosine similarity contribute less unique information and can be
suppressed at lower cost.

Experiment entry point: `experiments/kv-cache-shortcuts/layer.py`
Matrix runner: `experiments/kv-cache-shortcuts/run_kv_matrix.py`
Cross-model analysis: `experiments/kv-cache-shortcuts/analyze_matrix.py`

# Methodology

**Redundancy metric** (`_analyze_kv_cache`): for each layer, key vectors are extracted from
the 4D cache tensor (shape `[batch, heads, seq, dim]`), reshaped to `[seq, heads*dim]`, then
avg pairwise cosine similarity is computed. High cosine → tokens look similar to each other
in this layer's key space → candidate for suppression. Value norms are also recorded (top-5
positions by L2 norm, max V norm position) but cosine similarity drives the ranking.

**Layer plan** (`_analyze_and_plan`): sorts layers by avg pairwise K cosine, extracts top-3
(most redundant) and bottom-3 (least redundant), saves to `layer_plan.json`.

**Ablation** (`_zero_out_layers`): deep-copies the cache, sets selected layers' K and V
tensors to zeros. Two variants are run: zeroing the most-redundant layers and zeroing the
least-redundant layers (expected to hurt more).

**Comparison** (`_compare_from_disk`): uses `distribution_shift_summary` which computes avg
JS divergence, avg KL divergence, and avg top-10 token-overlap between baseline and ablated
teacher-forced logits. Results are not saved to `comparison.json` in this experiment — the
layer plan lives in `layer_plan.json`. `analyze_matrix.py` reads `layer_plan.json` directly.

**VRAM management**: `nuke_vram()` takes no meaningful arguments; callers null their own
refs before calling. Each model load/unload cycle is isolated.

**Model loading**: same `load_model` from `experiments/shared/model_loader.py` as the other
experiments. Auto 8-bit applies for ≥3B param models on CUDA.

**Output layout** (via `checkpoint_io.run_dir`):
- `KV_MODEL_RUN_ROOT` set → `{KV_MODEL_RUN_ROOT}/layer/`
- Otherwise → `outputs/kv-cache-shortcuts/layer/{sanitized_model_name}/`

Artifacts per run:
```
baseline/
  kv_cache.pt
  teacher_logits.pt
  rollout_tokens.json
  meta.json
layer_plan.json
ablation_most_redundant/
  kv_cache.pt, teacher_logits.pt, rollout_tokens.json, meta.json
ablation_least_redundant/
  kv_cache.pt, teacher_logits.pt, rollout_tokens.json, meta.json
```

**Matrix runner** (`run_kv_matrix.py`): runs all four experiments (quantization, layer,
sliding, compressed_attention) per model sequentially. Layer experiment can be skipped with
`--skip layer`. Model list overridable via `MATRIX_MODELS` env var.

**Cross-model analysis** (`analyze_matrix.py`):
Reads `layer_plan.json` files across all model dirs and prints a layer redundancy table plus
cross-model vote counts (how many models flagged each layer index as most redundant). No GPU
required. Usage:

```
python3 experiments/kv-cache-shortcuts/analyze_matrix.py --kv-root /data/contxt/kv-cache-runs
```

**Related experiment** (`experiments/context-management/compressed_attention.py`):
A token-merging approach that reduces sequence length rather than zeroing cache entries.
Complements the layer-ablation perspective by attacking redundancy at the token level
rather than the layer level.

# Results

### Qwen_Qwen2.5-0.5B-Instruct

Most redundant layers (highest avg pairwise K cosine): [8, 1, 2]
Least redundant layers: [16, 9, 11]

Zeroing layers 8, 1, 2 is expected to have low impact; zeroing 16, 9, 11 should degrade
output more substantially. Quantitative ablation metrics (JS/KL/top-10 overlap) pending
full run.

### Qwen_Qwen2.5-Coder-1.5B

Most redundant layers: [0, 1, 15]
Least redundant layers: [27, 5, 14]

The most-redundant set differs significantly from the 0.5B model: layer 0 (embedding
projection) and layer 15 (mid-network) appear redundant in the Coder model. Layer 27
(near final) is among the least redundant — consistent with later layers carrying more
task-specific representations.

Cross-model overlap: layers 1 appears in both models' most-redundant sets, suggesting
early-layer redundancy may generalize.

### bigcode_starcoder2-3b

Pending.

### deepseek-ai_deepseek-coder-1.3b-base

Pending.

### Qwen_Qwen2.5-3B-Instruct

Pending.

# Discussion

Layer redundancy rankings differ between models, but early layers (0, 1, 2) appear
consistently more redundant than later layers. The Coder-1.5B model's most-redundant set
skews slightly later (layer 15) compared to the 0.5B base model, possibly reflecting
domain-specific encoding differences.

The practical implication: selective layer suppression may be viable for inference
optimization if the most-redundant layers can be identified cheaply (from a short calibration
forward pass) and their suppression is validated against a quality threshold. Combining this
with sliding quantization (lower bits for redundant layers) is a natural extension.
