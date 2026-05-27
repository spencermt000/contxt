from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


def _as_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[0] == 1:
        return logits.squeeze(0)
    raise ValueError("Expected logits shape [vocab] or [1, vocab].")


def _entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return (-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item()


def _print_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    sep = "-+-".join("-" * width for width in widths)
    print(f"\n{title}")
    print(_fmt_row(list(headers)))
    print(sep)
    for row in rendered_rows:
        print(_fmt_row(row))


def compare_distributions(
    logits_a: torch.Tensor, logits_b: torch.Tensor, top_k: int = 10
) -> Dict[str, Any]:
    logits_a = _as_1d_logits(logits_a)
    logits_b = _as_1d_logits(logits_b)
    if logits_a.shape != logits_b.shape:
        raise ValueError("logits_a and logits_b must have identical shapes.")

    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    eps = 1e-12

    kl_ab = (probs_a * (torch.log(probs_a.clamp_min(eps)) - torch.log(probs_b.clamp_min(eps)))).sum().item()
    kl_ba = (probs_b * (torch.log(probs_b.clamp_min(eps)) - torch.log(probs_a.clamp_min(eps)))).sum().item()
    midpoint = 0.5 * (probs_a + probs_b)
    js_div = 0.5 * (
        (probs_a * (torch.log(probs_a.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps)))).sum()
        + (probs_b * (torch.log(probs_b.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps)))).sum()
    ).item()

    k = min(top_k, logits_a.numel())
    topk_a = torch.topk(logits_a, k=k).indices.tolist()
    topk_b = torch.topk(logits_b, k=k).indices.tolist()
    overlap_count = len(set(topk_a).intersection(topk_b))
    overlap_ratio = overlap_count / float(k)

    result = {
        "kl_a_to_b": kl_ab,
        "kl_b_to_a": kl_ba,
        "js_divergence": js_div,
        "top_k": k,
        "top_k_overlap_count": overlap_count,
        "top_k_overlap_ratio": overlap_ratio,
        "top_k_ids_a": topk_a,
        "top_k_ids_b": topk_b,
    }

    _print_table(
        "Distribution Comparison",
        ["Metric", "Value"],
        [
            ("KL(A||B)", f"{kl_ab:.6f}"),
            ("KL(B||A)", f"{kl_ba:.6f}"),
            ("JS", f"{js_div:.6f}"),
            (f"Top-{k} Overlap", f"{overlap_count}/{k} ({overlap_ratio:.2%})"),
        ],
    )
    return result


def compute_rank_of_target(logits: torch.Tensor, target_token_id: int) -> Tuple[int, float]:
    logits = _as_1d_logits(logits)
    if target_token_id < 0 or target_token_id >= logits.numel():
        raise ValueError("target_token_id is out of vocabulary range.")

    probs = torch.softmax(logits, dim=-1)
    target_logit = logits[target_token_id]
    rank = int((logits > target_logit).sum().item()) + 1
    target_prob = probs[target_token_id].item()
    return rank, target_prob


def _extract_layer_name(param_name: str) -> str:
    markers = [".lora_A", ".lora_B", ".lora_embedding_A", ".lora_embedding_B"]
    for marker in markers:
        if marker in param_name:
            return param_name.split(marker)[0]
    return param_name.rsplit(".", 1)[0] if "." in param_name else param_name


def diff_lora_weights(
    lora_before: Mapping[str, torch.Tensor], lora_after: Mapping[str, torch.Tensor]
) -> Dict[str, Any]:
    all_keys = sorted(set(lora_before.keys()).intersection(lora_after.keys()))
    layer_sq_norms: Dict[str, float] = {}
    total_sq_norm = 0.0

    for key in all_keys:
        before_t = lora_before[key]
        after_t = lora_after[key]
        if before_t.shape != after_t.shape:
            continue

        diff = (after_t.detach().float().cpu() - before_t.detach().float().cpu()).reshape(-1)
        sq_norm = float(torch.dot(diff, diff).item())
        total_sq_norm += sq_norm
        layer_name = _extract_layer_name(key)
        layer_sq_norms[layer_name] = layer_sq_norms.get(layer_name, 0.0) + sq_norm

    per_layer = [
        {"layer": layer_name, "l2_norm": math.sqrt(sq)}
        for layer_name, sq in layer_sq_norms.items()
    ]
    per_layer.sort(key=lambda entry: entry["l2_norm"], reverse=True)
    total_l2 = math.sqrt(total_sq_norm)

    _print_table(
        "LoRA Weight Diff (Top Changed Layers)",
        ["Layer", "L2 Norm"],
        [(entry["layer"], f"{entry['l2_norm']:.6f}") for entry in per_layer[:20]],
    )

    return {"total_l2_norm": total_l2, "per_layer": per_layer, "compared_keys": len(all_keys)}


def save_snapshot(
    lora_state_dict: Mapping[str, torch.Tensor], version_string: str, path: str
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": version_string,
        "state_dict": {k: v.detach().cpu().clone() for k, v in lora_state_dict.items()},
    }
    torch.save(payload, path)
    print(f"Saved snapshot version='{version_string}' to {path}")


def load_snapshot(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Invalid snapshot format; expected dict with 'state_dict'.")
    return payload


def eval_on_examples(
    model: Any,
    tokenizer: Any,
    examples: Iterable[Tuple[str, str]],
) -> Dict[str, Any]:
    examples = list(examples)
    if not examples:
        raise ValueError("examples must not be empty.")

    device = next(model.parameters()).device
    exact_matches = 0
    first_token_ranks: List[float] = []
    entropies: List[float] = []
    correct_token_probs: List[float] = []
    per_example_rows: List[Tuple[Any, ...]] = []

    with torch.no_grad():
        for idx, (input_text, expected_output_text) in enumerate(examples):
            prompt_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            expected_ids = tokenizer.encode(expected_output_text, add_special_tokens=False)

            outputs = model(
                input_ids=prompt_ids,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :].squeeze(0)

            if expected_ids:
                first_rank, _first_prob = compute_rank_of_target(next_logits, expected_ids[0])
                first_token_ranks.append(float(first_rank))

                for token_id in expected_ids:
                    probs = torch.softmax(next_logits, dim=-1)
                    correct_prob = probs[token_id].item()
                    correct_token_probs.append(correct_prob)
                    entropies.append(_entropy_from_logits(next_logits))

                    token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
                    step_outputs = model(
                        input_ids=token_tensor,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_key_values = step_outputs.past_key_values
                    next_logits = step_outputs.logits[:, -1, :].squeeze(0)
            else:
                first_rank = float("nan")

            greedy_steps = max(1, len(expected_ids))
            generated_ids: List[int] = []
            gen_past = outputs.past_key_values
            gen_logits = outputs.logits[:, -1, :].squeeze(0)
            for _ in range(greedy_steps):
                next_id = int(torch.argmax(gen_logits).item())
                generated_ids.append(next_id)
                step_outputs = model(
                    input_ids=torch.tensor([[next_id]], device=device, dtype=torch.long),
                    past_key_values=gen_past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                gen_past = step_outputs.past_key_values
                gen_logits = step_outputs.logits[:, -1, :].squeeze(0)

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            match = generated_text == expected_output_text
            exact_matches += int(match)

            per_example_rows.append(
                (
                    idx,
                    "Y" if match else "N",
                    f"{first_rank:.1f}" if not math.isnan(first_rank) else "n/a",
                    generated_text.replace("\n", "\\n"),
                    expected_output_text.replace("\n", "\\n"),
                )
            )

    accuracy = exact_matches / len(examples)
    avg_rank = float(sum(first_token_ranks) / len(first_token_ranks)) if first_token_ranks else float("nan")
    avg_entropy = float(sum(entropies) / len(entropies)) if entropies else float("nan")
    avg_correct_prob = (
        float(sum(correct_token_probs) / len(correct_token_probs)) if correct_token_probs else float("nan")
    )

    _print_table(
        "Example-Level Evaluation",
        ["Idx", "Exact", "Rank@1st", "Generated", "Expected"],
        per_example_rows,
    )
    _print_table(
        "Evaluation Summary",
        ["Metric", "Value"],
        [
            ("Accuracy (exact match)", f"{accuracy:.4f}"),
            ("Average rank of correct first token", f"{avg_rank:.4f}" if not math.isnan(avg_rank) else "n/a"),
            ("Average entropy", f"{avg_entropy:.6f}" if not math.isnan(avg_entropy) else "n/a"),
            (
                "Average prob of correct continuation",
                f"{avg_correct_prob:.6f}" if not math.isnan(avg_correct_prob) else "n/a",
            ),
        ],
    )

    return {
        "accuracy": accuracy,
        "avg_rank_of_correct_first_token": avg_rank,
        "avg_entropy": avg_entropy,
        "avg_prob_assigned_to_correct_continuation": avg_correct_prob,
        "num_examples": len(examples),
        "per_example": [
            {
                "index": row[0],
                "exact_match": row[1] == "Y",
                "first_token_rank": None if row[2] == "n/a" else float(row[2]),
                "generated_text": row[3].replace("\\n", "\n"),
                "expected_text": row[4].replace("\\n", "\n"),
            }
            for row in per_example_rows
        ],
    }
