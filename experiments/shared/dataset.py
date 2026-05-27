from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from .eval_utils import compute_rank_of_target


DEFAULT_PROMPTS_TRAIN_PATH = "/data/diffusion-ontop/datasets/prompts_diverse/train.jsonl"
DEFAULT_PROMPTS_VAL_PATH = "/data/diffusion-ontop/datasets/prompts_diverse/val.jsonl"
DEFAULT_CLASSIFIED_OUTPUT_PATH = "experiments/shared/classified_examples.json"


def _print_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    sep = "-+-".join("-" * w for w in widths)
    print(f"\n{title}")
    print(_fmt_row(list(headers)))
    print(sep)
    for row in rendered_rows:
        print(_fmt_row(row))


def load_real_data(path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "input_text" in row and "expected_output_text" in row:
                records.append(
                    {
                        "input_text": str(row["input_text"]),
                        "expected_output_text": str(row["expected_output_text"]),
                        "source_id": str(row.get("id", f"row_{line_no}")),
                    }
                )
                continue

            # Allow prompt-style rows with "prompt" for later label injection.
            if "prompt" in row:
                records.append(
                    {
                        "input_text": str(row["prompt"]),
                        "expected_output_text": str(row.get("expected_output_text", "")),
                        "source_id": str(row.get("id", f"row_{line_no}")),
                    }
                )
                continue

            raise ValueError(
                f"{path}:{line_no} must include either "
                "'input_text'+'expected_output_text' or 'prompt'."
            )
    return records


def generate_synthetic_examples(n: int = 100) -> List[Dict[str, str]]:
    capitals = {
        "France": "Paris",
        "Japan": "Tokyo",
        "Canada": "Ottawa",
        "Brazil": "Brasilia",
        "Australia": "Canberra",
        "Kenya": "Nairobi",
        "India": "New Delhi",
        "Germany": "Berlin",
    }
    tools = ["search_docs", "read_file", "run_tests", "open_ticket", "run_command"]
    casual_sentences = [
        "hey can you send me the report soon thanks",
        "yo this build is broken can u fix it asap",
        "i need the numbers from yesterday pls",
    ]
    formal_sentences = [
        "Could you please send me the report by end of day?",
        "Please investigate and fix the failing build as soon as possible.",
        "Please provide yesterday's metrics at your earliest convenience.",
    ]
    function_specs = [
        ("square", "x", "return x * x"),
        ("is_even", "n", "return n % 2 == 0"),
        ("to_upper", "text", "return text.upper()"),
        ("triple", "n", "return n * 3"),
    ]

    examples: List[Dict[str, str]] = []
    for idx in range(n):
        group = idx % 4
        if group == 0:
            fn_name, arg_name, expected_line = function_specs[(idx // 4) % len(function_specs)]
            input_text = (
                "Complete the Python function body with one line.\n"
                f"def {fn_name}({arg_name}):\n"
                "    "
            )
            expected = expected_line
            task_type = "code_completion"
        elif group == 1:
            country = list(capitals.keys())[(idx // 4) % len(capitals)]
            input_text = f"What is the capital of {country}? Answer with just the city name."
            expected = capitals[country]
            task_type = "factual_qa"
        elif group == 2:
            tool_name = tools[(idx // 4) % len(tools)]
            input_text = (
                "Given available tools [search_docs, read_file, run_tests, open_ticket, run_command], "
                f"produce a valid JSON tool call for '{tool_name}' using arguments "
                '{"query":"dependency graph","limit":5} . Return JSON only.'
            )
            expected = json.dumps(
                {"tool": tool_name, "arguments": {"query": "dependency graph", "limit": 5}},
                separators=(",", ":"),
            )
            task_type = "tool_call_formatting"
        else:
            i = (idx // 4) % len(casual_sentences)
            if (idx // 4) % 2 == 0:
                input_text = f"Rewrite this formally:\n{casual_sentences[i]}"
                expected = formal_sentences[i]
                task_type = "style_correction_formal"
            else:
                input_text = f"Rewrite this casually:\n{formal_sentences[i]}"
                expected = casual_sentences[i]
                task_type = "style_correction_casual"

        examples.append(
            {
                "id": f"syn_{idx:04d}",
                "task_type": task_type,
                "input_text": input_text,
                "expected_output_text": expected,
            }
        )

    random.seed(42)
    random.shuffle(examples)
    return examples


def _select_tier(rank: int, prob: float) -> str:
    if rank <= 10 and prob > 0.05:
        return "nudge"
    if rank <= 100 and 0.001 <= prob <= 0.05:
        return "local_update"
    return "full_training"


def _save_classification_json(classification: Mapping[str, Any], output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(classification, f, indent=2)
    print(f"Saved classified dataset to {out_path}")


def split_by_difficulty(
    model: Any,
    tokenizer: Any,
    examples: Iterable[Mapping[str, str]],
    output_path: str = DEFAULT_CLASSIFIED_OUTPUT_PATH,
) -> Dict[str, Any]:
    items = list(examples)
    if not items:
        raise ValueError("examples must not be empty.")

    device = next(model.parameters()).device
    tiers: Dict[str, List[Dict[str, Any]]] = {"nudge": [], "local_update": [], "full_training": []}

    with torch.no_grad():
        for i, item in enumerate(items):
            input_text = str(item["input_text"])
            expected_output_text = str(item["expected_output_text"])
            expected_ids = tokenizer.encode(expected_output_text, add_special_tokens=False)
            if not expected_ids:
                raise ValueError(f"Example index {i} has empty expected_output_text tokenization.")

            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
            )
            first_step_logits = outputs.logits[:, -1, :].squeeze(0)
            entropy = (-(torch.softmax(first_step_logits, dim=-1) * torch.log_softmax(first_step_logits, dim=-1))).sum().item()
            rank, prob = compute_rank_of_target(first_step_logits, expected_ids[0])
            tier = _select_tier(rank, prob)

            record: Dict[str, Any] = {
                "id": str(item.get("id", f"example_{i:04d}")),
                "task_type": item.get("task_type", "unknown"),
                "input_text": input_text,
                "expected_output_text": expected_output_text,
                "expected_first_token_id": int(expected_ids[0]),
                "correct_first_token_rank": int(rank),
                "correct_first_token_prob": float(prob),
                "entropy": float(entropy),
                "tier": tier,
                # Keep full baseline distribution for later before/after comparison.
                "original_logits": first_step_logits.detach().cpu().tolist(),
            }
            tiers[tier].append(record)

    summary_rows = [
        ("nudge", len(tiers["nudge"])),
        ("local_update", len(tiers["local_update"])),
        ("full_training", len(tiers["full_training"])),
        ("total", len(items)),
    ]
    _print_table("Difficulty Split Summary", ["Tier", "Count"], summary_rows)

    payload = {
        "meta": {
            "num_examples": len(items),
            "classification_rules": {
                "nudge": "rank<=10 and prob>0.05",
                "local_update": "rank<=100 and 0.001<=prob<=0.05",
                "full_training": "otherwise",
            },
            "default_prompt_paths": {
                "train": DEFAULT_PROMPTS_TRAIN_PATH,
                "val": DEFAULT_PROMPTS_VAL_PATH,
            },
        },
        "tiers": tiers,
    }
    _save_classification_json(payload, output_path)
    return payload
