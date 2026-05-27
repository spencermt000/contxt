from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch

from checkpoint_io import pkv_to_device


def rollout_from_cache(
    model: Any,
    first_token_id: int,
    past_key_values: Tuple[Any, ...],
    max_new_tokens: int,
) -> Tuple[List[int], List[torch.Tensor]]:
    device = next(model.parameters()).device
    pkv = pkv_to_device(past_key_values, device)
    token_ids = [first_token_id]
    logits_steps: List[torch.Tensor] = []
    current = torch.tensor([[first_token_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max(max_new_tokens - 1, 0)):
            out = model(
                input_ids=current,
                past_key_values=pkv,
                use_cache=True,
            )
            logits = out.logits[:, -1, :].squeeze(0).detach().cpu()
            logits_steps.append(logits)
            next_id = int(torch.argmax(logits).item())
            token_ids.append(next_id)
            current = torch.tensor([[next_id]], dtype=torch.long, device=device)
            pkv = out.past_key_values
    return token_ids, logits_steps


def teacher_forced_logits(
    model: Any,
    past_key_values: Tuple[Any, ...],
    continuation_ids: Sequence[int],
) -> List[torch.Tensor]:
    if len(continuation_ids) < 2:
        return []
    device = next(model.parameters()).device
    pkv = pkv_to_device(past_key_values, device)
    current = torch.tensor([[continuation_ids[0]]], dtype=torch.long, device=device)
    logits_steps: List[torch.Tensor] = []

    with torch.no_grad():
        for next_token in continuation_ids[1:]:
            out = model(
                input_ids=current,
                past_key_values=pkv,
                use_cache=True,
            )
            logits_steps.append(out.logits[:, -1, :].squeeze(0).detach().cpu())
            current = torch.tensor([[next_token]], dtype=torch.long, device=device)
            pkv = out.past_key_values
    return logits_steps
