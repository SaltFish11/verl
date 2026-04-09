from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from verl.utils.torch_functional import logprobs_from_logits

if TYPE_CHECKING:
    from verl.models.tree_attn.tree import TrieNode


def _compute_internal_node_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_idx: int,
    end_idx: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    num_internal = end_idx - start_idx
    if num_internal <= 0:
        return torch.empty(0, device=logits.device, dtype=logits.dtype)

    pred_start, pred_end = start_idx, end_idx
    label_start, label_end = start_idx + 1, end_idx + 1

    pred_logits = logits[pred_start:pred_end]
    labels = input_ids[label_start:label_end]

    if temperature != 1.0:
        pred_logits = pred_logits / temperature

    return logprobs_from_logits(logits=pred_logits, labels=labels)


def _compute_internal_node_logprobs_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_idx: int,
    end_idx: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_internal = end_idx - start_idx
    if num_internal <= 0:
        empty = torch.empty(0, device=logits.device, dtype=logits.dtype)
        return empty, empty

    pred_start, pred_end = start_idx, end_idx
    label_start, label_end = start_idx + 1, end_idx + 1

    pred_logits = logits[pred_start:pred_end]
    labels = input_ids[label_start:label_end]

    if temperature != 1.0:
        pred_logits = pred_logits / temperature

    lp = logprobs_from_logits(logits=pred_logits, labels=labels)
    log_probs_full = torch.log_softmax(pred_logits.float(), dim=-1)
    probs = log_probs_full.exp()
    ent = -(probs * log_probs_full).sum(dim=-1)

    return lp, ent


def _compute_transition_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pred_pos: int,
    label_pos: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    pred_logit = logits[pred_pos : pred_pos + 1]
    label = input_ids[label_pos : label_pos + 1]

    if temperature != 1.0:
        pred_logit = pred_logit / temperature

    return logprobs_from_logits(logits=pred_logit, labels=label).squeeze(0)


def _compute_transition_logprob_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pred_pos: int,
    label_pos: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_logit = logits[pred_pos : pred_pos + 1]
    label = input_ids[label_pos : label_pos + 1]

    if temperature != 1.0:
        pred_logit = pred_logit / temperature

    lp = logprobs_from_logits(logits=pred_logit, labels=label).squeeze(0)
    log_probs_full = torch.log_softmax(pred_logit.float(), dim=-1)
    probs = log_probs_full.exp()
    ent = -(probs * log_probs_full).sum(dim=-1).squeeze(0)

    return lp, ent


def gather_packed_tree_logprobs(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
) -> dict[int, torch.Tensor]:
    if not trie.all_sequence_ids:
        return {}

    results: dict[int, torch.Tensor] = {}
    device = logits.device
    dtype = torch.float
    input_ids = input_ids.squeeze(0)

    node_cache: dict[tuple[int, int], torch.Tensor] = {}
    transition_cache: dict[tuple[int, int], torch.Tensor] = {}

    for seq_id in trie.all_sequence_ids:
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            results[seq_id] = torch.empty(0, device=device, dtype=dtype)
            continue

        logprob_parts: list[torch.Tensor] = []

        for i, (start, end) in enumerate(indices):
            node_key = (start, end)
            if node_key not in node_cache:
                node_cache[node_key] = _compute_internal_node_logprobs(logits, input_ids, start, end, temperature)
            internal_logprobs = node_cache[node_key]
            if internal_logprobs.numel() > 0:
                logprob_parts.append(internal_logprobs)

            next_start = 0
            if i + 1 < len(indices):
                next_start, _ = indices[i + 1]
            trans_key = (end, next_start)
            if trans_key not in transition_cache:
                transition_cache[trans_key] = _compute_transition_logprob(
                    logits, input_ids, end, next_start, temperature
                )
            logprob_parts.append(transition_cache[trans_key].unsqueeze(0))

        if logprob_parts:
            results[seq_id] = torch.cat(logprob_parts, dim=0)
        else:
            results[seq_id] = torch.empty(0, device=device, dtype=dtype)

    return results


def gather_packed_tree_logprobs_entropy(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not trie.all_sequence_ids:
        empty = torch.empty(0, device=logits.device, dtype=torch.float)
        return empty, empty

    logprobs_results: dict[int, torch.Tensor] = {}
    entropy_results: dict[int, torch.Tensor] = {}
    device = logits.device
    dtype = torch.float
    input_ids = input_ids.squeeze(0)

    node_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    transition_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    for seq_id in trie.all_sequence_ids:
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            empty = torch.empty(0, device=device, dtype=dtype)
            logprobs_results[seq_id] = empty
            entropy_results[seq_id] = empty
            continue

        logprob_parts: list[torch.Tensor] = []
        entropy_parts: list[torch.Tensor] = []

        for i, (start, end) in enumerate(indices):
            node_key = (start, end)
            if node_key not in node_cache:
                node_cache[node_key] = _compute_internal_node_logprobs_entropy(
                    logits, input_ids, start, end, temperature
                )
            internal_logprobs, internal_entropy = node_cache[node_key]
            if internal_logprobs.numel() > 0:
                logprob_parts.append(internal_logprobs)
                entropy_parts.append(internal_entropy)

            next_start = 0
            if i + 1 < len(indices):
                next_start, _ = indices[i + 1]
            trans_key = (end, next_start)
            if trans_key not in transition_cache:
                transition_cache[trans_key] = _compute_transition_logprob_entropy(
                    logits, input_ids, end, next_start, temperature
                )
            trans_lp, trans_ent = transition_cache[trans_key]
            logprob_parts.append(trans_lp.unsqueeze(0))
            entropy_parts.append(trans_ent.unsqueeze(0))

        if logprob_parts:
            logprobs_results[seq_id] = torch.cat(logprob_parts, dim=0)
            entropy_results[seq_id] = torch.cat(entropy_parts, dim=0)
        else:
            empty = torch.empty(0, device=device, dtype=dtype)
            logprobs_results[seq_id] = empty
            entropy_results[seq_id] = empty

    logprob = torch.cat([logprobs_results[sid] for sid in trie.all_sequence_ids], dim=0)
    entropy = torch.cat([entropy_results[sid] for sid in trie.all_sequence_ids], dim=0)
    return logprob, entropy


def gather_packed_tree_logprobs_concat(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if not trie.all_sequence_ids:
        return torch.empty(0, device=logits.device, dtype=torch.float)

    logprob_results = gather_packed_tree_logprobs(logits, trie, input_ids, temperature)
    logprob = torch.cat([logprob_results[sid] for sid in trie.all_sequence_ids], dim=0)
    return logprob


def merge_packed_tree_results(
    results_list: list[dict[int, torch.Tensor]],
    batch_size: int,
    max_seq_len: int | None = None,
    padding_value: float = 0.0,
) -> torch.Tensor:
    combined: dict[int, torch.Tensor] = {}
    for results in results_list:
        for seq_id, tensor in results.items():
            if seq_id in combined:
                raise ValueError(f"Duplicate sequence_id {seq_id} found across microbatches")
            combined[seq_id] = tensor

    if not combined:
        device = torch.device("cpu")
        return torch.full((batch_size, max_seq_len or 0), padding_value, device=device)

    first_tensor = next(iter(combined.values()))
    device = first_tensor.device
    dtype = first_tensor.dtype

    if max_seq_len is None:
        max_seq_len = max(t.shape[0] for t in combined.values()) if combined else 0

    output = torch.full((batch_size, max_seq_len), padding_value, dtype=dtype, device=device)

    for seq_id, tensor in combined.items():
        if seq_id >= batch_size:
            raise ValueError(f"sequence_id {seq_id} exceeds batch_size {batch_size}")
        seq_len = min(tensor.shape[0], max_seq_len)
        output[seq_id, :seq_len] = tensor[:seq_len]

    return output
