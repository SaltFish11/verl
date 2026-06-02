from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any
import hashlib

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask

from verl.models.tree_attn.constants import BLOCK_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class TrieNode:
    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.start_idx == -1 and self.end_idx == -1

    @property
    def num_tokens(self) -> int:
        if self.is_root:
            return sum(node.num_tokens for node in self.nodes)
        return len(self.tokens)

    @property
    def tree_indices(self) -> tuple[int, int]:
        return (self.start_idx, self.end_idx)

    @property
    def all_sequence_ids(self) -> list[int]:
        if self.is_root:
            seq_ids: set[int] = set()
            for node in self.nodes:
                seq_ids.update(node.sequence_ids)
            return sorted(seq_ids)
        return sorted(set(self.sequence_ids))

    def get_all_tree_indices(self) -> list[tuple[int, int]]:
        indices = [ancestor.tree_indices for ancestor in self.ancestors]
        if not self.is_root:
            indices.append(self.tree_indices)
        return indices

    def get_sequence_tree_indices(self, seq_id: int) -> list[tuple[int, int]]:
        indices = []
        for node in self.nodes:
            if seq_id in node.sequence_ids:
                indices.append(node.tree_indices)
        return indices


class _BuildNode:
    __slots__ = ("tree_id", "token_id", "node_id", "children", "is_end", "sequence_ids")

    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _BuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _count_additional_nodes(root: _BuildNode, sequence: list[int]) -> int:
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _insert_sequence(
    root: _BuildNode,
    all_nodes: list[_BuildNode],
    sequence: list[int],
    tree_id: int,
    sequence_id: int,
) -> None:
    current = root
    for token in sequence:
        if token not in current.children:
            node_id = len(all_nodes)
            current.children[token] = _BuildNode(tree_id, token, node_id)
            all_nodes.append(current.children[token])
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True


def _compress_trie(root: _BuildNode) -> TrieNode:
    trie_root = TrieNode(tree_id=root.tree_id)

    def _compress_chain(
        node: _BuildNode,
        ancestors: list[TrieNode],
    ) -> TrieNode:
        tokens: list[int] = []
        current = node
        start_id = node.node_id

        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            next_child = next(iter(current.children.values()))
            if current.sequence_ids != next_child.sequence_ids:
                raise ValueError(
                    f"Sequence IDs mismatch along chain: {current.sequence_ids} vs {next_child.sequence_ids}"
                )
            if next_child.node_id != current.node_id + 1:
                raise ValueError("Node IDs not consecutive along chain.")
            current = next_child

        trie_node = TrieNode(
            tree_id=root.tree_id,
            start_idx=start_id,
            end_idx=current.node_id,
            tokens=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestors=ancestors.copy(),
        )
        trie_root.nodes.append(trie_node)

        if current.children:
            for token, child in sorted(current.children.items()):
                trie_node.children[token] = _compress_chain(
                    child,
                    ancestors + [trie_node],
                )

        return trie_node

    if root.children:
        for token, child in sorted(root.children.items()):
            trie_root.children[token] = _compress_chain(child, [])

    return trie_root


def trie_to_parent_array(trie: TrieNode, max_tokens: int) -> torch.Tensor:
    fa = torch.full((1, max_tokens), -1, dtype=torch.int32)

    if not trie.nodes:
        return fa

    for node in trie.nodes:
        parent_end_pos = -1
        if node.ancestors:
            parent_node = node.ancestors[-1]
            parent_end_pos = parent_node.end_idx

        if node.start_idx >= 0 and node.start_idx < max_tokens:
            fa[0, node.start_idx] = parent_end_pos
        for pos in range(node.start_idx + 1, node.end_idx + 1):
            if pos < max_tokens:
                fa[0, pos] = pos - 1

    return fa

def _hash_multimodal_value(value: Any) -> str:
    if value is None:
        return "none"
    if hasattr(value, "data") and not isinstance(value, torch.Tensor):
        value = value.data
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        h = hashlib.blake2b(digest_size=16)
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
        return h.hexdigest()
    if isinstance(value, dict):
        return "{" + "|".join(f"{k}:{_hash_multimodal_value(value[k])}" for k in sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + "|".join(_hash_multimodal_value(v) for v in value) + "]"
    return repr(value)


def _extract_multimodal_signatures(data: dict[str, Any]) -> list[str] | None:
    if "multi_modal_inputs" not in data:
        return None
    return [_hash_multimodal_value(item) for item in data["multi_modal_inputs"]]



def _greedy_build_tries(
    data: dict[str, Any],
    max_tokens_per_tree: int,
) -> tuple[list[TrieNode], list[int]]:
    signatures = _extract_multimodal_signatures(data)
    sequences = _extract_sequences(data)
    forests: list[dict[str, Any]] = []

    for seq_id, seq in enumerate(sequences):
        inserted = False
        signature = signatures[seq_id] if signatures is not None else None

        for tree_id, tree in enumerate(forests):
            if tree["signature"] != signature:
                continue
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(
                    tree["root"],
                    tree["all_nodes"],
                    seq,
                    tree_id,
                    seq_id,
                )
                tree["nodes"] += additional
                inserted = True
                break

        if inserted:
            continue

        if len(seq) > max_tokens_per_tree:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds max_tokens_per_tree "
                f"{max_tokens_per_tree}; adjust limit or split sequences."
            )

        new_tree_id = len(forests)
        new_root = _BuildNode(new_tree_id, -1, -1)
        all_nodes: list[_BuildNode] = []
        _insert_sequence(new_root, all_nodes, seq, new_tree_id, seq_id)
        forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq), "signature": signature})

        # forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq)})

    tries = [_compress_trie(f["root"]) for f in forests]
    num_tokens_list = [f["nodes"] for f in forests]

    return tries, num_tokens_list


def _extract_sequences(data: dict[str, Any]) -> list[list[int]]:
    assert "input_ids" in data, "Input data must contain 'input_ids'"
    assert "attention_mask" in data, "Input data must contain 'attention_mask'"

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    sequences = []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        seq = ids[mask.bool()].tolist()
        sequences.append(seq)
    return sequences


def _pack_input_ids(
    trie: TrieNode,
    input_template: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    # input_ids = torch.zeros(
    #     (max_tokens,),
    #     dtype=input_template.dtype,
    #     device=input_template.device,
    # )
    input_ids = torch.zeros((max_tokens,), dtype=input_template.dtype, device=input_template.device)

    for node in trie.nodes:
        seq_id = node.sequence_ids[0]
        seq_pos = sum(ancestor.num_tokens for ancestor in node.ancestors)
        tree_start, tree_end = node.tree_indices
        input_ids[tree_start : tree_end + 1] = input_template[seq_id][seq_pos : seq_pos + node.num_tokens]

    return input_ids.unsqueeze(0)

def _pack_position_ids(trie: TrieNode, position_template: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if position_template.dim() == 2:
        packed = torch.zeros((1, max_tokens), dtype=position_template.dtype, device=position_template.device)
        for node in trie.nodes:
            seq_id = node.sequence_ids[0]
            seq_pos = sum(ancestor.num_tokens for ancestor in node.ancestors)
            tree_start, tree_end = node.tree_indices
            packed[0, tree_start : tree_end + 1] = position_template[seq_id, seq_pos : seq_pos + node.num_tokens]
        return packed
    if position_template.dim() == 3:
        packed = torch.zeros((position_template.shape[1], 1, max_tokens), dtype=position_template.dtype, device=position_template.device)
        for node in trie.nodes:
            seq_id = node.sequence_ids[0]
            seq_pos = sum(ancestor.num_tokens for ancestor in node.ancestors)
            tree_start, tree_end = node.tree_indices
            packed[:, 0, tree_start : tree_end + 1] = position_template[seq_id, :, seq_pos : seq_pos + node.num_tokens]
        return packed
    raise ValueError(f"Unsupported packed position_ids shape: {tuple(position_template.shape)}")


_ATTN_MASK_BLOCK_SIZE = 2048


def _build_attention_mask(
    trie: TrieNode,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((max_tokens, max_tokens), dtype=torch.bool, device=device)

    for seq_id in trie.all_sequence_ids:
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            continue

        position_chunks = [torch.arange(start, end + 1, device=device) for start, end in indices if end >= start]
        if not position_chunks:
            continue

        positions = torch.cat(position_chunks, dim=0)
        seq_len = positions.numel()
        if seq_len == 0:
            continue

        _apply_causal_mask_blockwise(mask, positions, seq_len, device)

    return mask


def _apply_causal_mask_blockwise(
    mask: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> None:
    block_size = _ATTN_MASK_BLOCK_SIZE
    num_blocks = (seq_len + block_size - 1) // block_size

    for block_row in range(num_blocks):
        row_start = block_row * block_size
        row_end = min((block_row + 1) * block_size, seq_len)
        row_len = row_end - row_start

        tril = torch.tril_indices(row_len, row_len, device=device, dtype=torch.int32)
        local_rows, local_cols = tril
        global_rows = row_start + local_rows
        global_cols = row_start + local_cols
        mask[positions[global_rows], positions[global_cols]] = True
        del tril, local_rows, local_cols, global_rows, global_cols

        for block_col in range(block_row):
            col_start = block_col * block_size
            col_end = min((block_col + 1) * block_size, seq_len)

            block_rows = torch.arange(row_start, row_end, device=device, dtype=torch.int32)
            block_cols = torch.arange(col_start, col_end, device=device, dtype=torch.int32)

            row_positions = positions[block_rows].unsqueeze(1)
            col_positions = positions[block_cols].unsqueeze(0)
            mask[row_positions, col_positions] = True
            del block_rows, block_cols, row_positions, col_positions


def _pack_extra_data(
    trie: TrieNode,
    data: dict[str, Any],
    sequence_lens: torch.Tensor,
    packable_keys: set[str],
    non_packable_keys: set[str],
) -> dict[str, Any]:
    extra_data: dict[str, Any] = {}
    seq_ids = trie.all_sequence_ids
    lens = [sequence_lens[sid].item() for sid in seq_ids]

    for key in packable_keys:
        value = data[key]
        packed = torch.empty(
            (sum(lens), *value.shape[2:]),
            dtype=value.dtype,
            device=value.device,
        )
        cursor = 0
        for length, seq_id in zip(lens, seq_ids, strict=True):
            packed[cursor : cursor + length] = value[seq_id][:length]
            cursor += length
        extra_data[key] = packed

    for key in non_packable_keys:
        extra_data[key] = data[key]

    return extra_data


def get_packed_tree_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    input_ids = input_ids.squeeze()
    if input_ids.ndim != 1:
        raise ValueError("Packed tree 'input_ids' must be a 1D tensor after squeezing.")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
        raise ValueError("Packed tree attention_mask must be a square matrix.")
    if attention_mask.shape[0] != input_ids.shape[0]:
        raise ValueError("Packed tree attention_mask must align with input_ids length.")

    if attention_mask.shape[0] == 0:
        position_ids = torch.empty(0, dtype=torch.long, device=attention_mask.device)
    else:
        ancestor_counts = attention_mask.bool().sum(dim=-1, dtype=torch.long)
        position_ids = torch.clamp_min(ancestor_counts - 1, 0)

    return position_ids.unsqueeze(0)


def build_block_mask_from_trie(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
) -> BlockMask:
    from verl.models.tree_attn.module_fsdp import create_block_mask_from_dense

    if not trie.all_sequence_ids:
        dummy_mask = torch.zeros((padded_size, padded_size), dtype=torch.bool, device=device)
        return create_block_mask_from_dense(dummy_mask, padded_size, device)

    attention_mask = _build_attention_mask(trie, padded_size, device)
    block_mask = create_block_mask_from_dense(attention_mask, padded_size, device)
    del attention_mask

    return block_mask


def build_attention_mask_from_trie(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not trie.all_sequence_ids:
        return torch.zeros((padded_size, padded_size), dtype=torch.bool, device=device)

    attention_mask = _build_attention_mask(trie, padded_size, device)
    return attention_mask


def build_tree_attn_kwargs(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
) -> dict[str, Any]:
    return {"tree_block_mask": build_block_mask_from_trie(trie, padded_size, device)}


@dataclass
class TreeMicroBatchItem:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    trie_node: TrieNode
    padded_to_length: int
    padding_length: int
    extra_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeMicroBatchList:
    items: list[TreeMicroBatchItem]
    original_data: dict[str, Any]
    group_lens: list[int]
    padded_to_lengths: list[int]
    padding_lengths: list[int]
    tree_token_ratio: float = 1.0

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def build_packed_tree_batch(
    data: dict[str, Any],
    max_tokens_per_tree: int,
    pad_to_maximum: bool = True,
    dp_group: dist.ProcessGroup | None = None,
    parallel_size: int = 1,
) -> TreeMicroBatchList:
    if max_tokens_per_tree is None or max_tokens_per_tree <= 0:
        raise ValueError("max_tokens_per_tree must be a positive value for tree training.")

    if not pad_to_maximum:
        raise ValueError(
            "No padding is not supported for tree training. "
            "Block masks require padded sequences for efficient computation. "
            "Please set pad_to_maximum=True."
        )

    block_align = math.lcm(BLOCK_SIZE, parallel_size)
    if pad_to_maximum and max_tokens_per_tree % block_align != 0:
        raise ValueError(
            f"max_tokens_per_tree ({max_tokens_per_tree}) must be a multiple of "
            f"{block_align} (lcm of BLOCK_SIZE={BLOCK_SIZE} and "
            f"parallel_size={parallel_size}) when pad_to_maximum=True."
        )

    tries, num_tokens_list = _greedy_build_tries(data, max_tokens_per_tree)

    if dist.is_initialized():
        num_trees = len(tries)
        input_template: torch.Tensor = data["input_ids"]

        local_count = torch.tensor([num_trees], dtype=torch.int64, device=input_template.device)
        world_size = dist.get_world_size(dp_group)
        all_counts = [torch.zeros(1, dtype=torch.int64, device=input_template.device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count, group=dp_group)

        max_num_trees = max(c.item() for c in all_counts)

        if num_trees < max_num_trees:
            num_dummy_trees = max_num_trees - num_trees
            for _ in range(num_dummy_trees):
                dummy_tree_id = len(tries)
                dummy_trie = TrieNode(tree_id=dummy_tree_id)
                tries.append(dummy_trie)
                num_tokens_list.append(0)

    input_template: torch.Tensor = data["input_ids"]
    mask_template: torch.Tensor = data["attention_mask"]
    position_template: torch.Tensor | None = data.get("position_ids")

    original_num_tokens = mask_template.sum()
    total_tree_tokens = sum(num_tokens_list)
    ratio = total_tree_tokens / original_num_tokens if original_num_tokens > 0 else 1.0

    sequence_lens = mask_template.sum(dim=1, dtype=torch.int32)

    packable_keys = {
        key
        for key, value in data.items()
        if key not in {"input_ids", "attention_mask", "position_ids"}
        and torch.is_tensor(value)
        and value.shape == input_template.shape
    }

    # non_packable_keys = set(data.keys()) - packable_keys - {"input_ids", "attention_mask"}
    non_packable_keys = set(data.keys()) - packable_keys - {"input_ids", "attention_mask", "position_ids"}

    items: list[TreeMicroBatchItem] = []
    padding_lengths: list[int] = []
    padded_to_lengths: list[int] = []

    for trie, num_tokens in zip(tries, num_tokens_list, strict=True):
        padded_size = max_tokens_per_tree if pad_to_maximum else num_tokens

        input_ids = _pack_input_ids(trie, input_template, padded_size)

        # attention_mask = _build_attention_mask(trie, padded_size, mask_template.device)

        # position_ids = get_packed_tree_position_ids(input_ids, attention_mask)

        # del attention_mask

        if position_template is not None:
            position_ids = _pack_position_ids(trie, position_template, padded_size)
        else:
            attention_mask = _build_attention_mask(trie, padded_size, mask_template.device)
            position_ids = get_packed_tree_position_ids(input_ids, attention_mask)
            del attention_mask
        extra_data = _pack_extra_data(trie, data, sequence_lens, packable_keys, non_packable_keys)

        item = TreeMicroBatchItem(
            input_ids=input_ids,
            position_ids=position_ids,
            trie_node=trie,
            padded_to_length=padded_size,
            padding_length=padded_size - num_tokens,
            extra_data=extra_data,
        )
        items.append(item)
        padding_lengths.append(padded_size - num_tokens)
        padded_to_lengths.append(padded_size)

    batch = TreeMicroBatchList(
        items=items,
        original_data=data,
        group_lens=num_tokens_list,
        padded_to_lengths=padded_to_lengths,
        padding_lengths=padding_lengths,
        tree_token_ratio=float(ratio),
    )
    return batch
