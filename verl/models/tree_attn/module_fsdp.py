import logging
import os

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from verl.models.tree_attn.constants import BLOCK_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_FLEX_DYNAMIC = not (os.environ.get("VERL_DISABLE_FLEX_ATTENTION_DYNAMIC", "0") == "1")
_TORCH_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": not _FLEX_DYNAMIC,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}
_flex_attention = torch.compile(
    flex_attention,
    dynamic=_FLEX_DYNAMIC,
    options=_TORCH_COMPILE_OPTIONS,
)


def create_block_mask_from_dense(
    attention_mask: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> BlockMask:
    def arbitrary_mask(
        batch: torch.Tensor,
        head: torch.Tensor,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
    ):
        return attention_mask[q_idx, k_idx]

    block_mask = create_block_mask(
        arbitrary_mask,
        B=1,
        H=1,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        device=device,
        _compile=False,
    )
    return block_mask


def _tree_attn_fwd_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    *args,
    **kwargs,
):
    tree_block_mask = kwargs.get("tree_block_mask", None)
    if tree_block_mask is None or not isinstance(tree_block_mask, BlockMask):
        raise ValueError(
            "_tree_attn_fwd_func requires a pre-created BlockMask in "
            "kwargs['tree_block_mask']. "
            "Use create_block_mask_from_dense() during data preparation."
        )

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    enable_gqa = query.shape[1] != key.shape[1]

    output = _flex_attention(
        query,
        key,
        value,
        block_mask=tree_block_mask,
        score_mod=None,
        scale=softmax_scale,
        enable_gqa=enable_gqa,
    )
    output = output.permute(0, 2, 1, 3).contiguous()
    return output


ORIGINAL_FLASH_ATTENTION_FORWARD = None


def patch_fsdp_for_tree_training(enable: bool = True):
    if not enable:
        return

    global ORIGINAL_FLASH_ATTENTION_FORWARD
    if ORIGINAL_FLASH_ATTENTION_FORWARD is not None:
        logger.warning("FSDP patch for tree training is already applied.")
        return

    from transformers.integrations import flash_attention

    ORIGINAL_FLASH_ATTENTION_FORWARD = flash_attention._flash_attention_forward
    flash_attention._flash_attention_forward = _tree_attn_fwd_func
    logger.info("Patched transformers.integrations.flash_attention._flash_attention_forward with tree implementation.")


def restore_patch_fsdp_for_tree_training():
    global ORIGINAL_FLASH_ATTENTION_FORWARD
    if ORIGINAL_FLASH_ATTENTION_FORWARD is None:
        logger.warning("FSDP patch for tree training was not applied or already restored.")
        return

    from transformers.integrations import flash_attention

    flash_attention._flash_attention_forward = ORIGINAL_FLASH_ATTENTION_FORWARD
    ORIGINAL_FLASH_ATTENTION_FORWARD = None
    logger.info(
        "Restored transformers.integrations.flash_attention._flash_attention_forward to original implementation."
    )
