import os

BLOCK_SIZE = int(os.environ.get("VERL_FLEX_ATTENTION_BLOCK_SIZE", "128"))
USE_TRITON_TREE_ATTN = int(os.environ.get("VERL_USE_TRITON_TREE_ATTN", "0")) == 1
