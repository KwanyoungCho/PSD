from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    is_jit: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # DUET: layout-aware attention wrapper selection
    active_mq_len: int | None = None
    active_wrappers: dict | None = None
    active_layout: object | None = None
    # Step 9B-0: glue bucket dispatch (long-hit=K_long, short-hit=K_short).
    # Read by model_runner to pick the right glue CG.
    glue_valid_k: int | None = None
    # Batch 1b: pre-sync'd Python list of cache_hits (per-seq int 0/1).
    # Set by _decode_tree_step so run_fi_tree_decode_cudagraph avoids a
    # duplicate tolist() sync at step 0 (the value is already known from
    # _glue_decode's earlier tolist).
    active_cache_hits_list: list | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_jit=False, active_mq_len=None, active_wrappers=None, active_layout=None, glue_valid_k=None, active_cache_hits_list=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, is_jit, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, active_mq_len, active_wrappers, active_layout, glue_valid_k, active_cache_hits_list)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
