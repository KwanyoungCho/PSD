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
    # MESA: layout-aware attention wrapper selection
    active_mq_len: int | None = None
    active_wrappers: dict | None = None
    active_layout: object | None = None
    # Step 9B-0: glue bucket dispatch (long-hit=K_long, short-hit=K_short).
    # Read by model_runner to pick the right glue CG.
    glue_valid_k: int | None = None
    # Step 9B-1: hybrid Phase 2 CG dispatch by bucket (graph_key).
    hybrid_graph_key: str | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_jit=False, active_mq_len=None, active_wrappers=None, active_layout=None, glue_valid_k=None, hybrid_graph_key=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, is_jit, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, active_mq_len, active_wrappers, active_layout, glue_valid_k, hybrid_graph_key)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
