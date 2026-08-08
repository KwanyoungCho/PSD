import math

import torch
from ssd.config import Config
from transformers import AutoTokenizer


def compute_tree_forward_width(root_count: int, scale: float) -> int:
    """Return the fixed forward canvas used by a dynamic tree executor.

    Keep this calculation shared by the executor and scheduler.  If the
    scheduler reserves only ``root_count`` cells while the executor replays a
    wider canvas, page-boundary requests can expose unallocated KV slots.
    """
    if root_count < 1:
        raise ValueError(f"root_count must be positive; got {root_count}")
    if scale < 1.0:
        raise ValueError(f"scale must be at least one; got {scale}")
    return max(int(root_count), math.ceil(int(root_count) * float(scale)))

@torch.inference_mode()
def compute_megaspec_lookahead(
    MQ_LEN: int,
    K: int,
    *,
    split_k1k2: bool = False,
    K1: int = 0,
    K2: int = 0,
    mq_p1: int = 0,
    mq_p2: int = 0,
    glue_width: int = 0,
    cells_p1: int = 0,
    cells_p2: int = 0,
) -> int:
    """Per-step draft KV slot reservation for the scheduler.

    Non-split path (hybrid DUET, legacy two-pass DUET, or non-DUET async SD):
        glue (K+1) + K sequential forwards each writing MQ_LEN disjoint slots.
        Total = K + 1 + K * MQ_LEN.

    Split-K1/K2 path (SSD_FORCE_SPLIT_K1K2=1, DUET only):
        Phase 1 writes ``cells_p1`` compact cells (or K1*mq_p1 for a fixed
        width). Phase 2 likewise writes ``cells_p2`` (or K2*mq_p2) starting
        at the SAME base, overlapping Phase 1's region. This is safe
        because Phase 1's outputs (tokens/logits) are already extracted into
        result tensors before Phase 2 begins, and Phase 2's custom attention
        mask only reads its own slice. Required reservation is therefore
        max(chain glue, tree glue) + max(K1*mq_p1, K2*mq_p2), not
        K_long * MQ_LEN_full.  ``glue_width`` is zero for the legacy chain
        path and the common P1/P2 response width when either dynamic tree is
        enabled.

        This matters because the old K_long * MQ_LEN_full formula over-reserves
        by a factor of ~5× for typical (dfo, pfo, K1, K2) — pushing the
        scheduler into preemption (and the formerly-crashing B=0 cascade) at
        async_fan_out ≥ 5 even though physical scratch fits comfortably.
    """
    if split_k1k2:
        # K2 ≤ K1 invariant (docs/duet/04-split-k1k2-design.md), so K_step = K1.
        # Phase 1/2 default to fixed-width footprints.  Dynamic P1 can pass
        # the exact root_width + continuation_width*(K1-1) compact footprint.
        # Either pass can dominate depending on (dfo, pfo) — pfo > dfo cases
        # (e.g., dfo=1, pfo=3) push K2*mq_p2 above K1*mq_p1 even at K2 ≤ K1.
        # Reserve the worst-case footprint of the two.
        p1_footprint = int(cells_p1) if cells_p1 else K1 * mq_p1
        p2_footprint = int(cells_p2) if cells_p2 else K2 * mq_p2
        return max(K1 + 1, int(glue_width)) + max(
            p1_footprint, p2_footprint)
    return K + 1 + K * MQ_LEN

@torch.inference_mode()
def make_glue_decode_input_ids(
    draft_tokens: torch.Tensor,  # [B, K_max]
    rec_tokens: torch.Tensor,   # [B]
    valid_k: int | None = None,
) -> torch.Tensor:
    """
    Creates glue_token_input_ids of shape [B, valid_k+1] with recovery token first.

    Step 9B-0: ``valid_k`` plumbing for DUET short-hit bucket. When None
    (legacy path), uses draft_tokens.shape[1] as the glue width. When set,
    only the first ``valid_k`` columns of draft_tokens are used (the rest
    is zero-padding for cache row uniformity).
    """
    assert draft_tokens.shape[0] == rec_tokens.shape[0], f"Expected draft_tokens and rec_tokens to have the same number of rows, got {draft_tokens.shape[0]} and {rec_tokens.shape[0]}"

    if valid_k is not None:
        # Slice to the batch dispatch width. M2 (docs/duet/13 §1): the
        # scalar is vk_max over the batch — rows with vk_i < vk_max carry
        # filler in columns (vk_i, vk_max] (cache-padding zeros / JIT-short
        # random init); acceptable for glue input (see
        # hit_cache_and_respond: fork slicing + verify clamp make those
        # positions unreachable).
        draft_tokens = draft_tokens[:, :valid_k]
    out = torch.cat([rec_tokens.unsqueeze(1), draft_tokens], dim=1).view(-1)
    return out

def get_forked_recovery_tokens_from_logits(config: Config, logits: torch.Tensor, cache_hits: torch.Tensor, returned_tokens: torch.Tensor, tokenizer: AutoTokenizer):
    """
    logits: Float[Tensor] of shape [B, K+1, V]
    fan_out_list: list[int] of length K+1 with per-position topk, or int to use for all positions

    Returns:
        idxs: [B, sum(fan_out_list)]
    """
    B, _, V_actual = logits.shape
    K = config.speculate_k
    fan_out_list = config.fan_out_list
    fan_out_list_miss = config.fan_out_list_miss
    assert cache_hits.shape == (B,), f"cache_hits must have shape (B,), got {cache_hits.shape}"
    assert logits.shape[0] == B and logits.shape[1] == K+1, f"logits must have shape (B, K+1, V), got {logits.shape}"
    assert len(fan_out_list) == K + 1, f"fan_out_list must have length K+1={K+1}, got {len(fan_out_list)}"
    assert returned_tokens.shape == (B, K+1), f"returned_tokens must have shape (B, K+1), got {returned_tokens.shape}"

    # Use scatter_ to set returned tokens to -inf
    logits = logits.clone()
    logits[:, :-1, :] = logits[:, :-1, :].scatter(
        dim=2,
        index=returned_tokens[:, 1:].unsqueeze(2),
        value=float('-inf'),
    )

    # Compute top-k at max fanout
    k_max = max(max(fan_out_list), max(fan_out_list_miss))
    _, topk_idx = torch.topk(logits, k_max, dim=-1)  # [B, K+1, k_max]

    # Build per-b, per-(K+1) counts depending on cache_hits
    hit_counts = torch.as_tensor(
        fan_out_list, device=logits.device, dtype=torch.int64)
    miss_counts = torch.as_tensor(
        fan_out_list_miss, device=logits.device, dtype=torch.int64)
    ch_bool = cache_hits.to(torch.bool).view(B, 1)
    counts_b = torch.where(ch_bool, hit_counts.view(1, -1).expand(B, -1),
                           miss_counts.view(1, -1).expand(B, -1))

    ar = torch.arange(k_max, device=logits.device)
    mask = ar.view(1, 1, -1) < counts_b.view(B, K + 1, 1)

    idxs_flat = topk_idx.masked_select(mask).view(B, -1)
    assert idxs_flat.shape == (B, sum(fan_out_list)), f"idxs_flat should be (B, MQ_LEN), got {idxs_flat.shape}"

    return idxs_flat

 
def apply_sampler_x_rescaling(probs: torch.Tensor, sampler_x: float, F: int) -> torch.Tensor:
    """Apply sampler_x rescaling to probabilities.
    
    Args:
        probs: Probability tensor of shape [B, S, V] where S can be =1
        sampler_x: Rescaling factor for top-F probabilities
        F: Number of top probabilities to rescale
        
    Returns:
        Rescaled and renormalized probabilities
    """
    # Find topF indices with highest probs
    _, topk_indices = torch.topk(probs, F+1, dim=-1)  # [B, S, F]

    # Create a mask for topF positions
    topf_mask = torch.zeros_like(probs, dtype=torch.bool)
    topf_mask.scatter_(dim=-1, index=topk_indices, value=True)

    # Rescale topF probs by sampler_x factor
    probs = torch.where(topf_mask, probs * sampler_x, probs)

    # Renormalize to get valid distribution
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs
