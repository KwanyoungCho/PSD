import torch
from ssd.config import Config
from transformers import AutoTokenizer

@torch.inference_mode()
def compute_megaspec_lookahead(MQ_LEN: int, K: int) -> int:
    return K + 1 + K * MQ_LEN 

@torch.inference_mode()
def make_glue_decode_input_ids(
    draft_tokens: torch.Tensor,  # [B, K]
    rec_tokens: torch.Tensor,   # [B]
) -> torch.Tensor:
    """
    Creates glue_token_input_ids of shape [B, K+1] with recovery token first.
    """
    assert draft_tokens.shape[0] == rec_tokens.shape[0], f"Expected draft_tokens and rec_tokens to have the same number of rows, got {draft_tokens.shape[0]} and {rec_tokens.shape[0]}"
    
    # we need flat [num_ttl_q_tokens, num_q_heads, head_dim] for multi-query decode inputs 
    # all info about batch size etc goes through wrapper.plan()
    
    out = torch.cat([rec_tokens.unsqueeze(1), draft_tokens], dim=1).view(-1) # [B, K+1] -> B(K+1)
    
    return out 

def get_forked_recovery_tokens_from_logits(config: Config, logits: torch.Tensor, cache_hits: torch.Tensor, returned_tokens: torch.Tensor, tokenizer: AutoTokenizer, mesa_proxy=None):
    """
    logits: Float[Tensor] of shape [B, K+1, V]
    fan_out_list: list[int] of length K+1 with per-position topk, or int to use for all positions
    mesa_proxy: optional dict with accept_probs [B,K], topk_ids [B,K,top_k], topk_probs [B,K,top_k]

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

    # MESA: proxy-sourced token swap at positions 0..K-1
    if mesa_proxy is not None:
        proxy_topk_ids = mesa_proxy["topk_ids"]  # [B, K, proxy_top_k]

        # Fallback candidates for refill
        total_need = max(k_max, proxy_topk_ids.shape[-1] + k_max)
        _, fallback_topk = torch.topk(logits, min(total_need, V_actual), dim=-1)

        for b in range(B):
            for pos in range(K):
                draft_set = set(topk_idx[b, pos].tolist())
                proxy_tokens = proxy_topk_ids[b, pos].tolist()

                # Proxy tokens not in draft
                selected = [t for t in proxy_tokens if t not in draft_set]

                # Fallback: tokens not in draft or selected proxy
                if len(selected) < k_max:
                    used = draft_set | set(selected)
                    fallback = [t for t in fallback_topk[b, pos].tolist() if t not in used]
                    selected.extend(fallback[:k_max - len(selected)])

                # Rewrite full row: proxy-sourced first, then draft refill
                for j in range(min(len(selected), k_max)):
                    topk_idx[b, pos, j] = selected[j]
        # Position K (all-accept): keep draft top-k unchanged

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
