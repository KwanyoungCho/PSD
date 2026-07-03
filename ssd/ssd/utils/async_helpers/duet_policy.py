"""Candidate-set Policy B — shared by the two raw-proxy consumers.

Given the target's top-M exit candidates per verify position (ids +
exact logits + logsumexp) and the draft distribution lq (logits_q), this
computes the Policy B wire {chosen_pos, chosen_tok} exactly as the
legacy full-vocab verifier path (docs/duet/05), except the residual
candidate set is top-M(p_E) instead of the full vocab — residual <= p_E
pointwise, so top-M(p_E) contains the true residual top-k up to the p_E
mass below rank M (CPU-equivalence: 100% overlap at M>=16).

Consumers:
- DraftRunner._policy_b_from_raw_proxy (SSD_DUET_PROXY_ON_DRAFT=1):
  raw arrives over NCCL, lq = the draft's own response logits.
- Verifier._compute_and_send_proxy (SSD_DUET_EXIT_TOPM_GATHER=1):
  raw comes from the rank-local top-M lm_head gather, lq = logits_q.
"""
import torch


def policy_b_from_candidates(raw, lq, y_tok, K_step, top_k, wire_N):
    """Compute Policy B {chosen_pos, chosen_tok} from top-M candidates.

    Args:
        raw: {"topk_ids" [>=K+1, M] int64, "topk_logits" [>=K+1, M] fp32,
              "lse" [>=K+1] fp32, "y_logit" [>=K] fp32} — rows beyond
              K_step+1 (K_max padding) are ignored.
        lq: [K_alloc, V] draft logits (the logits_q the target verifies
            against); rows [0..K_step) used.
        y_tok: [K_step] draft tokens (speculations[1:K_step+1]).
        K_step: the step's valid_k (verify width - 1).
        top_k / wire_N: config.duet_proxy_top_k / duet_proxy_wire_N.
    """
    ids = raw["topk_ids"][:K_step + 1]                    # [K+1, M]
    pE_top = torch.exp(
        raw["topk_logits"][:K_step + 1]
        - raw["lse"][:K_step + 1].unsqueeze(1))           # [K+1, M]
    pE_y = torch.exp(raw["y_logit"][:K_step] - raw["lse"][:K_step])  # [K]

    # Gather-based p_D: p_D(v) = exp(lq(v) - lse_q) — identical values to
    # a full softmax without materializing [K, V] fp32.
    lq = lq[:K_step, :]
    lse_q = torch.logsumexp(lq.float(), dim=-1)                     # [K]
    y_lq = lq.gather(1, y_tok.unsqueeze(1)).squeeze(1).float()      # [K]
    pD_y = torch.exp(y_lq - lse_q)                                  # [K]
    accept_probs = (pE_y / (pD_y + 1e-10)).clamp(max=1.0)           # [K]

    # h_i (first-reject distribution) — verbatim from the verifier.
    h = torch.zeros(K_step + 1, device=lse_q.device)
    cumprod = torch.cumprod(accept_probs, dim=0)
    h[0] = 1 - accept_probs[0]
    if K_step > 1:
        h[1:K_step] = cumprod[:-1] * (1 - accept_probs[1:])
    h[K_step] = cumprod[-1]

    # Residual [p_E - p_D]_+ on the M-candidate set, draft token excluded.
    at_lq = lq.gather(1, ids[:K_step]).float()            # [K, M]
    pD_at = torch.exp(at_lq - lse_q.unsqueeze(1))         # [K, M]
    resid = (pE_top[:K_step] - pD_at).clamp(min=0)
    resid = resid.masked_fill(ids[:K_step] == y_tok.unsqueeze(1), 0.0)
    r_probs, r_idx = resid.topk(top_k, dim=-1)            # [K, top_k]
    r_probs = r_probs / r_probs.sum(-1, keepdim=True).clamp(min=1e-10)
    r_ids = ids[:K_step].gather(1, r_idx)

    # Position K (all-accept): target's exit distribution directly.
    k_probs, k_idx = pE_top[K_step].topk(top_k)
    k_probs = k_probs / k_probs.sum().clamp(min=1e-10)
    k_ids = ids[K_step].gather(0, k_idx)

    corr_probs = torch.cat([r_probs, k_probs.unsqueeze(0)], dim=0)  # [K+1, top_k]
    corr_ids = torch.cat([r_ids, k_ids.unsqueeze(0)], dim=0)
    P_iv = h.unsqueeze(1) * corr_probs
    _, top_idx = P_iv.flatten().topk(wire_N)
    return {
        "chosen_pos": top_idx // top_k,
        "chosen_tok": corr_ids.view(-1).gather(0, top_idx),
    }
