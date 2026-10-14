"""Cache-root scores with request-matched temperatures and terminal mixture.

Scores select cache roots, never actual correction tokens or proposal q.
Startup Config options propagate to eager, dense and flat CUDA-graph paths.
"""
import torch
from ssd.engine.helpers.root_policy_config import options


def probabilities(logits, temperatures=1.0, sampler_x=None, fan_out=1):
    """Batched [B,contexts,V]; T=0 retains soft scores for budget ranking."""
    from ssd.engine.helpers.p2_tree import q_probs_from_logits
    b, n, v = logits.shape
    # A Python scalar must not create a pageable CPU->CUDA copy in capture.
    # Tensor temperatures remain live inputs of serving graphs.
    t = (torch.full((b,), float(temperatures), dtype=torch.float32, device=logits.device)
         if isinstance(temperatures, (int, float)) else
         torch.as_tensor(temperatures, dtype=torch.float32, device=logits.device))
    t = t.reshape(-1).expand(b)
    return q_probs_from_logits(logits.reshape(b*n, v),
        t[:, None].expand(b, n).reshape(-1), sampler_x, fan_out).view(b, n, v)


def token_scores(e, q, source):
    if source == 'residual':
        return (e-q).clamp_min(0)
    if source == 'proxy':
        return e.clone()
    if source == 'complement':
        return e*(1-q).clamp_min(0)
    raise ValueError(f'Unknown cache-root source: {source}')


def vocabulary_mass(scores):
    """Canonical row reduction, independent of excluded-child padding stride.

    Dense exclusion leaves one padding column per request; flat exclusion
    leaves one per context. Summing those strided views directly produced
    one-ULP denominator differences and changed near-tied candidate IDs.
    Only the reduction layout changes, not the normalization formula.
    """
    v = scores.shape[-1]
    return scores.reshape(-1, v).contiguous().sum(-1).reshape(*scores.shape[:-1], 1)


def rank(scores, terminal, top_k, wire_n, normalization, pack_scores=True):
    from ssd.engine.helpers.p2_tree import pack_piv
    values, ids = scores.topk(min(top_k, scores.shape[-1]), -1)
    denom = (vocabulary_mass(scores) if normalization == 'full'
             else vocabulary_mass(values))
    piv = values / denom.clamp_min(1e-10) * terminal[..., None]
    value, index = piv.flatten(1).topk(min(wire_n, piv[0].numel()), 1)
    pos = index // values.shape[-1]
    tok = ids.flatten(1).gather(1, index)
    if pos.shape[1] < wire_n:
        pad = wire_n-pos.shape[1]
        pos = torch.cat([pos, pos.new_zeros(pos.shape[0], pad)], 1)
        tok = torch.cat([tok, tok.new_zeros(tok.shape[0], pad)], 1)
        value = torch.cat([value, value.new_zeros(value.shape[0], pad)], 1)
    return pos, pack_piv(tok, value) if pack_scores else tok, value


def hazard(alpha):
    prod = alpha.cumprod(1)
    reach = torch.cat([alpha.new_ones(alpha.shape[0],1), prod], 1)
    return torch.cat([reach[:, :-1]*(1-alpha), reach[:, -1:]], 1)


def chain_candidates(exit_logits, q_logits, tokens, valid_k, top_k, wire_n,
                     pack_scores=False, target_temps=1.0, draft_temps=1.0,
                     source='complement', normalization='topm', overlap_mix=.25,
                     sampler_x=None, fan_out=1, cache_hits=None):
    b, k, v = q_logits.shape
    e = probabilities(exit_logits, target_temps)
    q = probabilities(q_logits, draft_temps, sampler_x, fan_out)
    gather = tokens[:, :k, None]
    a = (e[:, :k].gather(2, gather).squeeze(2) /
         (q.gather(2, gather).squeeze(2)+1e-10)).clamp(max=1)
    active = torch.arange(k, device=e.device)[None] < valid_k[:, None]
    if cache_hits is not None:
        # Without JIT a miss has no genuine q. Rank only the current exit
        # context and do not use the arbitrary proposal payload.
        active = active & cache_hits[:, None].bool()
    h = hazard(torch.where(active,a,0.))
    if overlap_mix:
        avg = torch.where(active,torch.minimum(e[:, :k], q).sum(-1).clamp(0, 1),0.)
        h = (1-overlap_mix)*h + overlap_mix*hazard(avg)
    score = token_scores(e[:, :k], q, source)
    score.scatter_(2, gather, 0)
    # A ragged row's bonus is at its REAL end, not the batch's maximum K.
    score = torch.where(active[:, :, None], score, e[:, :k])
    score = torch.cat([score, e[:, k:k+1]], 1)
    return rank(score, h, top_k, wire_n, normalization, pack_scores)


def tree_candidates(exit_logits, q_logits, tokens, topology, wire_n, depth,
                    top_k, target_temps=1.0, draft_temps=1.0,
                    source='complement', normalization='topm', overlap_mix=.25,
                    sampler_x=None, fan_out=1):
    from ssd.engine.helpers.batch_tree_sampling import ladder
    e = probabilities(exit_logits, target_temps)
    q = probabilities(q_logits, draft_temps, sampler_x, fan_out)
    _, term, residual = ladder(tokens, e, q, topology, depth,
                               overlap_mix=overlap_mix,need_residual=(source=='residual'))
    b, r, v = e.shape
    if source == 'residual':
        scores = residual
    else:
        # Every sibling stores the same original parent proposal logits.
        q_ext = torch.cat([q, q.new_zeros(b, 1, v)], 1)
        bi = torch.arange(b, device=e.device)[:, None]
        parent_q = q_ext[bi, topology['child'][:, :, 0]]
        scores = token_scores(e, parent_q, source)
        scores = torch.where(topology['child_valid'][:, :, :1], scores, e)
    flat = torch.cat([scores.reshape(b, -1), scores.new_zeros(b, 1)], 1)
    exclude = (topology['par']+1).clamp(0, r-1)*v+tokens
    exclude = torch.where(topology['node_valid'], exclude,
                          torch.full_like(exclude, r*v))
    flat.scatter_(1, exclude, 0)
    return rank(flat[:, :-1].view(b, r, v), term, top_k, wire_n,
                normalization)
