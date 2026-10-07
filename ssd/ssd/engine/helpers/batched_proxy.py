"""Captured batched Policy B, preserving the eager ragged-chain policy."""
import os
import torch


def batched_chain_candidates(exit_logits, q_logits, tokens, valid_k,
                             top_k, wire_n, pack_scores=False):
    b, k, _ = q_logits.shape
    e = exit_logits[:, :k].float().softmax(-1)
    q = q_logits.float().softmax(-1)
    gather = tokens[:, :k, None]
    accept = (e.gather(2, gather).squeeze(2) /
              (q.gather(2, gather).squeeze(2) + 1e-10)).clamp(max=1.)
    residual = (e - q).clamp(min=0.)
    residual.scatter_(2, gather, 0.)
    prob, ids = residual.topk(top_k, dim=-1)
    prob = prob / prob.sum(-1, keepdim=True).clamp(min=1e-10)
    if os.environ.get("SSD_MIXED_MISS_AR", "0") == "1":
        # Zero proposals means an ordinary target draw, not residual
        # correction. Do not exclude the arbitrary padded draft token.
        ep, ei = e[:, 0].topk(top_k, dim=-1)
        ep = ep / ep.sum(-1, keepdim=True).clamp_min(1e-10)
        empty = (valid_k == 0)[:, None]
        prob[:, 0] = torch.where(empty, ep, prob[:, 0])
        ids[:, 0] = torch.where(empty, ei, ids[:, 0])
    padding = torch.arange(k, device=e.device)[None] >= valid_k[:, None]
    accept = accept.masked_fill(padding, 0.)
    prod = accept.cumprod(1)
    h = torch.zeros(b, k+1, dtype=accept.dtype, device=e.device)
    h[:, 0] = 1 - accept[:, 0]
    if k > 1:
        h[:, 1:k] = prod[:, :-1] * (1 - accept[:, 1:])
    h[:, k] = prod[:, -1]
    last_prob, last_ids = exit_logits[:, k].float().softmax(-1).topk(top_k, dim=-1)
    last_prob = last_prob / last_prob.sum(-1, keepdim=True).clamp(min=1e-10)
    prob = torch.cat((prob, last_prob[:, None]), dim=1)
    ids = torch.cat((ids, last_ids[:, None]), dim=1)
    piv = h[:, :, None] * prob
    values, indexes = piv.flatten(1).topk(wire_n, dim=-1)
    chosen_pos = indexes // top_k
    chosen_tok = ids.flatten(1).gather(1, indexes)
    if pack_scores:
        from ssd.engine.helpers.p2_tree import pack_piv
        chosen_tok = pack_piv(chosen_tok, values)
    return chosen_pos, chosen_tok, values


class BatchedChainProxyCUDAGraph:
    @torch.inference_mode()
    def __init__(self, batch_size, k, vocab_size, top_k, wire_n,
                 pack_scores, dtype, device):
        self.batch_size, self.k = int(batch_size), int(k)
        self.in_exit = torch.zeros(batch_size, k+1, vocab_size, dtype=dtype, device=device)
        self.in_q = torch.zeros(batch_size, k, vocab_size, dtype=dtype, device=device)
        self.in_tokens = torch.zeros(batch_size, k, dtype=torch.long, device=device)
        self.in_valid_k = torch.full((batch_size,), k, dtype=torch.long, device=device)
        args = (self.in_exit, self.in_q, self.in_tokens, self.in_valid_k,
                top_k, wire_n, pack_scores)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                batched_chain_candidates(*args)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = batched_chain_candidates(*args)

    @torch.inference_mode()
    def replay(self, exit_logits, q_logits, tokens, valid_k=None):
        b = int(exit_logits.shape[0])
        if not 0 < b <= self.batch_size or exit_logits.shape[1:] != self.in_exit.shape[1:]:
            raise ValueError("Batched proxy input does not match captured shape")
        self.in_exit[:b].copy_(exit_logits)
        self.in_q[:b].copy_(q_logits)
        self.in_tokens[:b].copy_(tokens[:, :self.k])
        if valid_k is None:
            self.in_valid_k[:b].fill_(self.k)
        else:
            self.in_valid_k[:b].copy_(valid_k)
        # Padded batch rows are independent; their outputs are never sent.
        self.graph.replay()
        return tuple(x[:b] for x in self.out)
