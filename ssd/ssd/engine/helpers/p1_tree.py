"""Phase-1 dynamic tree preparation.

P1 and P2 share the same captured rollout after their roots exist.  P2 roots
arrive with an early-exit proxy score; P1 instead creates a fixed number of
uniform candidates at every glue context.  Its initial score is the draft
probability of reaching that context times the draft probability of the
alternative root token.  Round zero evaluates every root; later rounds use
the same global cumulative-confidence expansion as P2.  The only phase
difference is the source of the root prior: P2 receives a target proxy score,
whereas P1 derives it from draft context reach and root-token probability.

This module contains only fixed-shape/GPU-friendly preparation and shape
selection.  Cache serving and target verification are phase-agnostic and live
in the existing DUET runtime.
"""
from __future__ import annotations

import torch

from ssd.engine.helpers.p2_tree import q_probs_from_logits
from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
from ssd.utils.async_helpers.async_spec_helpers import (
    compute_tree_forward_width)


def tree_node_buckets(max_nodes: int, first: int = 4) -> tuple[int, ...]:
    """Small fixed set of response buckets, always including ``max_nodes``."""
    if max_nodes < 1:
        raise ValueError(f"max_nodes must be positive; got {max_nodes}")
    first = min(max(1, int(first)), int(max_nodes))
    return tuple(sorted(set(range(first, max_nodes + 1, 2)) | {max_nodes}))


def p1_context_buckets(k1: int, k2: int,
                       p1_max_nodes: int,
                       p2_max_nodes: int) -> tuple[int, ...]:
    """Reachable glue-context buckets for P1.

    Use only two coarse canvases in the champion shape: one covers both chain
    widths and every P2 tree, and the optional larger one covers a P1 tree.
    Capturing one executor for every possible valid-node count would multiply
    full-model CUDA graphs and their FlashInfer workspace for no semantic
    gain; inactive roots already have a safe zero-score representation.
    """
    common = max(int(k1) + 1, int(k2) + 1, int(p2_max_nodes) + 1)
    largest = max(common, int(p1_max_nodes) + 1)
    return tuple(sorted({common, largest}))


def choose_p1_context_bucket(actual_contexts: int,
                             buckets: tuple[int, ...]) -> int:
    """Return the smallest pre-captured context bucket that fits a request."""
    if actual_contexts < 1:
        raise ValueError(
            f"P1 requires at least one glue context; got {actual_contexts}")
    for bucket in buckets:
        if bucket >= actual_contexts:
            return int(bucket)
    raise ValueError(
        f"P1 contexts {actual_contexts} exceed captured maximum "
        f"{max(buckets) if buckets else 0}")


@torch.inference_mode()
def build_uniform_p1_roots(
    glue_logits: torch.Tensor,
    returned_tokens: torch.Tensor,
    roots_per_position: int,
    temperatures: torch.Tensor,
    *,
    sampler_x: float | None,
    async_fan_out: int,
    root_width: int | None = None,
    context_glue_rows: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """Build uniform P1 roots and their initial confidence on device.

    Args:
        glue_logits: ``[P,V]`` draft logits for the recovery/tree contexts.
        returned_tokens: ``[P]`` tokens already present in those contexts.
        roots_per_position: fixed P1 candidate count (pfo) for every context.
        temperatures: scalar or ``[P]`` draft sampling temperature.
        root_width: optional captured width.  Real roots occupy the prefix;
            the tail is finite zero-score padding and never produces a view.
        context_glue_rows: optional ``[P,G]`` visibility rows.  Chain inputs
            default to a lower-triangular visibility matrix.

    Returns fixed-address-friendly tensors ``tokens/scores/context_ids/valid``
    of length ``root_width`` plus root-specific glue rows.
    """
    if glue_logits.ndim != 2:
        raise ValueError(
            f"glue_logits must be [P,V]; got {tuple(glue_logits.shape)}")
    p, _ = glue_logits.shape
    if returned_tokens.shape != (p,):
        raise ValueError(
            f"returned_tokens must be [{p}]; got "
            f"{tuple(returned_tokens.shape)}")
    if roots_per_position < 1:
        raise ValueError(
            f"roots_per_position must be positive; got {roots_per_position}")
    real_roots = p * int(roots_per_position)
    width = real_roots if root_width is None else int(root_width)
    if width < real_roots:
        raise ValueError(
            f"root_width {width} cannot hold {real_roots} uniform roots")

    masked = glue_logits.clone()
    # Same exclusion contract as the chain P1 selector: at context j, the
    # already returned next token lives at returned_tokens[j+1].  The final
    # context has no next returned token to exclude.
    if p > 1:
        masked[:-1].scatter_(
            1, returned_tokens[1:].view(-1, 1), float("-inf"))
    root_tok = masked.topk(roots_per_position, dim=-1).indices

    temps = temperatures.to(device=glue_logits.device, dtype=torch.float32)
    if temps.numel() == 1:
        temps = temps.expand(p)
    elif temps.shape != (p,):
        raise ValueError(
            f"temperatures must be scalar or [{p}]; got {tuple(temps.shape)}")
    # Candidate exclusion is a coverage rule, not a probability
    # renormalization.  Score the selected alternative under the original
    # draft distribution; otherwise a context whose returned token owns
    # almost all mass would make a tiny alternative look spuriously certain.
    probs = q_probs_from_logits(
        glue_logits.float(), temps, sampler_x, async_fan_out)
    # Approximate the probability that verification reaches each context.
    # context_glue_rows[c] contains context c plus all of its ancestors.  The
    # direct parent is therefore the largest visible earlier context index.
    # Multiplying the edge probabilities on that row yields the same reach
    # factor for chains and for tree-shaped glue without a host traversal.
    if context_glue_rows is None:
        reach_rows = torch.tril(torch.ones(
            p, p, dtype=torch.uint8, device=glue_logits.device))
    else:
        if context_glue_rows.ndim != 2 or context_glue_rows.shape[0] != p \
                or context_glue_rows.shape[1] < p:
            raise ValueError(
                "context_glue_rows must be [P,G] with G>=P; got "
                f"{tuple(context_glue_rows.shape)}")
        reach_rows = context_glue_rows[:, :p].to(
            device=glue_logits.device, dtype=torch.uint8)
    idx = torch.arange(p, device=glue_logits.device, dtype=torch.int64)
    prior = idx.unsqueeze(0) < idx.unsqueeze(1)
    parent = torch.where(
        reach_rows.bool() & prior, idx.unsqueeze(0),
        torch.zeros((), dtype=torch.int64, device=glue_logits.device)
    ).amax(dim=1)
    edge = probs[parent, returned_tokens]
    edge = edge.clone()
    edge[0] = 1.0
    context_reach = torch.exp(
        reach_rows.to(torch.float32).matmul(edge.clamp_min(1e-30).log()))
    root_score = (
        probs.gather(1, root_tok)
        * context_reach.unsqueeze(1)).reshape(-1)
    root_tok = root_tok.reshape(-1)
    ctx = torch.arange(p, device=glue_logits.device, dtype=torch.int64) \
        .repeat_interleave(roots_per_position)

    if context_glue_rows is None:
        context_glue_rows = reach_rows
    else:
        context_glue_rows = context_glue_rows.to(
            device=glue_logits.device, dtype=torch.uint8)
    root_glue = context_glue_rows.index_select(0, ctx)

    out_tok = torch.zeros(width, dtype=torch.int64, device=glue_logits.device)
    out_score = torch.zeros(
        width, dtype=torch.float32, device=glue_logits.device)
    out_ctx = torch.zeros(width, dtype=torch.int64, device=glue_logits.device)
    out_valid = torch.zeros(width, dtype=torch.bool, device=glue_logits.device)
    out_glue = torch.zeros(
        width, root_glue.shape[1], dtype=torch.uint8,
        device=glue_logits.device)
    out_tok[:real_roots] = root_tok
    out_score[:real_roots] = root_score
    out_ctx[:real_roots] = ctx
    out_valid[:real_roots] = True
    out_glue[:real_roots] = root_glue
    return {
        "tokens": out_tok,
        "scores": out_score,
        "context_ids": out_ctx,
        "valid": out_valid,
        "glue_rows": out_glue,
        "context_reach": context_reach,
        "real_roots": real_roots,
        "width": width,
    }


class P1TreeExecutor(P2TreeExecutor):
    """Full-graph dynamic tree executor specialized to one P1 root bucket."""

    def __init__(self, model, compute_logits_fn, config, device,
                 block_size, max_blocks, vocab_size,
                 num_heads, num_kv_heads, head_dim,
                 *, context_bucket: int, dtype=torch.float16):
        pfo = int(config.duet_p1_roots_per_position)
        root_count = int(context_bucket) * pfo
        scale = float(config.duet_p1_tree_forward_scale)
        width = compute_tree_forward_width(root_count, scale)
        super().__init__(
            model, compute_logits_fn, config, device,
            block_size, max_blocks, vocab_size,
            num_heads, num_kv_heads, head_dim, dtype=dtype,
            phase="p1", width=width, root_count=root_count,
            depth=int(config.duet_phase1_k),
            max_nodes=int(config.duet_p1_tree_max_nodes),
            glue_width=int(context_bucket))
        self.context_bucket = int(context_bucket)
        self.roots_per_position = pfo
        self.forward_scale = scale
        # P1 and P2 intentionally share one expansion algorithm.  P1's
        # in_root_piv contains context-reach * root-token probability, which
        # occupies the same ranking role as P2's target proxy prior.
        self.policy = "dynamic"
