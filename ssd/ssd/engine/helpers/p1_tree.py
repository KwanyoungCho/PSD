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

import os

import torch

from ssd.engine.helpers.p2_tree import selected_q_probs_from_logits
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

    Keep an exact short-chain bucket in addition to the long-chain/common and
    largest-tree canvases.  In the K1=9/K2=4 champion shape, merging the short
    five-context path into the ten-context bucket made P1 execute W=20 lanes
    for nine rounds although only W=10 roots were live.  That padding is full
    transformer work, not cheap metadata padding, and occurs on roughly half
    of requests.  Three coarse canvases retain bounded capture memory while
    avoiding this dominant hot-path waste.
    """
    short = int(k2) + 1
    common = max(int(k1) + 1, short, int(p2_max_nodes) + 1)
    largest = max(common, int(p1_max_nodes) + 1)
    return tuple(sorted({short, common, largest}))


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
    # Only U root candidates and one parent->returned-token edge per context
    # are consumed.  ``source_rows`` lets all pair probabilities share one
    # scaled-logit/logsumexp pass even for tree-shaped contexts whose parent
    # row is not the immediately preceding row.
    selected_ids = torch.cat(
        (root_tok, returned_tokens.reshape(p, 1)), dim=1)
    source_rows = idx.reshape(p, 1).expand(
        p, roots_per_position + 1).clone()
    source_rows[:, -1] = parent
    selected_probs = selected_q_probs_from_logits(
        glue_logits, temps, selected_ids, sampler_x, async_fan_out,
        source_rows=source_rows)
    root_probs = selected_probs[:, :roots_per_position]
    edge = selected_probs[:, -1]
    edge = edge.clone()
    edge[0] = 1.0
    context_reach = torch.exp(
        reach_rows.to(torch.float32).matmul(edge.clamp_min(1e-30).log()))
    root_score = (
        root_probs
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
                 *, context_bucket: int, dtype=torch.float16,
                 materialize_backbone_logits=True):
        pfo = int(config.duet_p1_roots_per_position)
        root_count = int(context_bucket) * pfo
        scale = float(config.duet_p1_tree_forward_scale)
        # Every P1 cache-key root is evaluated once.  After that first round,
        # only the globally best frontier nodes need model forwards.  The old
        # executor replayed ``root_count`` rows for all K1 rounds, including
        # up to 38 rows after a P1 tree hit, even when most lanes had already
        # been rejected by the score thresholds.  Use the corresponding P1
        # chain fanout sum as the continuation compute budget.  This keeps
        # tree compute close to the established chain rather than silently
        # shrinking it to the usually smaller P2 width.
        fanout = getattr(config, "duet_split_phase1_fan_out_list", None)
        if fanout is None:
            default_fanout = int(getattr(
                config, "duet_draft_fan_out",
                config.duet_proxy_total_budget
                // max(1, int(config.duet_phase1_k) + 1)))
            fanout = [default_fanout] * (
                int(config.duet_phase1_k) + 1)
        else:
            fanout = list(fanout)
        if len(fanout) != int(config.duet_phase1_k) + 1:
            raise ValueError(
                "P1 executor fanout list must have K1+1 entries; got "
                f"K1={config.duet_phase1_k}, fanout={fanout}")
        short_contexts = int(config.duet_phase2_k) + 1
        chain_width = sum(
            fanout[:short_contexts]
            if int(context_bucket) <= short_contexts else fanout)
        if chain_width < 1:
            raise ValueError(
                f"P1 continuation width must be positive; fanout={fanout}")
        # ``scale`` remains an explicit compute experiment multiplier, not a
        # root-coverage knob.  It never forces more lanes than live roots.
        continuation_width = min(
            root_count,
            compute_tree_forward_width(chain_width, scale))
        round_widths = ((root_count,)
                        + (continuation_width,)
                        * max(0, int(config.duet_phase1_k) - 1))
        width = max(round_widths)
        super().__init__(
            model, compute_logits_fn, config, device,
            block_size, max_blocks, vocab_size,
            num_heads, num_kv_heads, head_dim, dtype=dtype,
            phase="p1", width=width, root_count=root_count,
            depth=int(config.duet_phase1_k),
            max_nodes=int(config.duet_p1_tree_max_nodes),
            glue_width=int(context_bucket),
            materialize_backbone_logits=materialize_backbone_logits,
            round_widths=round_widths)
        # Three context buckets remove substantial live-lane padding, but a
        # generic 64 MiB FlashInfer workspace for every one of seven page
        # shapes would exhaust the 24 GiB production draft GPU.  The real
        # TinyLlama/FA2 P1 shape requests about 41 MiB; keep a 48 MiB P1-only
        # margin (the generic/P2 executor retains its conservative 64 MiB).
        # This is set before prepare_bucket() allocates any workspace.
        p1_workspace_mb = int(os.environ.get(
            "SSD_P1_TREE_EXEC_WORKSPACE_MB", "48"))
        if p1_workspace_mb <= 0:
            raise ValueError(
                "SSD_P1_TREE_EXEC_WORKSPACE_MB must be positive")
        self._workspace_bytes = p1_workspace_mb * 2**20
        self.context_bucket = int(context_bucket)
        self.roots_per_position = pfo
        self.forward_scale = scale
        self.continuation_width = continuation_width
        self.chain_forward_width = chain_width
        # P1 and P2 intentionally share one expansion algorithm.  P1's
        # in_root_piv contains context-reach * root-token probability, which
        # occupies the same ranking role as P2's target proxy prior.
        self.policy = "dynamic"
