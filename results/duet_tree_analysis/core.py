"""CPU reference analysis. No production policy or sampler is changed.

Subset optimization is a fixed-realized-tree diagnostic, NOT a lossless
post-sampling pruning policy. See the selection-bias audit before deployment.
"""
from __future__ import annotations
import ast
from collections import defaultdict
from functools import lru_cache
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def production_functions():
    """Load actual pure helper bodies, avoiding engine/CUDA initialization."""
    import torch
    path = ROOT / 'ssd/ssd/engine/helpers/p2_tree.py'
    names = {'rerank_tree_indices', 'validate_tree_ints',
             'tree_policy_b_ladder', 'tree_verify_walk_tensor'}
    tree = ast.parse(path.read_text())
    module = ast.Module(body=[n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = dict(torch=torch, math=math, np=np, defaultdict=defaultdict)
    exec(compile(module, str(path), 'exec'), scope)
    return {name: scope[name] for name in names}


def groups(par, sib):
    kids = defaultdict(list)
    for j, p in enumerate(par):
        if not -1 <= int(p) < j:
            raise ValueError('Invalid parent order')
        kids[int(p)].append(j)
    for p, js in kids.items():
        js.sort(key=lambda j: int(sib[j]))
        if [int(sib[j]) for j in js] != list(range(len(js))):
            raise ValueError('Sibling prefix required')
    return kids


def normalize(x, fallback=None):
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    return x / total if total > 1e-14 else (
        np.zeros_like(x) if fallback is None else np.asarray(fallback).copy())


def exact_ladder(par, sib, tokens, p_rows, q_rows):
    """q_rows[j] is the ORIGINAL parent q, before sibling exclusions.

    Returns conditional alpha, joint node reach, joint attempt probability,
    terminal mass, terminal residual, and expected accepted token count.
    All quantities condition on the already realized, unpruned candidate tree.
    """
    par = list(map(int, par)); sib = list(map(int, sib))
    tokens = np.asarray(tokens, dtype=np.int64)
    p_rows = np.asarray(p_rows, dtype=np.float64)
    q_rows = np.asarray(q_rows, dtype=np.float64)
    n = len(par)
    kids = groups(par, sib)
    alpha = np.zeros(n); reach = np.zeros(n); attempt = np.zeros(n)
    residual = p_rows.copy(); terminal = np.zeros(n + 1)
    for ctx in [-1] + list(range(n)):
        incoming = 1.0 if ctx == -1 else reach[ctx]
        R = p_rows[ctx + 1].copy()
        remaining = incoming
        js = kids.get(ctx, [])
        D = q_rows[js[0]].copy() if js else None
        for j in js:
            t = tokens[j]
            if D[t] <= 0:
                raise ValueError('Zero proposal probability')
            a = min(1.0, R[t] / D[t])
            alpha[j] = a; attempt[j] = remaining
            reach[j] = remaining * a
            remaining *= 1.0 - a
            R = normalize(np.maximum(R - D, 0))
            D[t] = 0; D = normalize(D)
        terminal[ctx + 1] = remaining
        residual[ctx + 1] = normalize(R, p_rows[ctx + 1])
    assert abs(terminal.sum() - 1) < 1e-8
    depth = np.zeros(n, dtype=int)
    for j, p in enumerate(par): depth[j] = 1 if p < 0 else depth[p] + 1
    assert abs(reach.sum() - np.dot(terminal[1:], depth)) < 1e-8
    return dict(alpha=alpha, reach=reach, attempt=attempt,
                terminal=terminal, residual=residual, al=float(reach.sum()))


def path_product(par, edge):
    out = np.zeros(len(par), dtype=np.float64)
    for j, p in enumerate(par):
        out[j] = edge[j] * (1 if p < 0 else out[p])
    return out


def closure_masks(par, sib):
    kids = groups(par, sib); out=[]
    for j,p in enumerate(par):
        mask = 0 if p < 0 else out[p]
        for k in kids[p][:int(sib[j]) + 1]: mask |= 1 << k
        out.append(mask)
    return out


def valid_subset(mask, closure):
    return all(not (mask >> j & 1) or (mask & need) == need
               for j,need in enumerate(closure))


def prefix_knapsack(par, sib, weights, budget):
    """Exact tree DP with ancestor AND ordered-sibling-prefix constraints.

    Each subtree entry includes its node's globally weighted reward.
    Diagnostic only: score-dependent final retention may bias the verifier.
    """
    kids = groups(par, sib)
    budget = min(int(budget), len(par))
    @lru_cache(None)
    def solve(ctx):
        # Prefix of children included, indexed by exact cardinality.
        prefix = {0: (0.0, ())}
        best = dict(prefix)
        for child in kids.get(ctx, []):
            child_options = solve(child)
            nxt = {}
            for used,(score,ids) in prefix.items():
                for subused,(subscore,subids) in child_options.items():
                    count = used + 1 + subused
                    if count > budget: continue
                    candidate = (score + float(weights[child]) + subscore,
                                 ids + (child,) + subids)
                    if count not in nxt or candidate[0] > nxt[count][0] + 1e-14:
                        nxt[count] = candidate
            prefix = nxt
            for count,item in prefix.items():
                if count not in best or item[0] > best[count][0] + 1e-14:
                    best[count] = item
        return best
    _,answer = max(solve(-1).items(), key=lambda x:(x[1][0],x[0]))
    return sorted(answer[1]), float(answer[0])


def compact(par, sib, tokens, p_rows, q_rows, keep):
    keep = sorted(keep); remap = {j:k for k,j in enumerate(keep)}
    return ([(-1 if par[j] < 0 else remap[par[j]]) for j in keep],
            [sib[j] for j in keep], np.asarray(tokens)[keep],
            np.asarray(p_rows)[[0] + [j+1 for j in keep]],
            np.asarray(q_rows)[keep])
