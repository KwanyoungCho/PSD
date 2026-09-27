"""Graph-safe exact node allocation for a common nondecreasing gain curve.

For sorted nonnegative reach weights, rearrangement assigns larger fanouts to
larger reach. Enumerating integer partitions of the tiny node budget therefore
solves the otherwise exponential per-root problem, including nonconcave gain.
"""
import itertools
import numpy as np
import torch


def partitions(budget, c=3):
    patterns = []
    def visit(prefix, left, maximum):
        patterns.append(prefix + [0]*(budget-len(prefix)))
        for k in range(1, min(maximum, left)+1):
            visit(prefix+[k], left-k, k)
    visit([], budget, c)
    # Exact reward ties prefer more used nodes, then broader allocations.
    return sorted(patterns, key=lambda x: (-sum(x), -sum(c > 0 for c in x), x))


def make_constants(budget, curve, device='cpu'):
    curve = np.asarray(curve, dtype=float)
    if curve[0] != 0 or np.any(np.diff(curve) < 0) or np.any(curve > 1):
        raise ValueError('Gain curve must be nondecreasing, 0..1, gamma(0)=0')
    pat = torch.tensor(partitions(budget, len(curve)-1), device=device, dtype=torch.long)
    gamma = torch.tensor(curve, device=device, dtype=torch.float64)
    return dict(patterns=pat, values=gamma[pat], cost=pat.sum(1), count=(pat>0).sum(1))


def allocate(root, log_reach, valid, remaining, future_rounds, constants):
    """Inputs are selected parent lanes; output c in [0,C] for every lane.

    No tensor-to-host reads, dynamic output shapes or child token arguments.
    Physical forward width and future-round reserve remain unchanged.
    """
    r = remaining.numel(); w = root.numel()
    b = constants['patterns'].shape[1]
    safe_root = torch.where(valid, root, torch.zeros_like(root))
    score = torch.where(valid, log_reach.double(), -torch.inf)
    order = torch.argsort(score, descending=True, stable=True)
    rr = safe_root[order]; vv = valid[order]
    oh = torch.nn.functional.one_hot(rr, r).long()*vv[:, None].long()
    rank = ((oh.cumsum(0)-oh)*oh).sum(1)
    # Selector's per-root quota is <=remaining<=b. Invalid lanes contribute 0.
    index = rr*b + rank.clamp(max=b-1)
    mass = torch.zeros(r*b, dtype=torch.float64, device=root.device)
    mass.scatter_add_(0, index, torch.where(vv, score[order].exp(), 0.))
    mass = mass.reshape(r, b)
    count = oh.sum(0)
    now = torch.where(remaining > 0, (remaining-int(future_rounds)).clamp(min=1), 0)
    values = (mass[:, None, :]*constants['values'][None, :, :]).sum(2)
    allowed = (constants['cost'][None, :] <= now[:, None]) & (constants['count'][None, :] <= count[:, None])
    values = torch.where(allowed, values, -torch.inf)
    choice = values.argmax(1)
    fan = constants['patterns'][choice[rr], rank.clamp(max=b-1)]
    fan = torch.where(vv, fan, 0)
    result = torch.zeros(w, dtype=torch.long, device=root.device)
    return result.scatter(0, order, fan)


def exhaustive(weights, curve, budget):
    best = -1.; best_fan = None
    for fan in itertools.product(range(len(curve)), repeat=len(weights)):
        if sum(fan) > budget:
            continue
        value = sum(w*curve[c] for w, c in zip(weights, fan))
        if value > best:
            best, best_fan = value, fan
    return best, best_fan
