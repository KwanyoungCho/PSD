"""Score-selected, ancestor-closed trees for target-sampling verification.

Original q is a ranking statistic here, not the proposal law of the pruned
tree. Never combine these selected nodes with an original-q p/q verifier.
"""
import math


def select_tree(parents, raw_q, cap):
    if cap < 1 or len(parents) != len(raw_q):
        raise ValueError('Invalid score tree size')
    scores=[]
    for j,(p,q) in enumerate(zip(parents,raw_q)):
        if not -1 <= int(p) < j or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError(f'Invalid score tree node {j}')
        scores.append((scores[p] if p>=0 else 0.)+(math.log(q) if q>0 else -math.inf))
    # Ancestor confidence >= descendant confidence. Generation-index ties
    # therefore preserve ancestors, including zero/underflow probabilities.
    keep=sorted(sorted(range(len(parents)),key=lambda j:(-scores[j],j))[:cap])
    index={old:new for new,old in enumerate(keep)}
    compact=[];siblings=[];counts={}
    for old in keep:
        p=parents[old]
        if p>=0 and p not in index:raise AssertionError('Score selection removed an ancestor')
        parent=-1 if p<0 else index[p]
        compact.append(parent)
        siblings.append(counts.get(parent,0));counts[parent]=siblings[-1]+1
    return keep,compact,siblings


def generation_capacity(round_widths, siblings):
    """Natural finite bound from actual draft inputs; no extra tuning knob."""
    if siblings<1 or not round_widths or min(round_widths)<1:
        raise ValueError('Positive draft widths and sibling count required')
    return int(sum(round_widths)*siblings)
