"""Exact acceptance-coin expectation of the ORIGINAL recursive residual walk."""
import torch

R, W, K, C, G = 10, 10, 10, 3, 300
ARMS = ('duet', 'equal_tree', 'chain')


def merge(trees):
    paths = {(): -1}; tokens = []; parents = []; mapping = {}
    for arm in ARMS:
        tree = trees[arm]; prefix = []; indices = []
        for j, (token, parent) in enumerate(zip(tree['tokens'], tree['parents'])):
            if not -1 <= parent < j:
                raise ValueError('Noncausal shadow tree')
            path = (prefix[parent] if parent >= 0 else ()) + (token,)
            prefix.append(path)
            if path not in paths:
                paths[path] = len(tokens); tokens.append(token)
                parents.append(paths[path[:-1]])
            indices.append(paths[path])
        mapping[arm] = indices
    return dict(tokens=tokens, parents=parents, mapping=mapping)


def expected(tree, p_logits, q_logits, temperature=.7, sampler_x=None,
             async_fan_out=None, eos=2, root_token=-1):
    from ssd.engine.helpers.p2_tree import q_probs_from_logits
    parents, tokens, refs = tree['parents'], tree['tokens'], tree['qrefs']
    n = len(tokens); device = p_logits.device
    if n == 0:
        return dict(expected_al=1., expected_usable_al=1., survival=[1.]+[0.]*K)
    groups = {}
    for j, parent in enumerate(parents):
        groups.setdefault(parent, []).append(j)
    for siblings in groups.values():
        siblings.sort(key=lambda j: tree['siblings'][j])
        if [tree['siblings'][j] for j in siblings] != list(range(len(siblings))):
            raise ValueError('Non-prefix siblings')
        if len({refs[j] for j in siblings}) != 1:
            raise ValueError('Sibling proposals disagree on parent q')
    contexts = list(groups)
    rr = torch.softmax(p_logits[torch.tensor([p+1 for p in contexts], device=device)].float()/temperature, -1)
    qq = q_probs_from_logits(q_logits.float(), torch.full((len(q_logits),), temperature, device=device),
                             sampler_x, async_fan_out)
    dd = qq[torch.tensor([refs[groups[p][0]] for p in contexts], device=device)].clone()
    u = len(contexts)
    tok = torch.zeros((u, C), dtype=torch.long, device=device)
    valid = torch.zeros((u, C), dtype=torch.bool, device=device)
    for i, parent in enumerate(contexts):
        for s, j in enumerate(groups[parent]):
            tok[i, s] = tokens[j]; valid[i, s] = True
    survive = torch.ones(u, dtype=torch.float64, device=device)
    weights = torch.zeros((u, C), dtype=torch.float64, device=device)
    for s in range(C):
        idx = tok[:, s:s+1]
        denominator = dd.gather(1, idx).reshape(-1)
        numerator = rr.gather(1, idx).reshape(-1)
        if bool(((denominator <= 0) & valid[:, s] & (survive > 0)).any()):
            raise ValueError('Native proposal has zero probability')
        alpha = (numerator.double()/denominator.double().clamp_min(1e-300)).clamp(max=1)
        weights[:, s] = torch.where(valid[:, s], survive*alpha, 0.)
        survive = torch.where(valid[:, s], survive*(1-alpha), survive)
        rr = (rr-dd).clamp_min(0)
        z = rr.sum(-1, keepdim=True)
        rr = torch.where(z > 1e-12, rr/z.clamp_min(1e-30), torch.zeros_like(rr))
        dd.scatter_(1, idx, 0.)
        z = dd.sum(-1, keepdim=True)
        dd = torch.where(z > 1e-12, dd/z.clamp_min(1e-30), torch.zeros_like(dd))
    ww = weights.cpu().tolist()
    edge = [0.]*n
    for i, parent in enumerate(contexts):
        for s, j in enumerate(groups[parent]):
            edge[j] = ww[i][s]
    mass = []; usable = []; depths = []; survival = [1.] + [0.]*K;native_survival=[1.]+[0.]*K
    for j, parent in enumerate(parents):
        depth = 1 if parent < 0 else depths[parent]+1
        depths.append(depth)
        mass.append(edge[j]*(1. if parent < 0 else mass[parent]))
        native_survival[depth]+=mass[j]
        live = root_token != eos and (parent < 0 or tokens[parent] != eos)
        usable.append(edge[j]*(1. if parent < 0 else usable[parent]) if live else 0.)
        survival[depth] += usable[j]
    variance=max(0.,1.+sum((2*d+1)*native_survival[d] for d in range(1,K+1))-(1.+sum(mass))**2)
    return dict(expected_al=1.+sum(mass), expected_usable_al=1.+sum(usable),variance_al=variance,
                survival=survival, max_depth=max(depths), first_child_depth=first_depth(tree))


def first_depth(tree):
    first = {p:j for j,(p,s) in enumerate(zip(tree['parents'],tree['siblings'])) if s == 0}
    node=-1; depth=0
    while node in first:
        node=first[node]; depth+=1
    return depth


def check_native(tree,p_logits,q_logits,prediction,temperature,sampler_x,async_fan_out,seed):
    """Finite real-vocabulary cross-check; private coins leave online RNG alone."""
    import random,statistics,math
    from ssd.engine.helpers.p2_tree import q_probs_from_logits,tree_verify_walk_tensor
    p=p_logits.float().cpu();q=q_logits.float().cpu()
    qp=q_probs_from_logits(q,torch.full((len(q),),temperature),sampler_x,async_fan_out)
    ti=dict(valid=len(tree['tokens']),tok=tree['tokens'],parent_local=tree['parents'],
            sib_order=tree['siblings'],parent_q_ref=tree['qrefs'],u_valid=len(q))
    rng=random.Random(seed);values=[]
    for _ in range(256):
        path,_=tree_verify_walk_tensor(ti,p,qp,temperature,rng.random,lambda probs:0)
        values.append(1+len(path))
    sampled=statistics.mean(values);se=math.sqrt(prediction['variance_al']/len(values))
    if abs(sampled-prediction['expected_al'])>6*se+.01:
        raise RuntimeError(('Real-model expectation/native mismatch',sampled,prediction,se))
    return dict(draws=len(values),sampled_al=sampled,standard_error=se,
                exact_al=prediction['expected_al'],status='pass')
