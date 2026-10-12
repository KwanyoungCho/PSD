"""Minimal B1 paper-tree acceptance port; no new forward/tree/KV executor.

Compute all conditional sibling ladders in a reusable CUDA graph, copy only
acceptance probabilities to CPU, and retain the original visited-node CPU
coin draws and the single final GPU multinomial. Child order and original
parent-q references are preserved. This is an adoption candidate, not enabled
by production imports.
"""
import torch


class PaperTreeLadder:
    @torch.inference_mode()
    def __init__(self, n, v, c, device):
        self.n, self.c = n, c
        self.p = torch.zeros(n + 1, v, device=device)
        self.q = torch.zeros(n, v, device=device)
        self.tt = torch.ones((), device=device)
        self.child = torch.full((n + 1, c), n, dtype=torch.long, device=device)
        self.valid = torch.zeros(n + 1, c, dtype=torch.bool, device=device)
        self.tok = torch.zeros(n + 1, dtype=torch.long, device=device)
        self.qref = torch.zeros(n + 1, dtype=torch.long, device=device)
        for _ in range(2): self.compute()
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.alpha, self.mass, self.residual = self.compute()

    def compute(self):
        p = torch.softmax(self.p / self.tt.clamp_min(1e-8), -1)
        residual = p
        draft = self.q[self.qref].clone()
        alphas, masses = [], []
        for s in range(self.c):
            token = self.tok[self.child[:, s]][:, None]
            live = self.valid[:, s:s+1]
            d = draft.gather(1, token)
            # Original scalar walk uses Python double division of float32
            # probabilities. Retain it to permit coupled-coin comparison.
            alpha = (residual.gather(1, token).double() /
                     torch.where(d > 0, d, torch.ones_like(d)).double()).clamp_max(1)
            alphas.append(torch.where(live, alpha, torch.zeros_like(alpha)))
            masses.append(torch.where(live, d, torch.ones_like(d)))
            r = (residual - draft).clamp_min(0)
            z = r.sum(-1, keepdim=True)
            r = torch.where(z > 1e-12, r / z.clamp_min(1e-30), torch.zeros_like(r))
            residual = torch.where(live, r, residual)
            if s + 1 < self.c:
                dnext = draft.scatter(1, token, 0)
                z = dnext.sum(-1, keepdim=True)
                dnext = torch.where(z > 1e-12, dnext / z.clamp_min(1e-30), torch.zeros_like(dnext))
                draft = torch.where(live, dnext, draft)
        residual = torch.where(residual.sum(-1, keepdim=True) > 1e-12, residual, p)
        return torch.cat(alphas, 1), torch.cat(masses, 1), residual

    @torch.inference_mode()
    def walk(self, tree, p, q, temp, coin, multinomial):
        from ssd.engine.helpers.p2_tree import validate_tree_ints
        checked = dict(tree)
        checked.setdefault('u_valid', q.shape[0])
        validate_tree_ints(checked, len(tree['tok']))
        n = int(tree['valid'])
        if n != self.n or p.shape[0] < n+1 or q.shape[0] < 1:
            raise ValueError('Paper tree graph shape mismatch')
        child = [[self.n] * self.c for _ in range(n+1)]
        valid = [[False] * self.c for _ in range(n+1)]
        refs = [0] * (n+1)
        for j in range(n):
            row, s = int(tree['parent_local'][j])+1, int(tree['sib_order'][j])
            if s >= self.c: raise ValueError('Sibling capacity exceeded')
            child[row][s], valid[row][s] = j, True
            if s == 0: refs[row] = int(tree['parent_q_ref'][j])
        dev = self.p.device
        self.p.copy_(p[:n+1]); self.q.zero_()
        # Distinct parent q rows never exceed the number of valid nodes.
        self.q[:min(n,q.shape[0])].copy_(q[:n])
        self.tt.fill_(temp)
        self.child.copy_(torch.tensor(child, device=dev))
        self.valid.copy_(torch.tensor(valid, device=dev))
        self.qref.copy_(torch.tensor(refs, device=dev))
        self.tok[:n].copy_(tree['tok'][:n]); self.tok[n] = 0
        self.graph.replay()
        values = torch.cat([self.alpha, self.mass.double()], 1).tolist()
        path, ctx = [], 0
        for _ in range(n+1):
            accepted = None
            for s, j in enumerate(child[ctx]):
                if not valid[ctx][s]: continue
                if values[ctx][self.c+s] <= 0:
                    raise ValueError('D_j[x_j]=0 — invalid proposal')
                if coin() < values[ctx][s]:
                    accepted = j
                    break
            if accepted is None:
                return path, int(multinomial(self.residual[ctx]))
            path.append(accepted); ctx = accepted+1
        raise RuntimeError('Tree walk exceeded valid nodes')


def install():
    import ssd.engine.helpers.p2_tree as tree_module
    reference = tree_module.tree_verify_walk_tensor
    graphs = {}

    def walk(tree, p, q, temp, coin, multinomial):
        n = int(tree['valid'])
        if n == 0:
            return reference(tree, p, q, temp, coin, multinomial)
        c = max(int(x) for x in tree['sib_order'][:n]) + 1
        key = n, p.shape[-1], c, p.device
        if key not in graphs:
            graphs[key] = PaperTreeLadder(*key[:3], key[3])
        return graphs[key].walk(tree, p, q, temp, coin, multinomial)

    tree_module.tree_verify_walk_tensor = walk
    return graphs
