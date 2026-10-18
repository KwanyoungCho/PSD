"""Captured root-wise path-score selection and compact metadata export.

No model/RNG work and no host readback inside selection. The host receives
only each root's final M-node topology, rather than the whole search arena.
"""
import torch


class ScoreTreeExport:
    @torch.inference_mode()
    def __init__(self,executor,cap):
        a=executor.arena;R=executor.R;N=a.capacity;M=int(cap);dev=a.tok.device
        if M<1:raise ValueError('Positive final tree budget required')
        self.M=M;self.executor=executor
        rank=torch.arange(M,device=dev)
        # Graphs retain pointers, not Python ownership of tensors created
        # outside capture. Keep this constant alive across allocator reuse.
        self.rank=rank
        def run():
            valid=a.valid&(a.depth>0)&(a.root>=0)&(a.root<R)
            # Root q=1. Repeated parent recurrence computes the same path
            # log-probability as the CPU reference without using root priors.
            edge=a.raw_q.double().log();scores=edge
            parent=a.parent_idx.clamp_min(0)
            for _ in range(executor.F+1):
                scores=edge+torch.where(a.parent_idx>=0,scores[parent],0.)
            # Canonical +0 and a finite floor keep valid q=0 nodes ahead of
            # padding while retaining generation-order ties.
            scores=torch.where(scores==0,0.,scores)
            scores=torch.where(torch.isneginf(scores),-1e300,scores)
            owned=valid[None,:]&(a.root[None,:]==torch.arange(R,device=dev)[:,None])
            ranked=torch.where(owned,scores[None,:],-torch.inf)
            selected=ranked.argsort(dim=1,descending=True,stable=True)[:,:min(M,N)]
            selected=torch.where(owned.gather(1,selected),selected,N)
            if M>N:selected=torch.cat([selected,selected.new_full((R,M-N),N)],1)
            # Restore topological generation order within each selected root.
            selected=selected.sort(1).values
            live=selected<N;source=selected.clamp_max(N-1)
            remap=torch.full((N+1,),-1,dtype=torch.long,device=dev)
            remap.scatter_(0,selected.reshape(-1),rank.expand(R,M).reshape(-1))
            p=a.parent_idx[source];ps=p.clamp_min(0)
            parent_local=torch.where(p<R,-1,remap[ps])
            sibling=((parent_local[:,:,None]==parent_local[:,None,:])&
                     (rank[None,:,None]>rank[None,None,:])&live[:,None,:]).sum(-1)
            counts=torch.zeros(R+1,dtype=torch.long,device=dev)
            root=torch.where(valid,a.root,R)
            counts.scatter_add_(0,root,valid.long())
            return torch.cat([live.sum(1,keepdim=True),counts[:R,None],
                a.tok[source],parent_local,sibling,a.cell[ps]],1)
        for _ in range(2):run()
        torch.cuda.synchronize(dev)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=run()

    def replay(self):
        self.graph.replay()
        return self.output
