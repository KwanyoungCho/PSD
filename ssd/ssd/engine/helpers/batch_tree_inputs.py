"""Prepare all request arenas with one captured graph and one host upload."""
import torch
from ssd.engine.helpers.tree_host_topology import context_topology


class BatchedArenaInputs:
    @torch.inference_mode()
    def __init__(self,executor,pages,max_model_len,pool=None):
        self.executor=executor;self.pages=pages;self.max_model_len=max_model_len
        ex=executor.first;B=len(executor.executors);G=ex.in_glue.shape[1]
        # Prefix, width, safe, physical pages, depths, visibility, FA page IDs.
        self.G=G;self.M=ex.max_blocks;self.P=pages+ex.canvas_extra_pages
        self.stride=3+self.M+G+G*G+self.P
        self.meta=torch.zeros(B,self.stride,dtype=torch.int64,device=ex.dev)
        self.tokens=torch.zeros(B,ex.R,dtype=torch.int64,device=ex.dev)
        self.scores=torch.zeros(B,ex.R,device=ex.dev)
        self.contexts=torch.zeros_like(self.tokens)
        self.temps=torch.ones(B,device=ex.dev)
        def run():
            for b,ar in enumerate(executor.executors):
                row=self.meta[b];prefix,n,safe=row[0],row[1],row[2]
                bt=row[3:3+self.M]
                depth=row[3+self.M:3+self.M+G]
                visible=row[3+self.M+G:3+self.M+G+G*G].view(G,G)
                ci=self.contexts[b].clamp_min(0).minimum((n-1).clamp_min(0))
                ar.in_root_tok.copy_(self.tokens[b]);ar.in_root_piv.copy_(self.scores[b]*safe)
                ar.in_glue.copy_(visible[ci])
                ar.in_rope_base.copy_(prefix+1+depth[ci])
                ar.in_glue_w.copy_(n);ar.in_prefix_len.copy_(prefix)
                ar.in_temps.copy_(self.temps[b].expand_as(ar.in_temps))
                ar.in_block_tables[0].copy_(bt)
                final=prefix+n+ar.total_cells
                for f,w in enumerate(ar.round_widths):
                    pos=prefix+n+ar.round_offsets[f]+torch.arange(w,device=ar.dev)
                    slots=bt[(pos//ar.bs).clamp(0,self.M-1)]*ar.bs+pos%ar.bs
                    ar.in_slot[f].fill_(-1)
                    ar.in_slot[f][:w].copy_(torch.where(safe.bool(),slots,-1))
                    ar.in_ctx_len[f].copy_(final.clamp_max(max_model_len).expand_as(ar.in_ctx_len[f]))
                    ar.wrappers[pages][f]._paged_kv_indices_buf.copy_(row[-self.P:])
        for _ in range(2):run()
        torch.cuda.synchronize(ex.dev)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool):run()

    @torch.inference_mode()
    def replay(self,rows,params,temps):
        B=len(rows);ex=self.executor.first
        host=torch.zeros_like(self.meta,device='cpu');safe=[]
        self.tokens.zero_();self.scores.zero_();self.contexts.zero_()
        for b in range(len(self.executor.executors)):
            row=rows[min(b,B-1)];prefix=row['prefix'];n=len(row['tokens']);bt=row['blocks']
            final=prefix+n+ex.total_cells
            okay=b<B and final<=self.max_model_len and (final-1)//ex.bs<len(bt) and all(x>=0 for x in bt[:(final+ex.bs-1)//ex.bs])
            if b<B:safe.append(okay)
            depths,_,vis=context_topology(row['parents'])
            host[b,:3]=torch.tensor([prefix,n,int(okay)])
            host[b,3:3+len(bt)]=torch.tensor(bt)
            host[b,3+self.M:3+self.M+n]=torch.tensor([0]+[d+1 for d in depths])
            host[b,3+self.M+self.G:3+self.M+self.G+self.G*self.G].view(self.G,self.G)[:n,:n]=torch.as_tensor(vis)
            live=min((final+ex.bs-1)//ex.bs,len(bt)) if okay else (prefix+n+ex.bs-1)//ex.bs
            host[b,-self.P:]=torch.tensor([bt[j] if j<live else bt[0] for j in range(self.P)])
            if b<B:
                tok,score,ctx=params[b];nr=len(tok)
                self.tokens[b,:nr].copy_(tok);self.scores[b,:nr].copy_(score);self.contexts[b,:nr].copy_(ctx)
        self.meta.copy_(host)
        self.temps[:B].copy_(torch.tensor(temps))
        self.graph.replay()
        return safe
