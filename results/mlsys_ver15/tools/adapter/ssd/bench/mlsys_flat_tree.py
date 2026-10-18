"""Review-only flat forest probability work; request contexts remain separate.

Unlike dense [B,N+1,V] ladders, vocabulary-sized work runs only on the Q
physical query contexts. Only small top-k scores are restored to [B,N+1,K].
No production policy change: ordered sibling rejection and correction are
identical to batch_tree_sampling. Kept experimental until paired validation.
"""
import torch
import numpy as np
from ssd.engine.helpers.p2_tree import pack_piv, q_probs_from_logits


def topology(parents, siblings, n, c_max, device, query_capacity=None):
    b=len(parents)
    total=sum(len(p)+1 for p in parents)
    q=total if query_capacity is None else query_capacity
    if q<total: raise ValueError('Flat capacity too small')
    child=np.zeros((q,c_max),dtype=np.int64)
    valid=np.zeros((q,c_max),dtype=bool)
    par=np.zeros(q,dtype=np.int64);sib=par.copy()
    req=par.copy();local=par.copy();root=np.zeros(q,dtype=bool)
    ctx_valid=root.copy();dense=np.zeros((b,n+1),dtype=np.int64)
    dense_valid=np.zeros((b,n+1),dtype=bool)
    roots=[];off=0
    for i,(ps,ss) in enumerate(zip(parents,siblings)):
        if len(ps)!=len(ss) or len(ps)>n: raise ValueError('Invalid tree size')
        roots.append(off);root[off]=True
        z=len(ps)+1
        req[off:off+z]=i;local[off:off+z]=np.arange(z)
        ctx_valid[off:off+z]=True
        dense[i,:]=off;dense[i,:z]=np.arange(off,off+z);dense_valid[i,:z]=True
        for j,(p,s) in enumerate(zip(ps,ss)):
            if p>=j or p<-1 or s<0 or s>=c_max: raise ValueError('Invalid topology')
            a=off+p+1;node=off+j+1
            if valid[a,s]: raise ValueError('Duplicate sibling order')
            child[a,s]=node;valid[a,s]=True;par[node]=a;sib[node]=s
        off+=z
    result=dict(child=child,child_valid=valid,par=par,sib=sib,req=req,
                local=local,root=root,valid=ctx_valid,dense=dense,dense_valid=dense_valid,
                roots=np.asarray(roots,dtype=np.int64))
    return {k:torch.from_numpy(v).to(device) for k,v in result.items()}


def pack_nodes(t, topo):
    """Dense [B,N,...] proposal payload to flat context rows (root unused)."""
    return t[topo['req'],(topo['local']-1).clamp_min(0)]


def ladder(tokens,p,q,topo,depth,exact=False):
    child=topo['child'];cv=topo['child_valid']
    residual=p
    draft=q[child[:,0]]
    draft=torch.where(cv[:,:1],draft,torch.zeros_like(draft))
    reject=p.new_ones(p.shape[0],1)
    alphas=[];pres=[]
    for s in range(child.shape[1]):
        tok=tokens[child[:,s],None]
        valid=cv[:,s:s+1]
        d=draft.gather(1,tok)
        if exact:valid=valid&(d>0)
        denom=torch.where(d>0,d,torch.ones_like(d)) if exact else d+1e-10
        alpha=(residual.gather(1,tok)/denom).clamp(max=1)*valid
        alphas.append(alpha[:,0]);pres.append(reject[:,0]);reject=reject*(1-alpha)
        newer=(residual-draft).clamp_min(0)
        z=newer.sum(-1,keepdim=True)
        newer=torch.where(z>1e-12,newer/z.clamp_min(1e-30),torch.zeros_like(newer))
        residual=torch.where(valid,newer,residual)
        if s<child.shape[1]-1:
            newer=draft.scatter(1,tok,0);z=newer.sum(-1,keepdim=True)
            newer=torch.where(z>1e-12,newer/z.clamp_min(1e-30),torch.zeros_like(newer))
            draft=torch.where(valid,newer,draft)
    residual=torch.where(residual.sum(-1,keepdim=True)>1e-12,residual,p)
    alpha=torch.stack(alphas,-1)
    if exact:return alpha,None,residual
    pre=torch.stack(pres,-1)
    base=alpha[topo['par'],topo['sib']]*pre[topo['par'],topo['sib']]
    reach=topo['root'].to(p.dtype)
    for _ in range(depth):
        reach=torch.where(topo['root'],torch.ones_like(reach),base*reach[topo['par']])*topo['valid']
    return alpha,reach*reject[:,0],residual


def candidates(exit_logits,q_logits,tokens,topo,wire_n,depth,top_k,tt,dt):
    t=torch.where(tt>0,tt,torch.ones_like(tt))[topo['req']]
    d=torch.where(dt>0,dt,torch.ones_like(dt))[topo['req']]
    p=(exit_logits.float()/t[:,None]).softmax(-1)
    q=(q_logits.float()/d[:,None]).softmax(-1)
    _,terminal,scores=ladder(tokens,p,q,topo,depth)
    v=scores.shape[-1]
    exclude=torch.where(topo['child_valid'],tokens[topo['child']],v)
    scores=torch.cat([scores,scores.new_zeros(scores.shape[0],1)],1).scatter(1,exclude,0)[:,:v]
    values,ids=scores.topk(min(top_k,v),-1)
    piv=values/values.sum(-1,keepdim=True).clamp_min(1e-10)*terminal[:,None]
    piv=piv[topo['dense']]*topo['dense_valid'][...,None]
    ids=ids[topo['dense']]
    score,index=piv.flatten(1).topk(min(wire_n,piv[0].numel()),1)
    pos=index//values.shape[-1];tok=ids.flatten(1).gather(1,index)
    if pos.shape[1]<wire_n:
        pad=wire_n-pos.shape[1];b=pos.shape[0]
        pos=torch.cat([pos,pos.new_zeros(b,pad)],1)
        tok=torch.cat([tok,tok.new_zeros(b,pad)],1)
        score=torch.cat([score,score.new_zeros(b,pad)],1)
    return pos,pack_piv(tok,score)


def verify(logits_p,logits_q,tokens,topo,n,tt,dt,sampler_x=None,fan_out=1,
           greedy=False,uniforms=None,stochastic_only=True):
    b=topo['roots'].numel();q,v=logits_p.shape
    child=topo['child'];cv=topo['child_valid']
    if greedy:
        argmax=logits_p.argmax(-1)
        accepted=(tokens[child]==argmax[:,None])&cv
    else:
        t=tt[topo['req']]
        p=(logits_p.float()/t.clamp_min(1e-10)[:,None]).softmax(-1)
        if not stochastic_only:
            gp=torch.zeros_like(p).scatter(1,logits_p.argmax(-1,keepdim=True),1)
            p=torch.where((t==0)[:,None],gp,p)
        draft=q_probs_from_logits(logits_q,dt[topo['req']],sampler_x,fan_out)
        alpha,_,residual=ladder(tokens,p,draft,topo,n,exact=True)
        u=torch.rand(b,n,device=logits_p.device) if uniforms is None else uniforms
        coins=u[topo['req'],(topo['local']-1).clamp_min(0)][child]
        accepted=(coins<alpha)&cv
    order=torch.arange(child.shape[-1],device=child.device)
    first=torch.where(accepted,order,child.shape[-1]).min(-1).values
    selected=child.gather(1,first.clamp_max(child.shape[-1]-1)[:,None])[:,0]
    selected=torch.where(first<child.shape[-1],selected,-1)
    ctx=topo['roots'].clone();live=torch.ones_like(ctx,dtype=torch.bool);path=[]
    for _ in range(n):
        node=selected[ctx];live=live&(node>=0)
        path.append(torch.where(live,topo['local'][node.clamp_min(0)]-1,-1))
        ctx=torch.where(live,node,ctx)
    recovery=argmax[ctx,None] if greedy else torch.multinomial(residual[ctx],1)
    return torch.stack(path,1),recovery,topo['local'][ctx,None]


class FlatProxy:
    """Lazy persistent graphs per physical Q; compatible with review verifier."""
    def __init__(self,b,n,v,dtype,device,wire_n,depth,top_k,pool=None,q_dtype=None,c_max=3,**kwargs):
        self.b,self.n,self.v=b,n,v
        self.dtype,self.q_dtype,self.device=dtype,q_dtype or dtype,device
        self.wire,self.depth,self.top_k,self.c_max=wire_n,depth,top_k,c_max
        self.pool=pool;self.cells={};self.topology={}
        # Existing verifier asks for a pool immediately. Captures are deferred
        # until topology/physical Q are known; no artificial empty capture.
        class Pool:
            def __init__(self,p):self.p=p
            def pool(self):return self.p
        self.graph=Pool(pool)

    @torch.inference_mode()
    def prepare(self,parents,siblings,tokens,q,target_temps=1.,draft_temps=1.):
        self.actual=len(parents)
        size=((sum(len(p)+1 for p in parents)+7)//8)*8
        topo=topology(parents,siblings,self.n,self.c_max,self.device,size)
        if size not in self.cells:
            cell=dict(topo={k:torch.empty_like(v) for k,v in topo.items()},
                logits=torch.zeros(size,self.v,dtype=self.dtype,device=self.device),
                q=torch.zeros(size,self.v,dtype=self.q_dtype,device=self.device),
                tokens=torch.zeros(size,dtype=torch.long,device=self.device),
                tt=torch.ones(self.actual,device=self.device,dtype=torch.float32),
                dt=torch.ones(self.actual,device=self.device,dtype=torch.float32))
            self.cells[size]=cell
        cell=self.cells[size]
        for k,v in topo.items():cell['topo'][k].copy_(v)
        cell['q'].copy_(pack_nodes(q,topo));cell['tokens'].copy_(pack_nodes(tokens,topo))
        cell['tt'].copy_(torch.as_tensor(target_temps,device=self.device,dtype=torch.float32))
        cell['dt'].copy_(torch.as_tensor(draft_temps,device=self.device,dtype=torch.float32))
        self.current=cell;self.tokens=tokens
        if 'graph' not in cell:
            def run():return candidates(cell['logits'],cell['q'],cell['tokens'],cell['topo'],
                                         self.wire,self.depth,self.top_k,cell['tt'],cell['dt'])
            for _ in range(2):run()
            torch.cuda.synchronize(self.device);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):cell['out']=run()
            cell['graph']=graph
            if self.pool is None:self.pool=graph.pool()
        self.graph=cell['graph']

    @torch.inference_mode()
    def replay(self,logits):
        self.current['logits'].copy_(logits);self.graph.replay()
        return self.current['out']


class FlatAccept:
    def __init__(self,proxy,sampler_x,fan_out,greedy,pool=None,stochastic_only=False):
        self.proxy=proxy;self.sampler_x=sampler_x;self.fan_out=fan_out
        self.greedy=greedy;self.stochastic_only=stochastic_only;self.pool=pool;self.cells={}
        self._capture()

    @torch.inference_mode()
    def _capture(self):
        p=self.proxy;c=p.current;key=c['logits'].shape[0]
        if key not in self.cells:
            logits=torch.zeros_like(c['logits'])
            def run():return verify(logits,c['q'],c['tokens'],c['topo'],p.n,c['tt'],c['dt'],
                self.sampler_x,self.fan_out,self.greedy,stochastic_only=self.stochastic_only)
            for _ in range(2):run()
            torch.cuda.synchronize(p.device);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):out=run()
            if self.pool is None:self.pool=graph.pool()
            self.cells[key]=(logits,graph,out)
        self.live=self.cells[key];self.graph=self.live[1]

    @torch.inference_mode()
    def replay(self,logits,tt,dt):
        self._capture();self.live[0].copy_(logits);self.graph.replay()
        return self.live[2]


def install():
    """Review-only TP1 integration. Uniform trees keep grouped dense execution."""
    import os,inspect,textwrap
    mode=os.getenv('DUET_REVIEW_FLAT','0')
    if mode not in ('1','prob'):return
    from mlsys_tree_candidates import FixedPlanPackedTree
    from ssd.engine.helpers import batch_tree_verify as module
    from ssd.engine.model_runner import ModelRunner
    # The fixed-plan engine itself is unchanged except that logits remain Q,V.
    source=textwrap.dedent(inspect.getsource(FixedPlanPackedTree.run.__wrapped__))
    source=source.replace('proxy(logits.index_select(0, dense).view(batch, width, -1), batch)',
                          'proxy(logits, batch)')
    source=source.replace('return None if logits is None else logits.index_select(0, dense).view(b, width, -1)',
                          'return logits')
    import mlsys_tree_candidates as cand
    scope={};exec(compile(source,'<flat-review-forward>','exec'),cand.__dict__,scope)
    class FlatForward(FixedPlanPackedTree):pass
    FlatForward.run=torch.inference_mode()(scope['run'])
    globals()['FlatForward']=FlatForward
    original_forward=ModelRunner.run_batched_tree
    def forward(self,rows,width):
        if not getattr(self,'_review_flat_active',False):return original_forward(self,rows,width)
        if self.num_tp_gpus!=1:raise ValueError('Flat review currently validates TP1 only')
        if mode=='prob':
            # Preserve grouped target attention/GEMM exactly; remove only
            # padding from vocabulary-sized proxy and acceptance work.
            topo=self._review_flat_proxy.current['topo'];callback=self._duet_proxy_fn
            self._duet_proxy_fn=lambda logits,b:callback(logits[topo['req'],topo['local']],b)
            try:output=original_forward(self,rows,width)
            finally:self._duet_proxy_fn=callback
            self._review_physical_queries=len(rows)*width
            return output[topo['req'],topo['local']]
        if not hasattr(self,'_review_flat_forward'):self._review_flat_forward=FlatForward(self,True)
        self._review_physical_queries=((sum(len(row['tokens']) for row in rows)+7)//8)*8
        return self._review_flat_forward.run(rows,width,self._duet_proxy_fn)
    ModelRunner.run_batched_tree=forward
    original=module.verify
    source=textwrap.dedent(inspect.getsource(original.__wrapped__))
    source=source.replace('BatchedTreeProxy(', 'FlatProxy(').replace('BatchedTreeAccept(', 'FlatAccept(')
    source=source.replace('proxy_key=cap,N',"proxy_key=cap,N,'flat'")
    source=source.replace('proxy=r._batch_tree_proxies[proxy_key]',
                          'proxy=r._batch_tree_proxies[proxy_key]\n    r._review_flat_proxy=proxy')
    source=source.replace('key=cap,N,greedy,cfg.stochastic_only',"key=cap,N,greedy,cfg.stochastic_only,'flat'")
    scope={};glob=dict(module.__dict__,FlatProxy=FlatProxy,FlatAccept=FlatAccept)
    exec(compile(source,'<flat-review-verify>','exec'),glob,scope)
    flat_verify=torch.inference_mode()(scope['verify'])
    def dispatch(verifier,seqs,result):
        r=verifier.target_model_runner;cfg=r.config
        n=next(x for x in sorted({cfg.duet_phase1_k,cfg.duet_phase2_k,cfg.duet_response_token_width}) if x>=result.speculations.shape[1]-1)
        q=((sum(s.verify_valid_k+1 for s in seqs)+7)//8)*8
        use=(q<module.capacity(len(seqs))*(n+1) and module.capacity(len(seqs))==len(seqs))
        r._review_flat_active=use
        r._review_physical_queries=q if use else module.capacity(len(seqs))*(n+1)
        try:return (flat_verify if use else original)(verifier,seqs,result)
        finally:r._review_flat_active=False
    module.verify=dispatch
