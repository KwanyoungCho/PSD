"""Separate correctness-run instrumentation; disabled in performance trials."""
import json
import os
from pathlib import Path
import torch


def install():
    if os.environ.get('DUET_FOLLOWUP_TRACE') != '1':
        return
    import ssd.engine.helpers.p2_tree as pt
    from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor as cls
    from ssd.engine.draft_runner import DraftRunner
    original_init=cls.__init__;original_select=pt._arena_select_global
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.debug_buffers_enabled=True
        ar=self.arena
        ar._frontier_trace={k:torch.zeros((self.F,ar.capacity),dtype=getattr(ar,k).dtype,device=self.dev)
            for k in ('valid','state','depth','root','raw_q','logpri')}
        ar._frontier_trace['remaining']=torch.zeros((self.F,self.R),dtype=torch.long,device=self.dev)
        ar._frontier_trace['n']=torch.zeros(self.F,dtype=torch.long,device=self.dev)
        if os.environ.get('DUET_FOLLOWUP_PARITY')=='1':
            gen=torch.Generator(device=self.dev);gen.manual_seed(922760+self.R+self.F)
            self.parity_noise=[torch.empty(self.W,self.V,device=self.dev).exponential_(1,generator=gen) for _ in range(self.F)]
            self._followup_parity_seen=set()
    cls.__init__=init
    def select(ar,W,f,depth_cap,remaining,future_rounds,R,proxy_threshold=0.,conf_threshold=0.):
        for k in ('valid','state','depth','root','raw_q','logpri'):
            ar._frontier_trace[k][f].copy_(getattr(ar,k))
        ar._frontier_trace['remaining'][f].copy_(remaining)
        ar._frontier_trace['n'][f].copy_(ar.n.reshape(()))
        return original_select(ar,W,f,depth_cap,remaining,future_rounds,R,proxy_threshold,conf_threshold)
    pt._arena_select_global=select
    original_audit=DraftRunner._audit_tree_executor_node_coverage
    def audit(self,views,ex,R,*,phase):
        original_audit(self,views,ex,R,phase=phase)
        data={k:v.detach().cpu().tolist() for k,v in ex.arena._frontier_trace.items()}
        data.update(phase=phase,policy=os.environ['DUET_TREE_POLICY'],widths=list(ex.round_widths),
            max_nodes=ex.NV,depth_cap=ex.F,sel=ex.dbg_sel.cpu().tolist(),
            selected=ex.dbg_selv.cpu().tolist(),fanout=ex.dbg_fan.cpu().tolist(),
            root_prior=ex.in_root_piv.cpu().tolist())
        # Uninitialized arena tail is masked by n and valid when interpreted.
        path=Path(os.environ['DUET_FULL_OUT'])/f'frontier.{os.getpid()}.jsonl'
        with path.open('a') as stream:stream.write(json.dumps(data)+'\n')
    DraftRunner._audit_tree_executor_node_coverage=audit
    if os.environ.get('DUET_FOLLOWUP_PARITY')=='1':
        original_replay=cls.replay
        def replay(self,n_pages0):
            original_replay(self,n_pages0)
            if n_pages0 in self._followup_parity_seen:return
            self._followup_parity_seen.add(n_pages0)
            names=('out_tok','out_par','out_sib','out_valid','out_pq_ref','out_pq_cells','out_u_valid',
                   'dbg_sel','dbg_selv','dbg_fan','dbg_toks')
            saved={k:getattr(self,k).clone() for k in names}
            logits=self.cell_logits.clone();raw=self.out_rawq.clone()
            self.run_once(n_pages0)
            errors=[k for k,v in saved.items() if not torch.equal(v,getattr(self,k))]
            valid=torch.cat([self.dbg_selv[f,:w] for f,w in enumerate(self.round_widths)])
            diff=(logits[valid]-self.cell_logits[valid]).abs()
            maxerr=float(diff.max()) if diff.numel() else 0.
            rawerr=float((raw-self.out_rawq).abs().max())
            if not torch.isfinite(diff).all() or maxerr>1e-3 or rawerr>1e-6 or errors:
                raise RuntimeError(f'Graph/eager mismatch: fields={errors}, logits={maxerr}, rawq={rawerr}')
            result=dict(phase=self.phase,roots=self.R,pages=n_pages0,policy=os.environ['DUET_TREE_POLICY'],
                passed=True,logits_max_error=maxerr,rawq_max_error=rawerr,integer_fields=len(names))
            path=Path(os.environ['DUET_FULL_OUT'])/f'parity.{os.getpid()}.jsonl'
            with path.open('a') as stream:stream.write(json.dumps(result)+'\n')
        cls.replay=replay
