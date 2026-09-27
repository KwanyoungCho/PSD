"""Capture-safe selected rules; scoped experiment hook, no engine edits."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ssd'))
import torch


def make_policy(key,temperature,pack_fn):
    source,norm,weight=key.removeprefix('policy::').split('__')
    if source=='interval1':source='proxy'
    if not (source in ('proxy','residual') or source.startswith(('floor','complement_power'))):
        raise ValueError(f'Not a frozen runtime source: {source}')
    if not (weight=='original' or weight.startswith(('expected_mix','uniform'))):
        raise ValueError(f'Not a frozen runtime weight: {weight}')
    def fn(exit_logits,q_logits,tokens,top_k,wire_n,pack_scores,source_ignored='residual'):
        k,v=q_logits.shape
        e=torch.softmax(exit_logits.float()/temperature,-1)
        q=torch.softmax(q_logits.float()/temperature,-1)
        ids=tokens[:k].view(k,1)
        alpha=(e[:k].gather(1,ids).squeeze(1)/(q.gather(1,ids).squeeze(1)+1e-10)).clamp(max=1)
        def hazard(a):
            reach=torch.cat([torch.ones_like(a[:1]),a.cumprod(0)],0)
            return torch.cat([reach[:-1]*(1-a),reach[-1:]],0)
        h=hazard(alpha)
        if weight.startswith('expected_mix'):
            w=float(weight.removeprefix('expected_mix'))
            alpha_mean=torch.minimum(e[:k],q).sum(-1).clamp(0,1)
            h=(1-w)*h+w*hazard(alpha_mean)
        elif weight.startswith('uniform'):
            w=float(weight.removeprefix('uniform'));h=(1-w)*h+w/(k+1)
        if source=='proxy':s=e[:k].clone()
        elif source=='residual':s=(e[:k]-q).clamp_min(0)
        elif source.startswith('floor'):
            floor=float(source.removeprefix('floor'))
            s=torch.maximum((e[:k]-q).clamp_min(0),floor*e[:k])
        else:
            power=float(source.removeprefix('complement_power'))
            s=e[:k]*(1-q).clamp_min(0).pow(power)
        s.scatter_(1,ids,0)
        s=torch.cat([s,e[k:k+1]],0)
        values,index=s.topk(int(top_k),-1)
        denom=values.sum(-1,keepdim=True) if norm=='topm' else s.sum(-1,keepdim=True)
        scores=h[:,None]*values/denom.clamp_min(1e-10)
        top,chosen=scores.flatten().topk(int(wire_n))
        positions=(chosen//int(top_k)).long()
        token=index.flatten().gather(0,chosen)
        if pack_scores:token=pack_fn(token,top)
        return positions,token,top
    return fn


def main():
    import ssd.engine.helpers.p2_tree as p2
    policy=os.environ['SSD_TF_POLICY'];t=float(os.environ['SSD_TF_T'])
    if policy!='legacy':
        p2.chain_proxy_candidates_fixed=make_policy(policy,t,p2.pack_piv)
    import ssd.engine.verifier as verifier_module
    Verifier=verifier_module.Verifier
    compute=Verifier._compute_and_send_proxy;candidate_calls={}
    def checked_compute(self,exit_logits,draft_tokens,logits_q,orig_bs,K,*args,**kwargs):
        cfg=self.target_model_runner.config
        if (orig_bs!=1 or not self.jit_speculate or int(K) not in self._chain_proxy_graphs
            or isinstance(exit_logits,dict) or cfg.duet_proxy_on_draft or cfg.duet_policy!='b'
            or verifier_module._E0_TRACE):
            raise RuntimeError('Unpatched eager candidate path')
        candidate_calls[int(K)]=candidate_calls.get(int(K),0)+1
        return compute(self,exit_logits,draft_tokens,logits_q,orig_bs,K,*args,**kwargs)
    Verifier._compute_and_send_proxy=checked_compute
    sys.path.insert(0,str(ROOT/'ssd/bench'))
    import bench
    original=bench.run_benchmark
    def run(args,llm,prompts,sampling_params):
        outputs,elapsed,metrics=original(args,llm,prompts,sampling_params)
        if not candidate_calls:raise RuntimeError('Candidate experiment hook was not called')
        data={'policy':policy,'temperature':t,'candidate_temperature':1. if policy=='legacy' else t,
              'seed':args.seed,'prompts':len(prompts),'wall_s':elapsed,
              'output_tokens':sum(len(o['token_ids']) for o in outputs),'metrics':metrics,
              'candidate_calls_by_K':candidate_calls}
        Path(os.environ['SSD_TF_LIVE_OUT']).write_text(json.dumps(data,indent=2))
        return outputs,elapsed,metrics
    bench.run_benchmark=run
    candidate_t=1. if policy=='legacy' else t
    print(f'[TF experiment] policy={policy}, candidate T={candidate_t}; sampler T={t} unchanged',flush=True)
    bench.main()


if __name__=='__main__':main()
