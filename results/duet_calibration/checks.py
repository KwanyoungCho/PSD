"""Scientific invariants and domain guards; no inference jobs."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from campaign import HERE,config
from calibrator import Calibrator,load_run,metrics,profile_rows
from adapters import ChainAdapter,CurrentTreeAdapter


def check_replay():
    """Exact residual oracle, variable K, and exclusion of warmup snapshots."""
    from replay_candidates import evaluate,hz
    import torch
    rng=np.random.default_rng(31)
    with tempfile.TemporaryDirectory(prefix='duet_cal_replay_') as name:
        root=Path(name);layers=[40,56];arrays=[];oracle=[]
        for step,(sid,K) in enumerate([(1,2),(2,1),(3,3)]):
            V=64
            p=rng.dirichlet(np.ones(V),K+1).astype('float32')
            q=rng.dirichlet(np.ones(V),K+1).astype('float32');q[-1]=0
            y=q.argmax(-1);y[-1]=-1
            arrays.append(dict(p_T=p,p_D=q,p_E=np.repeat(p[:,None,:],len(layers),axis=1),y=y,
                               step=np.full(K+1,step),seq_id=np.full(K+1,sid)))
            if sid==1:continue
            py=p[np.arange(K),y[:K]];qy=q[np.arange(K),y[:K]]
            h=hz(torch.from_numpy(np.minimum(py/qy,1))[None])[0].numpy()
            a=np.maximum(p-q,0);a[-1]=p[-1];a/=a.sum(-1,keepdims=True)
            truth=(h[:,None]*a).reshape(-1)
            oracle.append({B:np.sort(truth)[-B:].sum() for B in [8,15,24]})
        np.savez(root/'chunk.npz',**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
        manifest=root/'data.json';manifest.write_text(json.dumps(dict(layers=layers,chunks=[dict(file='chunk.npz')])))
        result=evaluate(manifest,2)
        assert result['counts']==[1,1]
        for B in [8,15,24]:
            actual=result['prompt_coverage'][f'L40|B{B}|residual|full|w0']
            assert np.allclose(actual,[x[B] for x in oracle],atol=1e-6)


def main():
    plan=json.loads((HERE/'plan.json').read_text());frozen=json.loads((HERE/'frozen.json').read_text())
    environment=HERE/'environment.json'
    if environment.exists():
        for name,digest in json.loads(environment.read_text())['preexisting_modified_production'].items():
            assert hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest()==digest
    for name,h in frozen['hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen input changed: '+name)
    all_ids=[]
    for value in plan['datasets'].values():
        assert hashlib.sha256(Path(value['file']).read_bytes()).hexdigest()==value['sha256']
        all_ids += [m['token_sha256'] for m in value['mapping']]
    if (HERE/'refinement_plan.json').exists():
        value=json.loads((HERE/'refinement_plan.json').read_text())['dataset']
        assert hashlib.sha256(Path(value['file']).read_bytes()).hexdigest()==value['sha256']
        all_ids += [m['token_sha256'] for m in value['mapping']]
    assert len(all_ids)==len(set(all_ids))
    runs=[]
    for p in (HERE/'runs').glob('*/*/complete.json'):
        r=load_run(p.parent);metrics(r)
        assert hashlib.sha256((p.parent/'result.json').read_bytes()).hexdigest()==r['meta']['result_sha256']
        for rec in r['records']:
            assert rec['output_tokens']==int(r['meta']['command'][r['meta']['command'].index('--output_len')+1])
            assert len(rec['metrics']['phase_events'])==len(rec['metrics']['target_verify_times'])
        runs.append(r)
    adapter=ChainAdapter();assert adapter.prediction_supported(config())
    assert not adapter.prediction_supported(config(fanout=1))
    assert not adapter.prediction_supported(config(budget=24))
    assert not adapter.prediction_supported(config(exit=79))
    assert not adapter.prediction_supported(config(k1=4,k2=1))
    tree=CurrentTreeAdapter()
    a=tree.shape(config(mode='p2tree',n2=8,v2=8),8)
    b=tree.shape(config(mode='p2tree',n2=12,v2=8),8)
    assert a.phase2_verify_queries==b.phase2_verify_queries==9
    assert a.phase2_generated_per_root!=b.phase2_generated_per_root
    assert not tree.prediction_supported(config(mode='p2tree'))
    # Equal cycle times; hit*AL ranks A above B, but total emitted tokens ranks B above A.
    assert .8*2 > .5*2
    assert 1+.8*2+.2*0 < 1+.5*2+.5*2
    rng=np.random.default_rng(7)
    for _ in range(1000):
        U,T,dU,dT=rng.uniform(.01,10,4)
        assert ((U+dU)/(T+dT)>U/T)==(dU/dT>U/T)
    check_replay()
    out=dict(status='passed',complete_runs=len(runs),distinct_prompts=len(all_ids),
             checks=['immutable model/plan/adapter benchmark source','dataset disjointness','actual output counts',
                     'phase reward accounting identity','metric alignment','config domain rejection',
                     'generated/verified tree node separation','miss reward counterexample','marginal goodput identity',
                     'variable-length replay warmup exclusion and exact-target residual oracle'])
    (HERE/'checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
