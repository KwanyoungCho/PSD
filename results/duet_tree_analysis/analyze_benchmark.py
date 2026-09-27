"""Aggregate fixed-policy, untraced confirmation; bootstrap by prompt cluster."""
import json
import numpy as np
from core import HERE


def read(seed,policy):
    folder=HERE/f'bench_{seed}_{policy}'
    completion=json.loads((folder/'completion.json').read_text())
    if completion['exit_code']:raise ValueError('Failed benchmark')
    x=json.loads((folder/'result.json').read_text())
    if len(x['records'])!=16:raise ValueError('Incomplete benchmark')
    rows=[]
    for r in x['records']:
        events=r['metrics']['phase_events'];hits=[e for e in events if e['cache_hit']]
        rows.append(dict(prompt=r['prompt'],seed=seed,wall_s=r['wall_s'],output_tokens=r['output_tokens'],
                         hit_al=float(np.mean([e['accepted_spec_len'] for e in hits])),
                         hit_rate=len(hits)/len(events),steps=len(events),
                         output_per_step=r['output_tokens']/len(events),
                         nodes=float(np.mean([e['valid_k'] for e in hits]))))
    return rows


def aggregate(rows):
    return dict(output_tokens=sum(r['output_tokens'] for r in rows),wall_s=sum(r['wall_s'] for r in rows),
        output_tps=sum(r['output_tokens'] for r in rows)/sum(r['wall_s'] for r in rows),
        **{k:float(np.mean([r[k] for r in rows])) for k in ['hit_al','hit_rate','steps','output_per_step','nodes']})


def main():
    policies=['q_path','phase_sibling_q_bin'];seeds=[921,922]
    raw={p:[r for seed in seeds for r in read(seed,p)] for p in policies}
    by_seed={str(seed):{p:aggregate([r for r in raw[p] if r['seed']==seed]) for p in policies} for seed in seeds}
    all_stats={p:aggregate(raw[p]) for p in policies}
    rng=np.random.default_rng(924);samples=rng.integers(0,16,(2000,16));paired={}
    for field in ['hit_al','hit_rate','output_per_step','steps','nodes']:
        delta=np.array([np.mean([r[field] for r in raw[policies[1]] if r['prompt']==i])-
                        np.mean([r[field] for r in raw[policies[0]] if r['prompt']==i]) for i in range(16)])
        paired[field]=dict(mean_delta=float(delta.mean()),ci95=np.quantile(delta[samples].mean(1),[.025,.975]).tolist())
    timing={p:np.array([sum(r['wall_s'] for r in raw[p] if r['prompt']==i) for i in range(16)]) for p in policies}
    ratios=timing[policies[0]][samples].sum(1)/timing[policies[1]][samples].sum(1)-1
    paired['output_tps_relative']=dict(point=all_stats[policies[1]]['output_tps']/all_stats[policies[0]]['output_tps']-1,
        ci95=np.quantile(ratios,[.025,.975]).tolist())
    result=dict(scope='No full-distribution observer; request generation wall includes prefill; 16 reused prompts x 2 seeds; no policy reselection.',
                by_seed=by_seed,aggregate=all_stats,paired=paired,records=raw)
    (HERE/'benchmark_analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
