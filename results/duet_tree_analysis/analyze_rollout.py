"""Actual construction ablation, with synchronized observers: AL, not TPS."""
import json
import numpy as np
from core import HERE
from analyze_probe import load_trees


def summary(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    if len(manifest['records'])!=len(manifest['plan']['mapping']):raise ValueError('Incomplete rollout')
    trees=load_trees(manifest,folder)
    rows=[]
    for record in manifest['records']:
        p=record['prompt'];events=record['metrics']['phase_events'];hits=[e for e in events if e['cache_hit']]
        observed=[t for t in trees if t['prompt']==p]
        row=dict(prompt=p,steps=len(events),output_tokens=record['output_tokens'],
            hit_rate=len(hits)/len(events),hit_al=float(np.mean([e['accepted_spec_len'] for e in hits])),
            emitted_per_step=record['output_tokens']/len(events),
            snapshot_true_al=float(np.mean([t['calc']['al'] for t in observed])),
            hit_nodes=float(np.mean([e['valid_k'] for e in hits])))
        for phase in [1,2]:
            selected=[e for e in hits if e['source']==phase]
            row[f'p{phase}_hit_al']=float(np.mean([e['accepted_spec_len'] for e in selected])) if selected else None
        rows.append(row)
    fields=[k for k in rows[0] if k!='prompt']
    return dict(prompts=len(rows),snapshots=len(trees),tree_hits=sum(r['tree_hits'] for r in manifest['records']),
                prompt_balanced={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in fields},rows=rows)


def main():
    a=summary(HERE/'rollout_q_path');b=summary(HERE/'rollout_phase_sibling_q_bin')
    rng=np.random.default_rng(923);paired={}
    for field in ['hit_al','p1_hit_al','p2_hit_al','hit_rate','emitted_per_step','steps','snapshot_true_al','hit_nodes']:
        differences=np.array([y[field]-x[field] for x,y in zip(a['rows'],b['rows']) if x[field] is not None and y[field] is not None])
        boot=differences[rng.integers(0,len(differences),(2000,len(differences)))].mean(1)
        paired[field]=dict(mean_delta=float(differences.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                            positive_prompts=int((differences>1e-10).sum()),prompts=len(differences))
    out=dict(scope='One seed, 16 reused held-out prompts, changed trajectories, observer-instrumented AL only; no TPS claim.',
             baseline=a,calibrated=b,paired=paired)
    (HERE/'rollout_analysis.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(dict(baseline=a['prompt_balanced'],calibrated=b['prompt_balanced'],paired=paired),indent=2))


if __name__=='__main__':main()
