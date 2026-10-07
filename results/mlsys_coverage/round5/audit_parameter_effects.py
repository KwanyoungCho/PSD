"""Distinguish active interventions from repeated identical algorithms."""
import json
from make_plans import HERE,save
from result_metrics import read


def main():
    output=[]
    for model in ('llama2','llama3'):
        for batch in (1,8):
            decision=read(HERE/f'{model}_b{batch}_tree_decision.json')
            anchor=decision['profile_efficient_anchor']
            bases=[read(p) for p in HERE.glob(f'*/{anchor}.json')]
            base=next(r for r in bases if r.get('status')=='complete')['cells'][-1]
            for change in ('beta0','beta1','no_floors'):
                name=f'{model}_b{batch}_tree_{change}'
                row=read(HERE/f'{model}_b{batch}_tree/{name}.json')['cells'][-1]
                same_tokens=[o['token_ids'] for o in base['outputs']]==[o['token_ids'] for o in row['outputs']]
                keys=('accepted_len','valid_k','source','cache_hit','emitted_len')
                events=lambda c:[tuple(e[k] for k in keys) for e in c['metrics']['phase_events']]
                same_events=events(base)==events(row)
                if change.startswith('beta') and not(same_tokens and same_events):
                    raise ValueError('Inactive beta changed algorithm behavior')
                output.append(dict(model=model,batch=batch,anchor=anchor,job=name,
                    same_tokens=same_tokens,same_acceptance_events=same_events,
                    anchor_al=base['summary']['boundary_excluded_al'],
                    intervention_al=row['summary']['boundary_excluded_al'],
                    mechanism='tree_beta allocator is bypassed by dynamic global selection' if change.startswith('beta') else
                        'P2 proxy/confidence pruning floors changed to zero together'))
    save('PARAMETER_EFFECTS.json',output)
    print('Checked',len(output),'interventions; beta has no effect in dynamic mode')

if __name__=='__main__':main()
