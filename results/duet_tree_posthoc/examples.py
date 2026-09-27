"""Recover concrete, high-probability mistakes without running another model."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from calibrate import frozen_predict, from_edges

HERE=Path(__file__).resolve().parent
FULL=HERE.parent/'duet_tree_al_full'


def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/chokwans99/awq_calibrated/layerskip_llama2_70b',local_files_only=True)
    questions={q['question_id']:q for q in json.loads((FULL/'questions.json').read_text())}
    frozen=json.loads((HERE.parent/'duet_tree_analysis/calibration_frozen.json').read_text())
    candidates=defaultdict(list); collisions=defaultdict(dict); hist=defaultdict(lambda:np.zeros(7))
    for policy in ['q_path','phase_sibling_q_bin']:
        folder=HERE/'runs'/policy
        records=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
        contexts={};history={};recordmap={}
        for r in records:
            current=tokenizer.encode(questions[r['question_id']]['turns'][r['turn']],add_special_tokens=False)
            prompt=current if r['turn']==0 else history[r['question_id']]+tokenizer.encode('\n\n',add_special_tokens=False)+current
            contexts[r['uid']]=prompt;recordmap[r['uid']]=r
            history[r['question_id']]=prompt+r['output_ids']
        for line in (folder/'trees.jsonl').open():
            t=json.loads(line)
            if t['is_final_event']:continue
            fa=frozen_predict(t,frozen);fr,_=from_edges(t,fa)
            for j in range(t['n']):
                a=t['alpha'][j];q=t['raw'][j];attempt=t['attempt'][j]
                bi=int(np.searchsorted([1e-12,.01,.1,.5,.9,1-1e-12],a,side='right'))
                hist[(t['phase'],t['sib'][j])][bi]+=attempt
                if t['sib'][j]!=0 or attempt<.3:continue
                label=None
                if q>=.9 and a<.1:label='confident_draft_wrong'
                if q<.1 and a>.95:label='unlikely_draft_accepted'
                low=a<.05;high=a>.95
                key=(t['phase'],t['depth'][j],int(q/.005),int(t['q_entropy'][j]/.05),int(t['q_max'][j]/.005))
                need_collision=(low and 'low' not in collisions[key]) or (high and 'high' not in collisions[key])
                if label is None and not need_collision:continue
                r=recordmap[t['uid']];offset=sum(e['accepted_len'] for e in r['metrics']['phase_events'][:t['event_index']])
                common=contexts[t['uid']]+r['output_ids'][:offset+1]
                ancestors=[];parent=t['par'][j]
                while parent>=0:
                    ancestors.append(t['tok'][parent]);parent=t['par'][parent]
                context=common+ancestors[::-1]
                example=dict(policy=policy,uid=t['uid'],serial=t['serial'],node=j,phase=t['phase'],depth=t['depth'][j],
                    token_id=t['tok'][j],token=tokenizer.decode([t['tok'][j]]),context_tail=tokenizer.decode(context[-70:]),
                    q=q,p=t['p_token'][j],q_entropy=t['q_entropy'][j],q_max=t['q_max'][j],alpha=a,
                    attempt=attempt,reach=t['reach'][j],q_path=t['q_path'][j],frozen_reach=float(fr[j]),
                    frozen_alpha=float(fa[j]),p_rank=t['p_rank'][j])
                if label:candidates[label].append(example)
                if need_collision:collisions[key]['low' if low else 'high']=example
    result={}
    for label,values in candidates.items():
        values.sort(key=lambda r:abs(r['q_path']-r['reach']),reverse=True)
        result[label]=dict(eligible_nodes=len(values),examples=values[:5])
    pairs=[dict(feature_bin=list(k),**v) for k,v in collisions.items() if len(v)==2]
    result['feature_collisions']=dict(count=len(pairs),examples=pairs[:5],
        caveat='Same coarse observable-feature bins, not a proof of irreducibility with context or token identity.')
    result['attempt_weighted_alpha_histogram']=[dict(phase=p,sibling=s,mass=v.tolist(),fraction=(v/v.sum()).tolist())
        for (p,s),v in sorted(hist.items())]
    result['histogram_edges']=[0,1e-12,.01,.1,.5,.9,1-1e-12,1.00001]
    (HERE/'examples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v['eligible_nodes'] for k,v in result.items() if isinstance(v,dict) and 'eligible_nodes' in v}))


if __name__=='__main__':main()
