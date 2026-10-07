"""Audit every completed run's returned-token and boundary accounting."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
checked=[];issues=[]
expected={
 'llama2_full':10,'llama3_full':10,'llama2_b1':6,'llama3_b1':6,
 'llama2_follow':7,'llama3_follow':7,'llama2_b1_short':3,'llama3_b1_short':3,
 'llama2_budget':2,'llama3_budget':2,'llama2_replica':2,'llama3_replica':2,
 'llama2_budget_control':1,'llama3_budget_control':1}
for name,count in expected.items():
 p=HERE/name/'campaign.json'
 if not p.exists() or len(json.loads(p.read_text()))!=count:issues.append('Incomplete required stage: '+name)
for manifest in sorted(HERE.glob('*/campaign.json')):
 records=json.loads(manifest.read_text());plan=json.loads((manifest.parent/'plan.json').read_text())
 if len(records)!=len(plan):issues.append('Plan/campaign length mismatch: '+str(manifest.parent))
 for run in records:
  path=manifest.parent/(run['name']+'.json');d=json.loads(path.read_text())
  if run['status']!='complete' or run['returncode'] or d['status']!='complete':issues.append('Run failed: '+str(path))
  if run.get('external_pids'):issues.append('Other process on allocated GPU: '+str(path))
  for c in d['cells']:
   prefix=f'{path.parent.name}/{path.stem}:T{c["temperature"]}/seed{c["seed"]}/B{c["batch"]}'
   tokens=sum(len(o['token_ids']) for o in c['outputs']);m=c['metrics'];events=m['phase_events'];steps=m['decode_steps']
   tests={
    'all_questions_returned':len(c['outputs'])==len(d['question_indexes']),
    'summary_actual_tokens':tokens==c['summary']['output_tokens'],
    'decode_counter_actual_tokens':tokens==m['decode_total_tokens'],
    'sequence_event_tokens':tokens==sum(e['emitted_len'] for e in events),
    'batch_step_tokens':tokens==sum(s['emitted_tokens'] for s in steps),
    'step_times':abs(sum(s['seconds'] for s in steps)-m['decode_total_time'])<1e-6,
    'no_cap_overrun':all(len(o['token_ids'])<=cap for o,cap in zip(c['outputs'],d['output_caps'])),
    'clip_flags':all((e['accepted_len']>e['emitted_len'])==e['clipped'] for e in events),
    'positive_returns':all(0<=e['emitted_len']<=e['accepted_len'] for e in events),
    'one_seq_id_per_question':len({e['seq_id'] for e in events})==len(d['question_indexes']),
   }
   clean=[s for s in steps if not s['clipped'] and not s['output_cap_reached']]
   elapsed=sum(s['seconds'] for s in clean)
   measured=sum(s['emitted_tokens'] for s in clean)/elapsed if elapsed else None
   saved=c['summary']['boundary_excluded_step_tps']
   tests['boundary_step_tps']=(saved is None and measured is None) or (saved is not None and measured is not None and abs(saved-measured)<1e-6)
   bad=[k for k,v in tests.items() if not v]
   if bad:issues.append(prefix+': '+','.join(bad))
   checked.append(dict(cell=prefix,tokens=tokens,events=len(events),checks=len(tests),passed=not bad))
result=dict(cells=len(checked),tokens=sum(c['tokens'] for c in checked),checks=sum(c['checks'] for c in checked),issues=issues,details=checked)
(HERE/'ARTIFACT_VALIDATION.json').write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k!='details'})
if issues:raise SystemExit(1)

parity=[]
for model in ['llama2','llama3']:
 for stage,names in [('follow',['fused','fused_bulk','fused_bulk_parallel']),('replica',['fused_bulk'])]:
  base=json.loads((HERE/(model+'_'+stage)/(model+'_baseline.json')).read_text())
  for name in names:
   candidate=json.loads((HERE/(model+'_'+stage)/(model+'_'+name+'.json')).read_text())
   same=[sum(a['token_ids']==b['token_ids'] for a,b in zip(ac['outputs'],bc['outputs']))
         for ac,bc in zip(candidate['cells'],base['cells'])]
   if same != [480,480]:issues.append(f'Implementation output mismatch {model}/{stage}/{name}: {same}')
   parity.append(dict(model=model,stage=stage,arm=name,identical_questions=same))
 base=json.loads((HERE/(model+'_b1')/'q_path.json').read_text())
 candidate=json.loads((HERE/(model+'_b1_short')/(model+'_optimized_q_path.json')).read_text())
 same=sum(a['token_ids']==b['token_ids'] for a,b in zip(candidate['cells'][0]['outputs'],base['cells'][0]['outputs']))
 if same!=480:issues.append(f'B1 implementation output mismatch {model}: {same}')
 parity.append(dict(model=model,stage='b1',arm='fused_bulk',identical_questions=[same]))
(HERE/'OPTIMIZATION_OUTPUT_PARITY.json').write_text(json.dumps(parity,indent=2))
result['output_parity_cells']=sum(len(x['identical_questions']) for x in parity)
(HERE/'ARTIFACT_VALIDATION.json').write_text(json.dumps(result,indent=2))
print('Exact implementation output parity:',result['output_parity_cells'],'cells')
if issues:raise SystemExit(1)
