"""Check actual model outputs before interpreting the greedy host prototype."""
import json,statistics as st
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'MODEL_RESULTS.json').read_text());lookup={r['name']:r for r in rows};pairs=[]
for model in ('llama2','llama3'):
 for b in (1,8):
  for kind in ('profile','full'):
   base=lookup.get(f'{model}_greedy_b{b}_host0_{kind}');opt=lookup.get(f'{model}_greedy_b{b}_host1_{kind}')
   if not base or not opt:continue
   assert base['env']['CUDA_VISIBLE_DEVICES']==opt['env']['CUDA_VISIBLE_DEVICES']
   a=json.loads((HERE/base['path']).read_text());z=json.loads((HERE/opt['path']).read_text())
   assert len(a['cells'])==len(z['cells'])==2
   assert a['prompt_ids']==z['prompt_ids']
   assert {k:v for k,v in a['args'].items() if k!='output'}=={k:v for k,v in z['args'].items() if k!='output'}
   assert all(len(c['outputs'])==len(d['outputs'])==len(a['prompt_ids']) for c,d in zip(a['cells'],z['cells']))
   tokens_equal=[all(x['token_ids']==y['token_ids'] for x,y in zip(c['outputs'],d['outputs'])) for c,d in zip(a['cells'],z['cells'])]
   same_metrics=all(base['performance'][k]==opt['performance'][k] for k in ['al','cache_hit','emitted_tokens','included_steps','excluded_steps'])
   get=lambda r:r['steady_profile']['verify_ms']['mean'] if kind=='profile' else r['verify_ms']['mean']
   pairs.append(dict(model=model,batch=b,kind=kind,all_output_tokens_equal=all(tokens_equal),
      both_passes_equal=tokens_equal,same_population_and_al_hit=same_metrics,
      tps_change_percent=100*(opt['performance']['tps']/base['performance']['tps']-1),
      verify_ms=[get(base),get(opt)],verify_change_percent=100*(get(opt)/get(base)-1),
      base=base['path'],optimized=opt['path']))
probe=json.loads((HERE/'greedy_host_micro.json').read_text());micro=[]
for b in (1,2,4,8):
 cs=[c for c in probe['cases'] if c['batch']==b]
 micro.append(dict(batch=b,mean_case_medians_ms={k:st.mean(c['median_ms'][k] for c in cs) for k in cs[0]['median_ms']}))
(HERE/'GREEDY_COMPARISONS.json').write_text(json.dumps(dict(scope='benchmark adapter only; production default unchanged',micro=micro,model_pairs=pairs),indent=2))
print('Greedy full-model pairs',len(pairs),'both-pass output matches',sum(x['all_output_tokens_equal'] for x in pairs))
