"""Pair fixed-setting arms within process repetition; expose both gains and losses."""
import json,statistics as st
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'MODEL_RESULTS.json').read_text());lookup={r['name']:r for r in rows};out=[]
for model in ('llama2','llama3'):
 for b in (1,8):
  for rep in (0,1):
   base=lookup.get(f'{model}_b{b}_graph0_full_r{rep}')
   if not base:continue
   for arm in ('graph1','trim1'):
    opt=lookup.get(f'{model}_b{b}_{arm}_full_r{rep}')
    if not opt:continue
    a={k:v for k,v in base['args'].items() if k!='output'};z={k:v for k,v in opt['args'].items() if k!='output'}
    assert a==z,(base['name'],opt['name'])
    assert base['env']['CUDA_VISIBLE_DEVICES']==opt['env']['CUDA_VISIBLE_DEVICES']
    p=base['performance'];q=opt['performance']
    out.append(dict(model=model,batch=b,repeat=rep,arm=arm,
        tps_change_percent=100*(q['tps']/p['tps']-1),
        verify_change_percent=100*(opt['verify_ms']['mean']/base['verify_ms']['mean']-1),
        al_change=q['al']-p['al'],hit_change_pp=100*(q['cache_hit']-p['cache_hit']),
        base=base['name'],optimized=opt['name'],
        base_excluded=p['exclusion_reasons'],optimized_excluded=q['exclusion_reasons']))
trim_controls=[]
for model in ('llama2','llama3'):
 for b in (1,8):
  for rep in (0,1):
   base=lookup.get(f'{model}_b{b}_graph1_full_r{rep}');opt=lookup.get(f'{model}_b{b}_trim1_full_r{rep}')
   if not base or not opt:continue
   assert base['env']['CUDA_VISIBLE_DEVICES']==opt['env']['CUDA_VISIBLE_DEVICES']
   trim_controls.append(dict(model=model,batch=b,repeat=rep,
      all_output_tokens_equal=base['output_sha256']==opt['output_sha256'],
      tps_change_percent=100*(opt['performance']['tps']/base['performance']['tps']-1),
      verify_change_percent=100*(opt['verify_ms']['mean']/base['verify_ms']['mean']-1)))
summary=[]
for model in ('llama2','llama3'):
 for b in (1,8):
  for arm in ('graph1','trim1'):
   cs=[r for r in out if (r['model'],r['batch'],r['arm'])==(model,b,arm)]
   if cs:summary.append(dict(model=model,batch=b,arm=arm,repetitions=len(cs),
       tps_change_percent_mean=st.mean(x['tps_change_percent'] for x in cs),
       verify_change_percent_mean=st.mean(x['verify_change_percent'] for x in cs),
       tps_change_percent_range=[min(x['tps_change_percent'] for x in cs),max(x['tps_change_percent'] for x in cs)]))
(HERE/'COMPARISONS.json').write_text(json.dumps(dict(pairs=out,summary=summary,trim_controls=trim_controls),indent=2))
for r in summary:print(r)
