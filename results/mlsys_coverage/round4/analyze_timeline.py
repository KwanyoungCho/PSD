"""Join target/draft CUDA events per evaluation step, excluding graph captures."""
from pathlib import Path
import json, collections
import numpy as np
HERE=Path(__file__).resolve().parent

def stats(x):
 return None if not x else dict(n=len(x),mean=float(np.mean(x)),p50=float(np.median(x)),p90=float(np.quantile(x,.9)))

def events(path,marker):
 rows=json.loads(path.read_text())
 starts=[i for i,e in enumerate(rows) if e['label']==marker and e.get('step_id')==1]
 if starts:rows=rows[starts[-1]:]
 d=collections.defaultdict(dict)
 for e in rows:
  if e.get('step_id') is not None:d[e['step_id']][e['label']]=e
 return d

def end(e):return e['wall_end_ns']/1e6

def start(e):return e['wall_start_ns']/1e6

out=[]
for path in sorted((HERE/'profiles').glob('*')):
 ts=list(path.glob('duet_profile_target_rank0*json')); ds=list(path.glob('duet_profile_draft*json'))
 report=HERE/'profile_runs'/(path.name+'.json')
 if not report.exists():report=HERE/'optimization_profile_runs'/(path.name+'.json')
 if not report.exists():report=HERE/'fused_profile_runs'/(path.name+'.json')
 if not(ts and ds and report.exists()):continue
 result=json.loads(report.read_text())
 if result['status']!='complete':continue
 cell=result['cells'][-1];by_step=collections.defaultdict(list)
 for e in cell['metrics']['phase_events']:by_step[e['step_id']].append(e)
 t=events(ts[-1],'target_send_request');d=events(ds[-1],'batch_miss_draft')
 steps=[];excluded=collections.Counter()
 for step,evs in by_step.items():
  te=t.get(step,{});de=d.get(step,{})
  required_d=['batch_p1_total','batch_p2_total','batch_proxy_receive','batch_p1_replay','batch_p2_replay','batch_glue','batch_miss_draft']
  required_t=['batch_target_pre','batch_target_post','batch_target_final_logits','target_postprocess','target_spec_wait']
  if any(k not in de for k in required_d) or any(k not in te for k in required_t):excluded['missing']+=1;continue
  if any('capture' in k for k in de):excluded['capture']+=1;continue
  if step<20:excluded['warmup_step_lt20']+=1;continue
  p1=end(de['batch_p1_total']);proxy=end(de['batch_proxy_receive']);p2=end(de['batch_p2_total']);model=end(te['batch_target_final_logits']);ready=end(te['target_postprocess'])
  steps.append(dict(step=step,batch=len(evs),all_hit=all(e['cache_hit'] for e in evs),
    hit_rate=np.mean([e['cache_hit'] for e in evs]),
    p1_before_proxy_ms=proxy-p1,p2_before_model_ms=model-p2,p2_before_ready_ms=ready-p2,
    p1_ms=de['batch_p1_total']['cuda_ms'],p2_ms=de['batch_p2_total']['cuda_ms'],
    p1_replay_ms=de['batch_p1_replay']['cuda_ms'],p2_replay_ms=de['batch_p2_replay']['cuda_ms'],
    glue_ms=de['batch_glue']['cuda_ms'],miss_ms=de['batch_miss_draft']['cuda_ms'],
    target_pre_ms=te['batch_target_pre']['cuda_ms'],target_post_ms=te['batch_target_post']['cuda_ms'],
    target_wait_ms=te['target_spec_wait']['cuda_ms'],
    timeline={label:dict(start_ms=start(e)-start(te['batch_target_pre']),end_ms=end(e)-start(te['batch_target_pre'])) for label,e in (list(te.items())+list(de.items())) if label in required_t+required_d}))
 groups={}
 for label,ss in [('all',steps),('full_batch',[s for s in steps if s['batch']==cell['batch']]),('full_batch_all_hit',[s for s in steps if s['batch']==cell['batch'] and s['all_hit']]),('all_hit',[s for s in steps if s['all_hit']]),('some_miss',[s for s in steps if not s['all_hit']])]:
  groups[label]=dict(n=len(ss),p1_hidden_rate=np.mean([s['p1_before_proxy_ms']>=0 for s in ss]) if ss else None,
    p2_before_model_rate=np.mean([s['p2_before_model_ms']>=0 for s in ss]) if ss else None,
    p2_hidden_rate=np.mean([s['p2_before_ready_ms']>=0 for s in ss]) if ss else None,
    metrics={k:stats([s[k] for s in ss]) for k in steps[0] if k.endswith('_ms')} if steps else {})
 out.append(dict(name=path.name,args=result['args'],groups=groups,excluded=dict(excluded),steps=steps))
(HERE/'timeline.json').write_text(json.dumps(out,indent=2))
for r in out:
 g=r['groups']['full_batch'];print(r['name'],g['n'],{k:round(v,3) if v is not None else None for k,v in g.items() if k.endswith('rate')}, {k:round(v['p50'],2) for k,v in g.get('metrics',{}).items() if v is not None and k in ['p1_before_proxy_ms','p2_before_ready_ms','p1_ms','p2_ms','target_pre_ms','target_post_ms']})
