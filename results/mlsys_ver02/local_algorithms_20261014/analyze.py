"""Paired prompt-group comparisons; do not select parameters on full results."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'ssd/bench'))
from mlsys_tree_results import compare,summarize
from mlsys_measurement import base_eligible
from mlsys_request_reporting import common_question_summary

PAIRS=[('base','root_only'),('base','tree_only'),('base','combined'),
       ('root_only','combined'),('tree_only','combined'),
       ('base','root_norm'),('root_norm','root_position'),
       ('root_position','root_only'),('root_proxy','root_only')]

def clusters(report):
 b=report['args']['batch'];out={}
 for group in report['groups']:
  if not group['phase'].startswith('measured'):continue
  row=out.setdefault(group['index'],np.zeros(9))
  for s in group['steps']:
   if not base_eligible(s,b) or s.get('graph_preparation',False):continue
   events=s['events'];row[:3]+=[sum(e['accepted_len'] for e in events),sum(e['cache_hit'] for e in events),len(events)]
   for phase in (1,2):
    ev=[e for e in events if e['source']==phase]
    row[3+(phase-1)*3:6+(phase-1)*3]+=[sum(e['accepted_len'] for e in ev),len(ev),len(events)]
 return out

def metrics(a):
 x=a.sum(0)
 return np.array([x[0]/x[2],100*x[1]/x[2],x[3]/x[4] if x[4] else np.nan,
                  x[6]/x[7] if x[7] else np.nan,100*x[4]/x[2],100*x[7]/x[2]])

def paired(a,b):
 ga,gb=clusters(a),clusters(b);keys=sorted(ga.keys()&gb.keys())
 aa=np.array([ga[k] for k in keys]);bb=np.array([gb[k] for k in keys])
 rng=np.random.default_rng(426);draws=[]
 for _ in range(5000):
  i=rng.integers(len(keys),size=len(keys));draws.append(metrics(bb[i])-metrics(aa[i]))
 delta=metrics(bb)-metrics(aa);ci=np.nanquantile(draws,[.025,.975],axis=0)
 return dict(clusters=len(keys),**{name:dict(delta=float(delta[i]),ci95=ci[:,i].tolist())
  for i,name in enumerate(('al','hit_pp','p1_hit_al','p2_hit_al','p1_hit_pp','p2_hit_pp'))})

def source_decomposition(a,b):
 """Exact symmetric accounting identity, not a matched-context causal estimate."""
 x,y=summarize(a)['source'],summarize(b)['source']
 nx=sum(v['requests'] for v in x.values());ny=sum(v['requests'] for v in y.values())
 if not nx or not ny or any(x[k]['al'] is None or y[k]['al'] is None for k in x):return None
 out={}
 for k in x:
  wx,wy=x[k]['requests']/nx,y[k]['requests']/ny;ax,ay=x[k]['al'],y[k]['al']
  out[k]=dict(share_before=wx,share_after=wy,al_before=ax,al_after=ay,
      mix_contribution=(wy-wx)*(ax+ay)/2,conditional_contribution=(wx+wy)*(ay-ax)/2)
 assert np.isclose(sum(v['mix_contribution']+v['conditional_contribution'] for v in out.values()),
                   b['summary']['al']-a['summary']['al'])
 return out

def shape_summary(report):
 b=report['args']['batch'];es=[e for s in report['steps'] if base_eligible(s,b) and not s.get('graph_preparation') for e in s['events']]
 fields=('valid_k','tree_max_depth','tree_sibling_width','node_utilization','reached_depth_ceiling')
 out={}
 for phase in (0,1,2):
  rows=[e for e in es if e['source']==phase]
  out[str(phase)]={k:float(np.mean([e[k] for e in rows if e.get(k) is not None])) if any(e.get(k) is not None for e in rows) else None for k in fields}
 return out

def main():
 result=dict(status='partial',baseline='18e912f',candidate='992b8b8',models={},comparisons={},exclusions=[],common_requests={})
 for model in ('l2','l3'):
  folder=Path('/data/chokwans99')/(f'ver02-local-algorithms-full-{model}-20261014' if model=='l3' else 'ver02-local-algorithms-full-l2-final-20261014')
  if not (folder/'campaign.json').exists():continue
  jobs=[(folder,j) for j in json.loads((folder/'campaign.json').read_text())];raw={}
  retry=Path('/data/chokwans99/ver02-local-algorithms-l3-recheck3-20261014')
  if model=='l3' and (retry/'campaign.json').exists():
   jobs += [(retry,j) for j in json.loads((retry/'campaign.json').read_text())]
  for source_folder,j in jobs:
   if j['status']!='complete' or j['returncode'] or j['external_pids']:
    result['exclusions'].append(dict(model=model,job=j));continue
   p=source_folder/(j['name']+'.json');r=json.loads(p.read_text())
   if r['status']!='complete':continue
   assert len(r['outputs'])==960
   assert len(r['question_indexes'])==480
   raw[j['name']]=r
  result['models'][model]={k:dict(**summarize(r),shape=shape_summary(r)) for k,r in raw.items()}
  if raw:
   result['common_requests'][model]=common_question_summary({k:r['request_records'] for k,r in raw.items()})
  result['comparisons'][model]={}
  for before,after in PAIRS:
   if before in raw and after in raw:
    a,b=raw[before],raw[after]
    assert a['prompt_ids']==b['prompt_ids']
    result['comparisons'][model][before+' -> '+after]=dict(quality=paired(a,b),timing=compare(a,b),
                                                        al_accounting=source_decomposition(a,b))
 result['status']='complete' if len(result['models'])==2 and all(len(v)==7 for v in result['models'].values()) else 'partial'
 (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 print(result['status'])
 for model,arms in result['models'].items():
  for name,a in arms.items():print(model,name,a['summary'])
if __name__=='__main__':main()
