"""AL-only 2x2 analysis, aligning all four arms to target seeds2026/2027.

The combined arm also enables exact implementation optimizations. Thus this
is NOT a factorial TPS experiment. AL interpretation uses their separately
validated proposal/topology equivalence, and is explicitly scoped as such.
"""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent

def load(path):
 d=json.loads(path.read_text())
 if d['status']!='complete':return None
 out=np.zeros((480,2))
 cells=[c for c in d['cells'] if c['seed'] in (2026,2027)]
 if len(cells)!=2:raise ValueError('Two aligned target seeds required')
 for c in cells:
  es=c['metrics']['phase_events'];ids=sorted({e['seq_id'] for e in es});lookup={s:i for i,s in enumerate(ids)}
  for e in es:
   if e['output_cap_reached'] or e['clipped']:continue
   out[lookup[e['seq_id']]] += [e['accepted_len'],1]
 return out

rows=[]
for model in ['llama2','llama3']:
 paths=[HERE/(model+'_full')/'q_path.json',HERE/(model+'_full')/'reach_gain_frontier.json',HERE/(model+'_follow')/(model+'_miss_chain4.json'),HERE/(model+'_budget')/(model+'_combined_chain4.json')]
 if not all(p.exists() for p in paths):continue
 arrays=[load(p) for p in paths]
 if any(a is None for a in arrays):continue
 rng=np.random.default_rng(2841);sample=rng.integers(0,480,(2000,480))
 point=[a.sum(0)[0]/a.sum(0)[1] for a in arrays]
 draws=[a[sample].sum(1) for a in arrays];means=[d[:,0]/d[:,1] for d in draws]
 effects={}
 for name,coeff in [('tree_score_at_chain2',[-1,1,0,0]),('tree_score_at_chain4',[0,0,-1,1]),('miss_depth_at_q_path',[-1,0,1,0]),('miss_depth_at_reach_gain_frontier',[0,-1,0,1]),('interaction',[1,-1,-1,1])]:
  delta=sum(c*x for c,x in zip(coeff,point));draw=sum(c*x for c,x in zip(coeff,means))
  effects[name]=dict(delta_clean_al=delta,ci95=np.quantile(draw,[.025,.975]).tolist())
 rows.append(dict(model=model,seeds=[2026,2027],questions=480,
                  arm_order=['q_path_chain2','reach_gain_frontier_chain2','q_path_chain4','reach_gain_frontier_chain4_optimized'],
                  arm_clean_al=point,effects=effects))
(HERE/'AL_INTERACTIONS.json').write_text(json.dumps(dict(scope=__doc__,results=rows),indent=2))
for r in rows:print(r['model'],r['arm_clean_al'],r['effects'])
