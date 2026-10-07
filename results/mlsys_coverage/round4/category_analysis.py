"""Task-group heterogeneity; descriptive, unadjusted question bootstrap CIs."""
from pathlib import Path
import json,csv
import numpy as np
HERE=Path(__file__).resolve().parent
results=json.loads((HERE/'RESULTS.json').read_text());ix={(r['group'],r['name']):r for r in results}
questions=json.loads((HERE.parent/'questions.json').read_text());out=[]
for model in ['llama2','llama3']:
 group=model+'_full';base=ix[(group,'q_path')]
 for name in ['reach_gain_frontier','dense_reach','dense_reach_gain_frontier','miss_chain1','miss_chain4','miss_tree2x2']:
  row=ix.get((group,name))
  if not row:continue
  av=np.array(row['per_question']);bv=np.array(base['per_question'])
  for category in sorted({q['group'] for q in questions}):
   ids=[j for j,i in enumerate(row['question_indexes']) if questions[i]['group']==category]
   aa=av[ids];bb=bv[ids];a=aa.sum(0);b=bb.sum(0)
   sample=np.random.default_rng(4817).integers(0,len(ids),(2000,len(ids)))
   va=aa[sample].sum(1);vb=bb[sample].sum(1)
   delta=va[:,3]/va[:,4]-vb[:,3]/vb[:,4]
   ci=np.quantile(delta,[.025,.975])
   out.append(dict(model=model,policy=name,category=category,questions=len(ids),
      al_clean=a[3]/a[4],reference_al_clean=b[3]/b[4],
      relative_pct=100*((a[3]/a[4])/(b[3]/b[4])-1),delta_ci95=ci.tolist()))
(HERE/'CATEGORY_RESULTS.json').write_text(json.dumps(out,indent=2))
with (HERE/'CATEGORY_RESULTS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
lines=['# Task group별 결과','', 'B8 / 3 target-seed pass / task group각80질문. Boundary-excluded AL, 질문 paired bootstrap2,000회. 다중비교 보정 없이 세부 경향을 보는 사후 분석이며 개별 CI를 독립적인 확증으로 해석하지 않는다.','', '| Model | Policy | Task | AL* Δ% | ΔAL* CI95 |','|---|---|---|---:|---|']
for r in out:
 if 'reach' in r['policy']:
  lo,hi=r['delta_ci95'];lines.append(f'| {r["model"]} | {r["policy"]} | {r["category"]} | {r["relative_pct"]:+.2f}% | [{lo:+.3f}, {hi:+.3f}] |')
(HERE/'CATEGORY_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Group comparisons',len(out))

# Does dense-model calibration improve over the historical table itself?
# Remove calibration questions from both arms, rather than only comparing
# each policy independently against q-path.
calibration_comparison=[]
for model in ['llama2','llama3']:
 a=ix.get((model+'_full','dense_reach_gain_frontier'))
 b=ix.get((model+'_full','reach_gain_frontier'))
 if not a or not b:continue
 ids=[j for j,i in enumerate(a['question_indexes']) if i not in [0,60,120,180,240,300,360,420]]
 aa=np.array(a['per_question'])[ids];bb=np.array(b['per_question'])[ids]
 samples=np.random.default_rng(8041).integers(0,len(ids),(2000,len(ids)))
 av=aa[samples].sum(1);bv=bb[samples].sum(1);A=aa.sum(0);B=bb.sum(0)
 calibration_comparison.append(dict(model=model,questions=len(ids),
   delta_clean_al=A[3]/A[4]-B[3]/B[4],
   ci95=np.quantile(av[:,3]/av[:,4]-bv[:,3]/bv[:,4],[.025,.975]).tolist()))
(HERE/'DENSE_VS_OLD_CALIBRATION.json').write_text(json.dumps(calibration_comparison,indent=2))
