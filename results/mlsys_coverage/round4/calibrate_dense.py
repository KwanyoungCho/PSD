"""Freeze dense-model tables from eight prespecified calibration questions."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from calibration_math import exact_ladder,groups,fit,third_gain
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('model',choices=['llama2','llama3']);args=p.parse_args()
folder=HERE/'dense_calibration'/args.model
report=json.loads((HERE/'dense_calibration_runs'/(args.model+'_calibration.json')).read_text())
if report['status']!='complete':raise ValueError('Incomplete calibration collection')
manifest=json.loads((folder/'manifest.json').read_text());events=report['cells'][0]['metrics']['phase_events']
ids=sorted({e['seq_id'] for e in events});lookup={sid:report['question_indexes'][i] for i,sid in enumerate(ids)}
step_question={e['step_id']:lookup[e['seq_id']] for e in events}
trees=[];contexts=[];start=time.time()
for meta in manifest:
 path=folder/meta['file']
 with np.load(path) as z:
  par=z['par'];sib=z['sib'];tok=z['tok'];p=z['p'].astype(float);q=z['q'].astype(float)
  p/=p.sum(1,keepdims=True);q/=q.sum(1,keepdims=True)
  calc=exact_ladder(par,sib,tok,p,q)
  trees.append(dict(phase=meta['phase'],prompt=step_question[meta['step_id']],sib=sib,
                    raw=q[np.arange(len(tok)),tok],calc={k:v for k,v in calc.items() if k!="residual"}))
  for parent,kids in groups(par,sib).items():
   if parent<0 or calc['reach'][parent]<=1e-12:continue
   g=third_gain(p[parent+1],q[kids[0]],head=32,samples=256,seed=719000+len(contexts))
   contexts.append(dict(file=meta['file'],parent=parent,phase=meta['phase'],weight=calc['reach'][parent],prompt=step_question[meta['step_id']],**g))
def serial(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 raise TypeError(type(value))
(folder/'compact_audit.json').write_text(json.dumps(dict(trees=trees,contexts=contexts),default=serial,indent=2))
print('snapshots',len(trees),'contexts',len(contexts),flush=True)
reach=fit(trees,[0,.01,.03,.1,.3,.6,.9,1.00001],5)
curves={};integ={}
for phase in [1,2]:
 rows=[c for c in contexts if c['phase']==phase]
 if not rows:raise ValueError(f'No phase{phase} calibration contexts')
 w=np.array([c['weight'] for c in rows]);w/=w.sum();raw=w@np.array([c['g'] for c in rows])
 curves[str(phase)]=np.clip(np.maximum.accumulate(raw),0,1).tolist()
 integ[str(phase)]=dict(contexts=len(rows),raw=raw.tolist(),se=float(np.sqrt(np.sum((w*np.array([c['se'] for c in rows]))**2))),
                      deterministic_bound=(w@np.array([c['bound'] for c in rows])).tolist())
common=dict(model=args.model,training_question_indexes=report['question_indexes'],question_sha256=report['source_sha256'],
            snapshots=len(trees),neural_training=False,validation_used_for_fitting=False,
            snapshot_sha256={m['file']:hashlib.sha256((folder/m['file']).read_bytes()).hexdigest() for m in manifest})
reach.update(common);gain=common|dict(curves=curves,numerical_integration=integ,wall_s=time.time()-start)
for name,data in [('reach',reach),('gain',gain)]:
 dest=folder/(name+'.json')
 if dest.exists():
  old=json.loads(dest.read_text());key='means' if name=='reach' else 'curves'
  if old[key]!=data[key] or old['snapshot_sha256']!=data['snapshot_sha256']:
   raise ValueError('Frozen calibration changed; refusing overwrite')
 else:dest.write_text(json.dumps(data,indent=2))
print(json.dumps(dict(curves=curves,integration=integ,wall_s=time.time()-start),indent=2))
