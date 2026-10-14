"""Separate phase observations from unprofiled primary measurements."""
import collections,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'ssd/bench'))
from mlsys_measurement import base_eligible
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'target_rank0':['batch_target_pre','batch_target_proxy','batch_target_post',
                      'batch_target_final_logits','target_postprocess','target_spec_wait','batch_tree_prepare','batch_tree_accept','batch_tree_commit'],
        'draft':['batch_miss_draft','batch_glue','batch_p1_total','batch_p1_replay',
                 'batch_proxy_receive','batch_p2_total','batch_p2_replay','batch_bulk_export','batch_kv_restore','batch_kv_stage']}

def tag_batch_epochs(rows,tag):
 # The legacy plotter uses draft_recv_request/draft_send_response anchors.
 # Batched serving has neither: batch_kv_restore closes once per real step.
 anchor='batch_kv_restore' if tag=='draft' else 'target_send_request'
 assert any(e['label']==anchor for e in rows),tag
 out=[];epoch=0;previous=None
 for row in rows:
  e=dict(row);sid=e.get('step_id')
  if e['label']==anchor and sid is not None:
   if previous is not None and sid<=previous:epoch+=1
   previous=sid
  e['_request_epoch']=epoch;out.append(e)
 return out

def main():
 folder=Path(sys.argv[1]);out={}
 for arm in ('base','root_only','tree_only','combined'):
  report=json.loads((folder/f'profile_{arm}.json').read_text());assert report['status']=='complete'
  allowed=set()
  for epoch,g in enumerate(report['warm_groups']+report['groups']):
   if not g['phase'].startswith('measured'):continue
   for s in g['steps']:
    if not base_eligible(s,report['args']['batch']) or s.get('graph_preparation'):continue
    ids={e['step_id'] for e in s['events']};assert len(ids)==1
    allowed.add((epoch,next(iter(ids))))
  processes={};by_step=collections.defaultdict(dict);coverage={}
  for tag,labels in LABELS.items():
   paths=list((folder/f'profile_{arm}_profile').glob(f'duet_profile_{tag}_*.json'))
   assert len(paths)==1,paths
   rows=tag_batch_epochs(json.loads(paths[0].read_text()),tag);values=collections.defaultdict(list)
   for e in rows:
    key=(e['_request_epoch'],e.get('step_id'))
    if key not in allowed or e['label'] not in labels:continue
    values[e['label']].append(e['cuda_ms']);by_step[key][e['label']]=e
   coverage[tag]=dict(raw_rows=len(rows),eligible_observed_steps=len({(e['_request_epoch'],e.get('step_id')) for e in rows if (e['_request_epoch'],e.get('step_id')) in allowed}),
                      recorded_max_epoch=max(e['_request_epoch'] for e in rows),
                      observed_capture_spans_in_selected_steps=sum('capture' in e['label'] and (e['_request_epoch'],e.get('step_id')) in allowed for e in rows))
   assert max(e['_request_epoch'] for e in rows)+1==len(report['warm_groups'])+len(report['groups']),(arm,tag,'epoch mismatch')
   assert all(len(values[k])==len(allowed) for k in labels),(arm,tag,'missing phase observations')
   processes[tag]={k:dict(n=len(v),mean_ms=float(np.mean(v)),p50_ms=float(np.median(v)),p90_ms=float(np.quantile(v,.9))) for k,v in values.items()}
  deadlines=[]
  for key,e in by_step.items():
   if all(k in e for k in ('batch_p1_total','batch_proxy_receive','batch_p2_total','batch_target_final_logits','target_postprocess')):
    end=lambda k:e[k]['wall_end_ns']/1e6
    deadlines.append(dict(epoch=key[0],step=key[1],p1_slack_ms=end('batch_proxy_receive')-end('batch_p1_total'),
                          p2_model_slack_ms=end('batch_target_final_logits')-end('batch_p2_total'),
                          p2_ready_slack_ms=end('target_postprocess')-end('batch_p2_total')))
  out[arm]=dict(eligible_steps=len(allowed),coverage=coverage,processes=processes,deadlines=deadlines,
       preparation_audit='Separate capture audit disabled in profile mode; selected-step capture labels are reported, not an independent preparation-free certification.',
       caveat='CUDA span times include launch gaps; online shapes differ by algorithm. Cross-GPU wall alignment is approximate.')
 (HERE/'PROFILE_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
 for ax,(tag,labels) in zip(axes,[('target_rank0',['batch_target_pre','batch_target_proxy','batch_target_post','batch_tree_accept']),('draft',['batch_miss_draft','batch_glue','batch_p1_total','batch_p2_total'])]):
  for j,arm in enumerate(out):
   data=out[arm]['processes'][tag]
   y=[data[k]['mean_ms'] if k in data else np.nan for k in labels]
   ax.bar(np.arange(len(labels))+(j-1.5)*.2,y,width=.19,label=arm)
  ax.set_xticks(range(len(labels)),[x.replace('batch_','').replace('target_','').replace('_total','') for x in labels],rotation=25)
  ax.set_ylabel('CUDA span ms / invocation');ax.set_title(tag);ax.legend(fontsize=8)
 fig.suptitle('Phase observations: Llama3-8B / Qwama, B8, 48 stratified prompts\nProfiled diagnostic run; overlapping/nested spans must not be added as serial latency')
 fig.savefig(HERE/'phases.png',dpi=180);fig.savefig(HERE/'phases.pdf');plt.close(fig)
 print({k:{tag:{n:round(v['mean_ms'],3) for n,v in d.items()} for tag,d in a['processes'].items()} for k,a in out.items()})

if __name__=='__main__':main()
