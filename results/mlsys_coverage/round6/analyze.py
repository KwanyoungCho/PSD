"""Rebuild Round6 summaries; collect only completed, identified profile traces."""
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics as st

HERE=Path(__file__).resolve().parent

def load(p):return json.loads(p.read_text())
def save(n,x):
    temporary=(HERE/n).with_suffix('.tmp')
    temporary.write_text(json.dumps(x,indent=2));temporary.replace(HERE/n)
def stats(xs):
    return dict(n=len(xs),mean=st.mean(xs),median=st.median(xs),minimum=min(xs),maximum=max(xs)) if xs else None

def collect(report):
    log=report.with_suffix('.log').read_text()
    out=[]
    for proc in ('target_rank0','draft'):
        paths=re.findall(r'events -> (\S*duet_profile_'+proc+r'_\S+\.json)',log)
        if not paths:continue
        src=Path(paths[-1]);dst=HERE/'profiles'/report.stem/src.name
        if not dst.exists():
            if not src.exists():raise FileNotFoundError(src)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        elif src.exists() and hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dst.read_bytes()).digest():
            raise ValueError(f'Profile filename collision {src}; saved local copy retained')
        out.append(dst)
    return out

def grouped(path,marker):
    rows=load(path);starts=[i for i,e in enumerate(rows) if e['label']==marker and e.get('step_id')==1]
    if not starts:raise ValueError(f'Measured-pass boundary missing: {path} {marker}')
    rows=rows[starts[-1]:];d=collections.defaultdict(lambda:collections.defaultdict(list))
    for e in rows:
        if e.get('step_id') is not None:d[e['step_id']][e['label']].append(e)
    return d

def main():
    micros={}
    for kind,file in [('chain','chain_micro.json'),('tree','tree_micro_retry2.json')]:
        raw=load(HERE/file);assert raw['status']=='complete';groups=[]
        for v in (32000,128256):
          for t in (.7,0):
           for b in (1,2,4,8):
            cs=[c for c in raw['cases'] if (c['vocab'],c['temperature'],c['batch'])==(v,t,b)]
            groups.append(dict(vocab=v,temperature=t,batch=b,cases=len(cs),
                mean_of_case_medians_ms={k:st.mean(c['median_ms'][k] for c in cs) for k in cs[0]['median_ms']}))
        micros[kind]=dict(source=file,groups=groups,cases=raw['cases'])
    save('MICRO.json',micros)
    rows=[];failures=[]
    for p in sorted(HERE.glob('*/*.json')):
        if p.name in ('plan.json','campaign.json'):continue
        d=load(p)
        if not isinstance(d,dict) or 'cells' not in d:continue
        if d.get('status')!='complete':
            manifest=p.parent/'campaign.json'
            job=next((j for j in load(manifest) if j['name']==p.stem),{}) if manifest.exists() else {}
            failures.append(dict(path=str(p.relative_to(HERE)),status=job.get('status',d.get('status')),
                raw_status=d.get('status'),returncode=job.get('returncode'),error=d.get('error'),
                log=str(p.with_suffix('.log').relative_to(HERE))));continue
        cell=d['cells'][-1];m=cell['metrics'];perf=cell['summary']['performance'];keep=perf['included_step_indices']
        assert len(m['target_verify_times'])==len(m['decode_steps'])
        row=dict(name=p.stem,path=str(p.relative_to(HERE)),batch=cell['batch'],args=d['args'],env=d['env'],
            performance={k:v for k,v in perf.items() if k!='included_step_indices'},
            raw_decode_tps=cell['summary']['decode_tps'],verify_ms=stats([1000*m['target_verify_times'][i] for i in keep]),
            preemptions=m['preemptions'],kind='profile' if p.stem.endswith('profile') else 'full',
            prompt_count=len(d['prompt_ids']),warmup_seed=d['cells'][0]['seed'],measured_seed=cell['seed'],
            family=('greedy' if '_greedy_' in p.stem else 'chain' if d['args']['mode']=='duet-chain' else 'tree'),
            output_sha256=hashlib.sha256(json.dumps([o['token_ids'] for o in cell['outputs']]).encode()).hexdigest())
        row['step_ms_mean']=perf['seconds']*1000/perf['included_steps']
        row['nonverification_ms_mean']=row['step_ms_mean']-row['verify_ms']['mean']
        assert row['nonverification_ms_mean']>=0
        assert abs(perf['tps']-cell['batch']*perf['al']*1000/row['step_ms_mean'])<1e-8
        if row['kind']=='profile':
            paths=collect(p);row['profile_paths']=[str(x.relative_to(HERE)) for x in paths]
            if len(paths)!=2:raise ValueError(f'Incomplete profile {p}')
            tg=grouped(paths[0],'target_send_request')
            legacy='_legacy_' in p.stem or row['family']=='chain'
            dg=grouped(paths[1],'draft_recv_request' if legacy else 'batch_kv_restore')
            captures={sid for sid,ls in list(tg.items())+list(dg.items()) if any('capture' in k for k in ls)}
            excluded=captures|{sid+1 for sid in captures}
            measurements=collections.defaultdict(list);wallmeasure=collections.defaultdict(list);accepted=[]
            for i in keep:
                s=m['decode_steps'][i];ev=m['phase_events'][s['event_start']:s['event_end']];sid=ev[0]['step_id']
                if sid<20 or sid in excluded:continue
                assert all(e['step_id']==sid for e in ev)
                accepted.append(i)
                for k,vs in dict(tg.get(sid,{}),**dg.get(sid,{})).items():
                    measurements[k].append(sum(x['cuda_ms'] for x in vs))
                    wallmeasure[k].append(sum((x['cpu_dispatch_end_ns']-x['cpu_dispatch_start_ns'])/1e6 for x in vs))
            row['steady_profile']=dict(steps=len(accepted),verify_ms=stats([1000*m['target_verify_times'][i] for i in accepted]),
                gpu_span_ms={k:stats(v) for k,v in sorted(measurements.items())},
                host_dispatch_ms={k:stats(v) for k,v in sorted(wallmeasure.items())},capture_step_count=len(captures))
        rows.append(row)
    save('MODEL_RESULTS.json',rows);save('NONCOMPLETE.json',failures)
    kv=[]
    for f in ('amd_kv_retry1.json','qwama_kv_diagnostic_bf16_retry1.json','qwama_kv_diagnostic_fp16_retry1.json'):
        d=load(HERE/f)
        for b in (1,2,4,8):
         for c in (1,3):
          cs=[x for x in d['cases'] if x['batch']==b and x['siblings']==c]
          kv.append(dict(source=f,batch=b,siblings=c,cases=len(cs),
            mean_of_case_medians_ms={k:st.mean(x['median_ms'][k] for x in cs) for k in cs[0]['median_ms']},
            max_tv=max(x['checks']['max_tv'] for x in cs),argmax_disagree_cases=sum(not x['checks']['argmax_equal'] for x in cs),
            speedup_range=[min(x['median_ms']['confirmed_chain_glue']/x['median_ms']['promotion_plus_tip'] for x in cs),max(x['median_ms']['confirmed_chain_glue']/x['median_ms']['promotion_plus_tip'] for x in cs)]))
    save('KV_RESULTS.json',kv)
    print('Complete model runs',len(rows),'noncomplete',len(failures))

if __name__=='__main__':main()
