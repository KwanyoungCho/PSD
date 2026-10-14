"""Prompt-group bootstrap and source-specific metrics for finished campaigns."""
import argparse,json
from pathlib import Path
import numpy as np
from mlsys_measurement import base_eligible,batch_metrics
from mlsys_request_reporting import common_question_summary


def groups(report):
    b=report['args']['batch'];by={}
    for g in report['groups']:
        if not g['phase'].startswith('measured'):continue
        # Repeated passes over the same prompts are one bootstrap cluster.
        row=by.setdefault(g['index'],np.zeros(4))
        kept=[s for s in g['steps'] if base_eligible(s,b) and not s.get('graph_preparation',False)]
        row+=np.asarray([sum(s['emitted'] for s in kept),sum(s['seconds'] for s in kept),
                         sum(sum(s['verify']) for s in kept),len(kept)])
    return by


def compare(a,b):
    if len(a['outputs']) != len(b['outputs']):
        raise ValueError('Arms have different output populations')
    ga,gb=groups(a),groups(b);keys=sorted(ga.keys()&gb.keys())
    aa=np.asarray([ga[k] for k in keys]);bb=np.asarray([gb[k] for k in keys])
    gen=np.random.default_rng(714);samples=[];lat=[]
    for _ in range(5000 if keys else 0):
        ix=gen.integers(len(keys),size=len(keys));x=aa[ix].sum(0);y=bb[ix].sum(0)
        if min(x[0],x[1],x[3],y[0],y[1],y[3])<=0:continue
        samples.append((y[0]/y[1])/(x[0]/x[1])-1)
        if x[2] > 0 and y[2] > 0:
            lat.append((y[2]/y[3])/(x[2]/x[3])-1)
    def change(metric):
        x,y=a['summary'][metric],b['summary'][metric]
        return y/x-1 if x is not None and y is not None and x>0 else None
    return dict(prompt_group_clusters=len(keys),
        tps_change_ci95=np.quantile(samples,[.025,.975]).tolist() if samples else None,
        verify_change_ci95=np.quantile(lat,[.025,.975]).tolist() if lat else None,
        valid_tps_bootstraps=len(samples), valid_verify_bootstraps=len(lat),
        tps_change=change('tps'), verify_change=change('verify_ms'),
        equal_output_sequences=sum(x==y for x,y in zip(a['outputs'],b['outputs'])),
        output_sequences=len(a['outputs']),same_all_outputs=a['outputs']==b['outputs'],
        caveat='Online verify means also depend on reached shapes. Component controls establish execution-cost causality.')


def summarize(j):
    b=j['args']['batch'];ss=[s for s in j['steps'] if base_eligible(s,b) and not s.get('graph_preparation',False)]
    source={}
    for phase in (0,1,2):
        es=[e for s in ss for e in s['events'] if e['source']==phase]
        source[str(phase)]=dict(requests=len(es),al=sum(e['accepted_len'] for e in es)/len(es) if es else None,
                               valid_k=sum(e['valid_k'] for e in es)/len(es) if es else None)
    return dict(summary=j['summary'],source=source,batch_metrics=batch_metrics(ss,b),
        preparation=j.get('measurement',{}).get('preparation_all_decode'),
        parameters={k:j['args'][k] for k in ('target','draft','batch','temperature','k1','k2','exit_layer','p1_nodes','p2_nodes','p1_verify','p2_verify','roots','siblings')})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();reports={}
    for f in args.directory.glob('*.json'):
        j=json.loads(f.read_text())
        if isinstance(j,dict) and j.get('status')=='complete' and 'summary' in j:reports[f.stem]=j
    out=dict(directory=str(args.directory),arms={k:summarize(j) for k,j in reports.items()},comparisons={})
    if reports and all(j.get('request_records') is not None for j in reports.values()):
        out['common_request_population'] = common_question_summary(
            {k:j['request_records'] for k,j in reports.items()})
    else:
        out['common_request_population'] = dict(status='unavailable',
            reason='Historical raw lacks explicit per-request termination records')
    if 'base' in reports:
        out['comparisons']={k:compare(reports['base'],j) for k,j in reports.items() if k!='base'}
    args.output.write_text(json.dumps(out,indent=2))
    for k,j in out['comparisons'].items():print(k,j)

if __name__=='__main__':main()
