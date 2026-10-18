"""Validate the final functional runs; these are not full performance runs."""
from pathlib import Path
import collections,hashlib,json,math,os,sys


def main():
    raw=Path(os.environ['VER15_RAW']);code=Path(os.environ['VER15_CHECKOUT'])
    sys.path.insert(0,str(code/'ssd/bench'))
    from mlsys_request_reporting import classify
    expected=json.loads((raw/'INPUTS_SMOKE.json').read_text());out={}
    for mode,temp in (('stochastic',.7),('greedy',0.)):
        tag=f'smoke_{mode}_v5';folder=raw/'runs'/tag
        status=json.loads((folder/'status.json').read_text());d=json.loads((folder/'run.json').read_text())
        assert status['status']=='complete' and status['source_unchanged'] and status['exit_code']==0
        assert d['status']=='complete' and len(d['warm_groups'])==3 and len(d['groups'])==6
        assert d['prompt_ids']==[m['input_ids'] for m in expected]
        groups=d['warm_groups']+d['groups']
        records=[r for g in groups for r in g['request_records']]
        assert [r['paper_uid'] for r in records]==[m['uid'] for m in expected]
        assert d['kwargs']['dtype']=='bfloat16' and d['args']['batch']==1
        for r,m in zip(records,expected):
            classify(r)
            assert len(r['output_ids'])==r['completion_tokens']<=r['effective_output_cap']==min(96,2048-len(m['input_ids']))
            assert r['target_temperature']==r['draft_temperature']==temp
        counts=collections.Counter(e['source'] for s in d['steps'] for e in s['events'])
        assert set(counts)=={0,1,2}
        assert all(e['verification']=='target_sample_match' for s in d['steps'] for e in s['events'])
        assert all(math.isfinite(s['seconds']) and s['seconds']>0 for s in d['steps'])
        assert not (folder/'UNEXPECTED_PREEMPTION.json').exists()
        assert all(hashlib.sha256((code/path).read_bytes()).hexdigest()==h for path,h in d['source_hashes'].items())
        old=json.loads((raw/'runs'/f'smoke_{mode}_v4'/'run.json').read_text())
        matched=sum(x['outputs']==y['outputs'] for x,y in zip(groups,old['warm_groups']+old['groups']))
        assert matched==9,'CPU/GPU score-selection functional comparison changed'
        out[tag]=dict(status='pass',mode=mode,temperature=temp,input_turns=9,measured_turns=6,
            warmup_turns=3,output_cap=96,measured_steps=len(d['steps']),sources=dict(counts),
            actual_tokens=sum(r['completion_tokens'] for r in records),cpu_score_vs_gpu_score_output_match=matched,
            input_ids_preserved=True,all_steps_target_sampling=True,no_preemption=True,source_unchanged=True,
            dirty_development_run=True,performance_claim=False,
            run_sha256=hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
    (code/'results/mlsys_ver15/checks/MODEL_SMOKE.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
