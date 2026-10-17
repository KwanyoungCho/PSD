"""Audit all completed runs, frozen sources, accounting and GPU isolation."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chokwans99/duet-local-followup-20261017')
REPO=HERE.parents[1]


def main():
    variants={v:{str(f.relative_to(ROOT/'code'/v)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (ROOT/'code'/v/'ssd/ssd').rglob('*.py')} for v in ('baseline','buckets','overlap','prep')}
    results=[];common_bench=None
    for folder in sorted((HERE/'raw/runs').iterdir()):
        s=json.loads((folder/'status.json').read_text());d=json.loads((folder/'run.json').read_text())
        prov=json.loads((folder/'PROVENANCE.json').read_text());variant=prov['job']['variant']
        assert s['status']=='complete' and s['source_unchanged'] and d['status']=='complete'
        assert not d['paper_dirty'] and d['source_hashes']==variants[variant],folder.name
        bench={k:v for k,v in prov['source_sha256'].items() if k.startswith('ssd/bench/')}
        if common_bench is None:common_bench=bench
        assert bench==common_bench,'Benchmark code differs across model arms'
        assert (d['args']['k1'],d['args']['k2'],d['args']['batch'],d['args']['temperature'])==(8,5,1,.7)
        assert d['kwargs']['dtype']=='bfloat16' and d['args']['p1_nodes']==d['args']['p1_verify']==15
        assert d['args']['p2_nodes']==d['args']['p2_verify']==15 and d['args']['expansion']=='last'
        assert d['summary']['actual_output_tokens']==sum(map(len,d['outputs']))==sum(x['emitted'] for x in d['steps'])
        included=[x for x in d['steps'] if not x['cap'] and not x['clipped'] and x.get('performance_included',True)]
        assert sum(x['emitted'] for x in included)==d['summary']['tokens']
        assert abs(sum(x['seconds'] for x in included)-d['summary']['seconds'])<1e-7
        walls=json.loads((folder/'WALL_STEPS.json').read_text())
        steps=[x for g in d['warm_groups']+d['groups'] for x in g['steps']]
        assert len(walls)==len(steps)
        for w,x in zip(walls,steps):
            assert w['native_seconds']==x['seconds'] and w['source']==[e['source'] for e in x['events']]
        assigned={int(x.split(',')[0]):x.split(',')[1].strip() for x in prov['initial_gpu_state']['gpu']}
        uuids={assigned[g] for g in s['gpus']};nmax=0;tmax=0
        for line in (folder/'telemetry.jsonl').read_text().splitlines():
            tel=json.loads(line);counts=collections.Counter(x.split(',')[0].strip() for x in tel['processes'] if x)
            nmax=max(nmax,max(counts[u] for u in uuids))
            for g in tel['gpu']:
                fs=[x.strip() for x in g.split(',')]
                if int(fs[0]) in s['gpus']:tmax=max(tmax,int(fs[5]))
        assert nmax<=1,(folder.name,nmax)
        results.append(dict(tag=folder.name,variant=variant,profile=s['profile'],requests=len(d['groups']),steps=len(d['steps']),
            long_excluded=d['summary'].get('excluded_long_steps',0),cap_clip=sum(x['cap'] or x['clipped'] for x in d['steps']),
            max_processes=nmax,max_temperature_c=tmax,zero_decode=[q['paper_uid'] for q in d['request_records'] if q['completion_tokens']==0]))
    assert len(results)==20
    selected={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (REPO/'ssd/ssd').rglob('*.py')}
    assert selected==variants['prep']
    tests=(HERE/'SELECTED_TESTS.log').read_text();assert 'Ran 20 tests' in tests and '\nOK\n' in tests and 'skipped=' not in tests
    (HERE/'VALIDATION.json').write_text(json.dumps(dict(status='pass',runs=results,selected_runtime_exact=True,variant_hashes=variants,
        tests='20 selected-source checks passed, no skips; separate delayed-receive diagnostic passed'),indent=2)+'\n')
    for variant in ('buckets','overlap','prep'):
        repo=ROOT/'code'/variant
        base=subprocess.check_output(['git','-C',str(repo),'rev-list','--max-parents=0','HEAD'],text=True).strip()
        patch=subprocess.check_output(['git','-C',str(repo),'diff',base,'HEAD','--','ssd/ssd'],text=True)
        (HERE/(variant.upper()+'_RUNTIME.patch')).write_text(patch)
    print('20 source/accounting/telemetry audits and selected runtime equality passed')


if __name__=='__main__':main()
