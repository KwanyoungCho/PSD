"""Audit release runtime, clean model arms, accounting, and GPU isolation."""
from collections import Counter
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('/data/chokwans99/duet-ver14-release-20261017')
TAGS=['v14_pair0_b','v13_pair0_a','v13_pair1_a','v14_pair1_b','v14_chain','v13_chain','v13_greedy','v14_greedy']
BASE='5642de044d9fa2d5e80dfaeeccc8f91cfd1be8e1'
RUNTIME='fa9963e56e45f73bba3aa4eb39990ebca3ec32c1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows=[];bench=None
    selected={str(f.relative_to(REPO)):sha(f) for f in (REPO/'ssd/ssd').rglob('*.py')}
    snapshot={str(f.relative_to(ROOT/'candidate')):sha(f) for f in (ROOT/'candidate/ssd/ssd').rglob('*.py')}
    assert selected==snapshot
    for tag in TAGS:
        f=ROOT/'runs'/tag;p=json.loads((f/'PROVENANCE.json').read_text());s=json.loads((f/'status.json').read_text());d=json.loads((f/'run.json').read_text())
        assert s['status']=='complete' and s['source_unchanged'] and d['status']=='complete'
        after=tag.startswith('v14')
        assert d['paper_commit']==(RUNTIME if after else BASE) and not d['paper_dirty'],tag
        assert not after or d['source_hashes']==selected
        b={k:v for k,v in p['source_sha256'].items() if k.startswith('ssd/bench/')}
        if bench is None:bench=b
        assert bench==b
        assert d['summary']['actual_output_tokens']==sum(map(len,d['outputs']))==sum(x['emitted'] for x in d['steps'])
        inc=[x for x in d['steps'] if not x['cap'] and not x['clipped'] and x.get('performance_included',True)]
        assert sum(x['emitted'] for x in inc)==d['summary']['tokens']
        assert abs(sum(x['seconds'] for x in inc)-d['summary']['seconds'])<1e-7
        wall=json.loads((f/'WALL_STEPS.json').read_text());steps=[x for g in d['warm_groups']+d['groups'] for x in g['steps']]
        assert len(wall)==len(steps)
        for w,x in zip(wall,steps):assert w['native_seconds']==x['seconds']
        uuids={r.split(',')[1].strip() for r in p['initial_gpu_state']['gpu'] if int(r.split(',')[0]) in s['gpus']}
        nmax=0;temp=0
        for line in (f/'telemetry.jsonl').read_text().splitlines():
            t=json.loads(line);counts=Counter(x.split(',')[0].strip() for x in t['processes'] if x)
            nmax=max(nmax,max(counts[u] for u in uuids))
            for gpu in t['gpu']:
                x=[v.strip() for v in gpu.split(',')]
                if int(x[0]) in s['gpus']:temp=max(temp,int(x[5]))
        assert nmax<=1,tag
        rows.append(dict(tag=tag,commit=d['paper_commit'],source_clean=True,gpus=s['gpus'],mode=d['args']['mode'],temperature=d['args']['temperature'],measured_requests=len(d['groups']),steps=len(d['steps']),excluded_long=d['summary']['excluded_long_steps'],cap_clip=sum(x['cap'] or x['clipped'] for x in d['steps']),max_processes=nmax,max_temp_c=temp))
    checks=json.loads((HERE/'checks/CHECKS.json').read_text());assert checks['status']=='pass' and checks['total']==106
    comparison=json.loads((HERE/'DIRECT_COMPARISON.json').read_text())
    assert comparison['tree_T07']['adopted_candidate']
    for key in ('chain_smoke','tree_greedy_smoke'):
        assert comparison[key]['outputs_equal'] and comparison[key]['events_equal']
    (HERE/'RELEASE_AUDIT.json').write_text(json.dumps(dict(status='pass',base=BASE,runtime_commit=RUNTIME,selected_runtime_exact=True,runs=rows),indent=2)+'\n')
    (HERE/'RUNTIME_MANIFEST.json').write_text(json.dumps(dict(commit=RUNTIME,sha256=selected),indent=2)+'\n')
    changed=subprocess.check_output(['git','diff',BASE,RUNTIME,'--name-status'],cwd=REPO,text=True)
    (HERE/'RUNTIME_CHANGED_FILES.txt').write_text(changed)
    print('Release source,',len(rows),'model runs, 106 checks, accounting and isolation PASS')

if __name__=='__main__':main()
