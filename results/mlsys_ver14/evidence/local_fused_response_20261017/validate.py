"""Audit complete frozen model runs without hiding compiled-cache provenance."""
from collections import Counter
from pathlib import Path
import hashlib,json,re,subprocess
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chokwans99/duet-local-followup-20261017')
TAGS=['fused_p0_smoke','base_fused_p0_a','fused_p0_b','fused_p0_c','base_fused_p0_d',
      'base_fused_p1_a','fused_p1_b','mixed_fused_control','heldout_fused','heldout_base_fused',
      'fused_profile','baseline_fused_profile','fused_bulk_reference','fused_bulk_candidate','heldout_fused_bulk','fused_bulk_profile']


def main():
    rows=[];benchmark=None
    for tag in TAGS:
        f=ROOT/'runs'/tag
        assert (f/'status.json').exists(),tag
        status=json.loads((f/'status.json').read_text());d=json.loads((f/'run.json').read_text());p=json.loads((f/'PROVENANCE.json').read_text())
        assert status['status']=='complete' and status['source_unchanged']
        checkout=ROOT/'code'/p['job']['variant']
        source={str(x.relative_to(checkout)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (checkout/'ssd/ssd').rglob('*.py')}
        assert d['source_hashes']==source,tag
        # Accidental compiled-cache tracking was confined to frozen snapshots.
        # Preserve the dirty report; no source, model or benchmark was dirty.
        dirty=d.get('paper_dirty','')
        assert all('__pycache__/' in x and x.endswith('.pyc') for x in dirty.splitlines() if x),tag
        bench={k:v for k,v in p['source_sha256'].items() if k.startswith('ssd/bench/')}
        if benchmark is None:benchmark=bench
        assert benchmark==bench
        assert d['summary']['actual_output_tokens']==sum(map(len,d['outputs']))==sum(x['emitted'] for x in d['steps'])
        inc=[s for s in d['steps'] if not s['cap'] and not s['clipped'] and s.get('performance_included',True)]
        assert sum(s['emitted'] for s in inc)==d['summary']['tokens']
        assert abs(sum(s['seconds'] for s in inc)-d['summary']['seconds'])<1e-7
        wall=json.loads((f/'WALL_STEPS.json').read_text())
        steps=[s for g in d['warm_groups']+d['groups'] for s in g['steps']]
        assert len(wall)==len(steps)
        for w,s in zip(wall,steps):assert w['native_seconds']==s['seconds']
        uuids={x.split(',')[1].strip() for x in p['initial_gpu_state']['gpu'] if int(x.split(',')[0]) in status['gpus']}
        maximum=0;temp=0;clocks=[]
        for line in (f/'telemetry.jsonl').read_text().splitlines():
            t=json.loads(line);c=Counter(x.split(',')[0].strip() for x in t['processes'] if x)
            maximum=max(maximum,max(c[u] for u in uuids))
            for gpu in t['gpu']:
                x=[v.strip() for v in gpu.split(',')]
                if int(x[0]) in status['gpus']:temp=max(temp,int(x[5]))
                if int(x[0])==status['gpus'][0] and float(x[3].split()[0])>0:clocks.append(float(x[4].split()[0]))
        assert maximum<=1,tag
        rows.append(dict(tag=tag,variant=p['job']['variant'],requests=len(d['groups']),
            profile=status['profile'],steps=len(d['steps']),excluded_long_steps=d['summary'].get('excluded_long_steps',0),
            cap_clip=sum(s['cap'] or s['clipped'] for s in d['steps']),max_processes=maximum,max_temp_c=temp,
            active_target_sm_clock_mean=sum(clocks)/len(clocks),paper_commit=d['paper_commit'],
            compiled_cache_dirty_lines=len(dirty.splitlines()),runtime_source_dirty=False))
    result=dict(status='pass',runs=rows,meaning='Exact source hash match; benchmark match; accounting and independent wall match; no external GPU process; compiled-cache-only dirty status is preserved, not relabeled clean.')
    (HERE/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    for variant in ('fused','fused_bulk','mixed_fused'):
        p=ROOT/'code'/variant
        base=subprocess.check_output(['git','rev-list','--max-parents=0','HEAD'],cwd=p,text=True).strip()
        diff=subprocess.check_output(['git','diff',base,'HEAD','--','ssd/ssd/**/*.py'],cwd=p,text=True)
        (HERE/(variant.upper()+'_RUNTIME.patch')).write_text(diff)
    print('Validated',len(rows),'model runs')


if __name__=='__main__':main()
