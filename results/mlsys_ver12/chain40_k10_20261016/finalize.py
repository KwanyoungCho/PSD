"""Validate native-chain evidence and save a portable completion record."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RAW=Path('/data/chokwans99/duet-chain40-k10-20261016')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    with gzip.open(HERE/'SAMPLES.json.gz','rt') as f:runs=json.load(f)
    assert {r['model'] for r in runs}=={'7b','13b'}
    status=json.loads((RAW/'STATUS.json').read_text())
    assert status['status']=='complete' and all(r['returncode']==0 for r in status['runs'])
    assert not subprocess.check_output(['git','diff','HEAD','--','ssd/ssd'],cwd=ROOT)
    inputs=json.loads((HERE/'INPUTS.json').read_text())
    expected=Counter({r['uid']:2 for r in inputs})
    manifests=[];provenance={};audit={}
    for run in runs:
        model=run['model'];folder=RAW/model
        r=json.loads((folder/'run.json').read_text())
        assert r['status']=='complete' and len(r['groups'])==len(r['outputs'])==24
        assert len(r['warm_groups'])==3
        assert Counter(g['request_records'][0]['paper_uid'] for g in r['groups'])==expected
        manifest=json.loads((folder/'SOURCE_MANIFEST.json').read_text())
        assert all(sha(ROOT/p)==s for p,s in manifest.items())
        manifests.append(manifest)
        assert all(sha(Path(p))==s for p,s in run['source_hashes'].items())
        for row in run['rows']:
            assert row['accepted_len']==row['emitted_len'] and row['verify_width']==10
            for phase in ('ds','ps'):
                values=row['rounds'][phase]
                assert len(values)==10 and all(v>0 for v in values)
                assert abs(sum(values)/10-row[phase+'_forward_once_ms'])<1e-9
                assert abs(sum(values)+row[phase+'_margin_ms']-row[phase+'_compute_window_ms'])<1e-9
        for key in ('survival','all_nonterminal_survival'):
            for curve in run[key]:
                probs=[p['probability'] for p in curve['points']]
                assert probs[0]==1 and all(a>=b for a,b in zip(probs,probs[1:]))
                assert abs(sum(probs)-curve['mean_al'])<1e-12
                assert sum(curve['histogram'].values())==curve['steps']
        cmd=json.loads((folder/'COMMAND.json').read_text())
        devices=set(map(int,cmd['env']['CUDA_VISIBLE_DEVICES'].split(',')))
        pids={str(int(p.stem.split('_')[-1])) for p in folder.glob('preparation_*.jsonl')}
        times=[];foreign=[];clocks=defaultdict(list);temps=defaultdict(list)
        for line in (folder/'telemetry.jsonl').read_text().splitlines():
            t=json.loads(line);times.append(t['wall_ns']);uuids=set()
            for g in t['gpu']:
                parts=[s.strip() for s in g.split(',')]
                if int(parts[0]) in devices:
                    uuids.add(parts[1]);clocks[parts[0]].append(float(parts[5]));temps[parts[0]].append(float(parts[6]))
            for job in t['jobs']:
                if not job:continue
                parts=[s.strip() for s in job.split(',')]
                if parts[0] in uuids and parts[1] not in pids:foreign.append(dict(time=t['wall_ns'],job=job))
        assert times and not foreign
        audit[model]=dict(devices=sorted(devices),samples=len(times),foreign=foreign,
            maximum_sampling_gap_s=float(max(np.diff(times))/1e9),
            clocks_mean_mhz={k:float(np.mean(v)) for k,v in clocks.items()},
            temperature_max_c={k:max(v) for k,v in temps.items()},
            shape_audited_steps=sum(len(json.loads(p.read_text())['records']) for p in folder.glob('chain_shapes_request_*.json')),
            eligible_steps=run['eligibility']['included'])
        provenance[model]={k:r[k] for k in ('paper_commit','paper_dirty','args','env','source_hashes','adapter_hashes')}
        provenance[model].update(command=cmd,resolved=run['resolved'],runtime_benchmark_manifest=manifest)
    assert manifests[0]==manifests[1]
    phases=json.loads((HERE/'timelines/PHASE_AVERAGES.json').read_text())
    assert len(phases['sets'])==2
    for item in phases['sets']:
        source=next(r['summary'] for r in runs if '_'+r['model']+'_' in item['model'])
        assert [r['n'] for r in source]==[r['n'] for r in item['rows']]
        for a,b in zip(source,item['rows']):
            for field in ('step_ms','al','tps','ds_forward_once_ms','ds_compute_window_ms','ps_forward_once_ms','ps_compute_window_ms'):
                assert abs(a[field]-b[field])<1e-9
    pages=len(PdfReader(HERE/'timelines/ALL_MODELS.pdf').pages);assert pages==6
    assert len(PdfReader(HERE/'AL_SURVIVAL_MODELS.pdf').pages)==2
    for name,count in [('timeline_tests.log',22),('chain_tests.log',4)]:
        content=(HERE/'checks'/name).read_text()
        assert f'Ran {count} tests' in content and '\nOK' in content
    write(HERE/'RUN_PROVENANCE.json',provenance)
    write(HERE/'AUDIT.json',audit)
    write(HERE/'PLAN.json',dict(status='complete',local_only=True,models=['layerskip-llama2-7b + llama-68m','layerskip-llama2-13b + llama-68m'],
        mode='native chain',batch=1,k1=10,k2=10,ds_roots=40,ps_roots=40,ds_fan_out=[4]*7+[3]*4,
        ds_width=40,ps_width=40,exit_layers_zero_based=[21,26],temperature=.7,dtype='bfloat16',
        independent_input_turns=12,passes=2,input_cap=512,output_cap=192,
        primary_outputs=['per-source latency timelines','P(AL >= x | cache source)'],
        limitations='Short diagnostic sample; not full Spec-Bench AL or an optimized-parameter comparison; detailed profiling enabled.'))
    write(HERE/'CHECKS.json',dict(status='complete',runtime_unchanged=True,native_chain=True,
        main_runs=2,warmup_turns_per_run=3,measured_turns_per_run=24,independent_inputs_per_run=12,
        shapes_verified='40 DS + 40 PS roots; 40 forward rows; 10 + 10 timed forwards in every step',
        target_verify_width=10,al_range=[1,11],eligible_steps={r['model']:r['eligibility']['included'] for r in runs},
        tests=dict(existing_timeline=22,chain_reporting=4),timeline_pages=pages,survival_pages=2,
        telemetry='No foreign process observed on assigned GPUs; polling coverage in AUDIT.json',
        runtime_benchmark_sources_identical=True,raw_hashes_verified=True,
        survival_monotone_and_mean_identity_verified=True,timeline_and_report_populations_identical=True,
        artifacts={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name!='CHECKS.json' and '__pycache__' not in p.parts},
        analysis_sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'ssd/tools/duet_timeline').glob('*.py')}))
    print('Complete: 2 native-chain runs, 6 timeline pages, 2 survival pages, 26 reporting tests.')


if __name__=='__main__':main()
