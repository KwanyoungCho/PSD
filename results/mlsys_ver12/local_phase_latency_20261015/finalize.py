"""Validate the completed campaign, then freeze portable evidence and status."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RAW=Path('/data/chokwans99/duet-local-phase-20261015')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    names=['7b_'+a for a in ('control','base','ds2','ds3','psbase','ps2','ps3','base_repeat')]
    names+=['13b_'+a for a in ('control','base','ds2','ds3','ps2','ps3')]
    inputs=json.loads((HERE/'INPUTS.json').read_text())
    expected=Counter({r['uid']:2 for r in inputs})
    manifests=[json.loads(p.read_text()) for p in RAW.glob('SOURCE_MANIFEST*.json')]
    assert all(m==manifests[0] for m in manifests)
    assert all(sha(ROOT/p)==s for p,s in manifests[0].items())
    assert not subprocess.check_output(['git','diff','HEAD','--','ssd/ssd'],cwd=ROOT)
    with gzip.open(HERE/'SAMPLES.json.gz','rt') as f:runs=json.load(f)
    by={r['name']:r for r in runs};assert set(names)==set(by)
    provenance={}
    for name in names:
        folder=RAW/name;r=json.loads((folder/'run.json').read_text())
        assert r['status']=='complete' and len(r['groups'])==len(r['outputs'])==24
        assert Counter(g['request_records'][0]['paper_uid'] for g in r['groups'])==expected
        assert len(r['warm_groups'])==3
        assert len(r['source_hashes'])>0
        for s in by[name]['rows']:
            assert s['emitted_len']==s['accepted_len']
            assert s['verify_width']<=39
            if name.endswith('control'):continue
            for phase in ('ds','ps'):
                p=s[phase+'_probe']
                assert len(p['forward'])==3 and all(0<=n<=39 for n in p['nodes'])
                assert abs(s[phase+'_compute_window_ms']-s[phase+'_forward_ms']-s[phase+'_margin_ms'])<1e-9
        assert all(sha(Path(p))==digest for p,digest in by[name]['source_hashes'].items())
        provenance[name]={k:r[k] for k in ('paper_commit','paper_dirty','args','env','source_hashes','adapter_hashes','resolved_tree_runtime')}
        provenance[name]['command']=json.loads((folder/'COMMAND.json').read_text())
    audit=json.loads((HERE/'AUDIT.json').read_text())
    assert all(v['outputs_equal'] for v in audit['probe_comparison'].values())
    assert audit['repeat_comparison']['7b']['outputs_equal']
    assert not any(v['foreign'] for v in audit['telemetry'].values())
    phase=json.loads((HERE/'timelines/PHASE_AVERAGES.json').read_text())
    assert len(phase['sets'])==2
    for item in phase['sets']:
        model='13b' if '13b' in item['model'] else '7b'
        source=by[model+'_base']['performance']
        assert [r['n'] for r in source]==[r['n'] for r in item['rows']]
        for a,b in zip(source,item['rows']):
            for field in ('step_ms','ds_forward_once_ms','ds_compute_window_ms','ps_forward_once_ms','ps_compute_window_ms'):
                assert abs(a[field]-b[field])<1e-9
    pages=len(PdfReader(HERE/'timelines/ALL_MODELS.pdf').pages);assert pages==6
    check=HERE/'checks';check.mkdir(exist_ok=True)
    for name,count in [('timeline_tests.log',22),('probe_tests.log',2)]:
        content=(RAW/name).read_text();assert f'Ran {count} tests' in content and '\nOK' in content
        shutil.copy2(RAW/name,check/name)
    for f in RAW.glob('SOURCE_MANIFEST*.json'):shutil.copy2(f,check/f.name)
    (HERE/'RUN_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    status=dict(status='complete',scope='Requested short local latency and node-count diagnosis; not optimal-parameter search or full-dataset quality validation',
        runs=names,model_pairs=2,measured_turns=24*len(names),independent_input_turns=12,
        timeline_pages=pages,checks=dict(plot_tests=22,probe_cuda_tests=2,output_parity='24/24 per model',
        runtime_unchanged=True,all_runtime_benchmark_manifests_identical=True,all_sources_hash_verified=True,
        no_foreign_gpu_process_observed=True,telemetry_gap_disclosed_and_repeat_completed=True),
        artifacts={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name not in ('PLAN.json','CHECKS.json') and '__pycache__' not in p.parts})
    status['analysis_source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [
        * (ROOT/'ssd/tools/duet_timeline').glob('*.py'),
        ROOT/'ssd/tests/test_consecutive_timeline.py',ROOT/'ssd/tests/test_phase_probe.py']}
    (HERE/'CHECKS.json').write_text(json.dumps(status,indent=2)+'\n')
    plan=json.loads((HERE/'PLAN.json').read_text());plan.update(status='complete',new_inference_complete=True)
    plan['completed_runs']=names
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    (RAW/'STATUS.json').write_text(json.dumps(dict(status='complete',runs=names,report=str(HERE/'REPORT.md')),indent=2)+'\n')
    print('Complete:',len(names),'runs;',pages,'timeline pages; core runtime unchanged.')


if __name__=='__main__':main()
