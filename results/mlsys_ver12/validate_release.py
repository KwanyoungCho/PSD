"""Offline ver1.2 source/evidence/timeline checks; no inference or GPU use.

The 382-test CUDA run belongs to the unchanged accepted runtime. This script
verifies that identity and runs the plotting regressions in fresh interpreters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
SUITES=('test_consecutive_timeline','test_model_breakdowns',
        'test_plot_duet_aligned_timeline','test_ver10_profile_analysis')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=SUITES)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.worker:
        sys.path[:0]=[str(ROOT/'ssd'),str(ROOT/'ssd/tests')]
        suite=unittest.defaultTestLoader.loadTestsFromName(a.worker)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        (out/(a.worker+'.json')).write_text(json.dumps(dict(tests=result.testsRun,
            errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped)),indent=2)+'\n')
        raise SystemExit(0 if result.testsRun and result.wasSuccessful() and not result.skipped else 1)
    proof=ROOT/'results/mlsys_ver11/ps_preparation_20261015/checks'
    provenance=json.loads((proof/'CHECKS_PROVENANCE.json').read_text())
    expected={k:v for k,v in provenance['source_hashes'].items() if k.startswith('ssd/ssd/')}
    current={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'ssd/ssd').rglob('*.py'))}
    assert current==expected,'Runtime no longer matches accepted 382-test source'
    prior=json.loads((proof/'CHECKS.json').read_text())
    assert len(prior)==40 and sum(r['tests'] for r in prior)==382
    assert all(r['returncode']==0 and r['errors']==r['failures']==r['skipped']==0 for r in prior)
    source={str(p.relative_to(ROOT)):sha(p) for d in ('ssd/ssd','ssd/bench','ssd/tests','ssd/tools/duet_timeline')
            for p in sorted((ROOT/d).rglob('*.py'))}
    env=dict(os.environ,PYTHONPATH=str(ROOT/'ssd')+os.pathsep+str(ROOT/'ssd/tests'),PYTHONDONTWRITEBYTECODE='1')
    rows=[]
    for name in SUITES:
        with (out/(name+'.log')).open('w') as log:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',name,'--output',str(out)],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        row=json.loads((out/(name+'.json')).read_text());row.update(name=name,returncode=run.returncode)
        rows.append(row);print(row,flush=True)
    from pypdf import PdfReader
    audits=[]
    figures=ROOT/'results/mlsys_ver12/model_timelines'
    index=json.loads((figures/'INDEX.json').read_text())
    assert index['model_count']==len(index['models'])==5
    assert index['pages']==17
    assert index['builder_sha256']==sha(ROOT/'ssd/tools/duet_timeline/build_model_timelines.py')
    assert index['renderer_sha256']==sha(ROOT/'ssd/tools/duet_timeline/plot_consecutive_steps.py')
    assert index['manifest_sha256']==sha(figures/'SOURCES.json')
    sys.path.insert(0,str(ROOT/'ssd/tools/duet_timeline'))
    from phase_averages import read_samples, summarize, overlap_summary
    manifest=json.loads((figures/'SOURCES.json').read_text())
    sample_path=figures/manifest['phase_samples']['path']
    assert sha(sample_path)==manifest['phase_samples']['sha256']==index['phase_samples_sha256']
    assert index['phase_averages_tool_sha256']==sha(ROOT/'ssd/tools/duet_timeline/phase_averages.py')
    averages_path=figures/'PHASE_AVERAGES.json'
    assert sha(averages_path)==index['phase_averages_sha256']
    averages=json.loads(averages_path.read_text())
    samples=read_samples(sample_path)
    keyed={(x['model'],x['setting']):x for x in averages['sets']}
    for item in samples['sets']:
        summary=keyed[(item['model'],item['setting'])]
        assert summarize(item['samples'])==summary['rows']
        assert overlap_summary(item['samples'])==summary['proxy_overlap']
    assert len(PdfReader(figures/'ALL_MODELS.pdf').pages)==index['pages']
    assert sha(figures/'ALL_MODELS.pdf')==index['all_models_pdf_sha256']
    for model in index['models']:
        path=figures/model['id']/'AUDIT.json'
        audit=json.loads(path.read_text());pdf=path.parent/'TIMELINE.pdf'
        assert sha(path)==model['audit_sha256']
        assert sha(pdf)==model['pdf_sha256']
        assert len(PdfReader(pdf).pages)==audit['pages']==len(audit['panels'])
        for panel in audit['panels']:
            window=path.parent/panel['window'];assert sha(window)==panel['window_sha256']
            w=json.loads(window.read_text());assert w['step_ids'][1]==w['step_ids'][0]+1
            assert panel['phase_summary']==keyed[(model['id'],panel['setting'])]
            assert panel['clock_shift_ns']==panel['minimum_bar_width_ms']==0
            for tr in w['transitions']:
                assert abs(tr['ready_margin_ms']-(tr['deadline_ns']-tr['draft_receive_ready_ns'])/1e6)<1e-9
                assert tr['deadline_ns']==tr['next_send']['wall_start_ns']
            # Each forward raw interval remains represented, not a phase sum.
            assert len(panel['forward_dividers'])==sum(e['label'] in w['forward_counts'] for e in w['draft_events'])
        audits.append(dict(file=str(path.relative_to(ROOT)),pages=audit['pages'],sha256=sha(path)))
    stable=all(sha(ROOT/k)==v for k,v in source.items())
    assert stable
    report=dict(status='complete' if all(r['returncode']==0 for r in rows) else 'failed',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime_commit='9390e21',runtime_file_count=len(current),runtime_matches_382_test_source=True,
        inherited_cuda_tests=382,inherited_cuda_suites=40,new_cuda_run=False,
        offline_suites=rows,offline_tests=sum(r['tests'] for r in rows),source_unchanged=stable,
        source_hashes=source,prior_provenance_sha256=sha(proof/'CHECKS_PROVENANCE.json'),
        timeline_audits=audits,python=sys.executable,
        note='Prior rejected 386-test candidate is not the released runtime. Plotting suites are isolated to avoid bench namespace collisions.')
    (out/'CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['status']=='complete'


if __name__=='__main__':main()
