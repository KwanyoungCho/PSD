"""Offline evidence audit; optional runtime/protected-worktree checks here."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-runtime',action='store_true');p.add_argument('--check-protected',action='store_true')
    args=p.parse_args();variants=load(HERE/'REMOTE_VARIANTS.json');counts={};failures=[];wall_count=0
    common_bench=None;settings={};cohort={}
    for folder in sorted((HERE/'raw/runs').iterdir()):
        status=load(folder/'status.json');data=load(folder/'run.json');prov=load(folder/'PROVENANCE.json')
        assert status['status']=='complete' and status['exit_code']==0 and status['source_unchanged']
        assert data['paper_dirty']=='' and data['status']=='complete'
        variant=status['variant'];counts[variant]=counts.get(variant,0)+1
        assert data['source_hashes']==variants[variant]['runtime_sha256']
        runtime={k:v for k,v in prov['source_sha256'].items() if k.startswith('ssd/ssd/')}
        bench={k:v for k,v in prov['source_sha256'].items() if k.startswith('ssd/bench/')}
        assert runtime==data['source_hashes']
        if common_bench is None:common_bench=bench
        assert bench==common_bench,'Benchmark differs across arms'
        model=status['model']
        assert data['kwargs']==settings.setdefault(model,data['kwargs'])
        assert data['prompt_ids']==cohort.setdefault(model,data['prompt_ids'])
        if (folder/'PARITY.json').exists():
            q=load(folder/'PARITY.json')
            if not(q['outputs_equal'] and q['events_equal']):failures.append(folder.name)
        if not status['profile']:
            wall=load(folder/'WALL_STEPS.json')
            steps=[s for g in data['warm_groups']+data['groups'] for s in g['steps']]
            assert len(wall)==len(steps)
            for w,s in zip(wall,steps):
                assert w['native_seconds']==s['seconds'] and w['wall_seconds']>=w['native_seconds']
                assert w['source']==[e.get('source') for e in s['events']]
                assert w['accepted_len']==[e.get('accepted_len') for e in s['events']]
            wall_count+=len(wall)
    assert sum(counts.values())==32
    assert failures==['hostlist_13b_host_a'],failures
    telemetry=load(HERE/'TELEMETRY.json')['runs'];assert len(telemetry)==32
    assert all(r['all_assigned_gpus_initially_free'] and r['source_unchanged'] for r in telemetry)
    assert all(d['max_simultaneous_processes']<=1 and d['samples'] for r in telemetry for d in r['devices'].values())
    selected=load(HERE/'SELECTED_RUNTIME.json');checks=load(HERE/'raw/checks_prepack/CHECKS.json')
    assert checks['status']=='complete' and checks['source_unchanged'] and checks['tests']==69
    assert checks['source_sha256']==selected['runtime_sha256']==variants['prepack']['runtime_sha256']
    if args.check_runtime:
        current={str(p.relative_to(REPO)):sha(p) for p in (REPO/'ssd/ssd').rglob('*.py')}
        assert current==selected['runtime_sha256']
    protected={}
    if args.check_protected:
        root=Path('/home/chokwans99/PSD-MLSys-ver1.3')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='5642de044d9fa2d5e80dfaeeccc8f91cfd1be8e1'
        assert subprocess.check_output(['git','status','--short'],cwd=root,text=True)==''
        expected=load(root/'results/mlsys_ver13/RUNTIME_MANIFEST.json')['runtime_sha256']
        assert {str(p.relative_to(root)):sha(p) for p in (root/'ssd/ssd').rglob('*.py')}==expected
        protected['ver1.3']='HEAD, clean status and runtime unchanged'
        root=Path('/home/chokwans99/PSD-MLSys-ver0.1')
        old=load(REPO/'results/mlsys_ver12/target_gap_optimization_20261017/ORIGINAL_WORKTREE.json')
        assert subprocess.check_output(['git','status','--short'],cwd=root,text=True)==old['status']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==old['head']
        assert {str(p.relative_to(root)):sha(p) for p in (root/'ssd/ssd').rglob('*.py')}==old['runtime_sha256']
        protected['original_ver1.2']='HEAD, dirty status and runtime unchanged'
    archive=load(HERE/'ARCHIVE_CHECK.json')
    assert archive['status']=='pass' and sha(HERE/'RAW_RESULTS.tar.xz')==archive['archive_sha256']
    assert all(sha(HERE/name)==digest for name,digest in archive['raw_sha256'].items())
    result=dict(status='complete_with_disclosed_candidate_rejection',completed_model_runs=32,
        variant_counts=counts,benchmark_source_identical=True,all_run_sources_unchanged=True,
        failed_parity_runs=failures,paired_model_variants=7,whole_step_comparisons=14,
        selected_candidate='prepack',selected_regression_tests=69,independent_wall_steps_checked=wall_count,
        selected_runtime_checked=args.check_runtime,protected=protected,
        telemetry_max_temperature_c=max(d['temperature_max_c'] for r in telemetry for d in r['devices'].values()),
        archive_files=archive['files'],archive_sha256=archive['archive_sha256'],
        analysis_complete_is_not_all_candidates_passed=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
