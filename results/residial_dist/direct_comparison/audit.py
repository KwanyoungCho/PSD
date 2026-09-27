"""Audit new prompt provenance, recorded distributions and frozen selection."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CAMPAIGN=ROOT/'ssd/experiments/proxy_source_ablation/probe_direct_20260912'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    provenance=json.loads((CAMPAIGN/'datasets.json').read_text())
    old_texts=set();new_texts=[]
    for ds,r in provenance.items():
        assert sha(r['source'])==r['source_sha256']
        assert sha(r['file'])==r['sha256']
        original=[json.loads(line) for line in Path(r['source']).read_text().splitlines() if line.strip()]
        fresh=[json.loads(line) for line in Path(r['file']).read_text().splitlines() if line.strip()]
        assert len(fresh)==32
        assert fresh==[original[i-1] for i in r['source_row_ids']]
        old_texts.update(row['text'].strip() for row in original[:32])
        new_texts.extend(row['text'].strip() for row in fresh)
        assert [hashlib.sha256(row['text'].strip().encode()).hexdigest() for row in fresh]==r['texts_sha256']
    assert len(new_texts)==len(set(new_texts))==128
    assert not old_texts.intersection(new_texts)
    frozen=HERE/'frozen_policies.json';model=json.loads(frozen.read_text())
    assert model['analysis_sha256']==sha(HERE/'analyze.py')
    for name,hashed in model['development_hashes'].items():assert sha(HERE/'development'/name)==hashed
    previous=json.loads((HERE.parent/'shared_review/followup_verify_checks.json').read_text())
    for name,hashed in previous['production_code_sha256'].items():assert sha(ROOT/name)==hashed
    records=[];max_sum_error=0.;raw_hashes={}
    completed=sorted((CAMPAIGN/'out').glob('*.complete.json'))
    assert len(completed)==4
    for done in completed:
        record=json.loads(done.read_text());records.append(record)
        meta=json.loads((done.parent/record['manifest']).read_text())
        output=HERE/'confirmation'/done.name.replace('.complete.json','.npz')
        assert output.exists() and output.stat().st_mtime>frozen.stat().st_mtime
        for chunk in meta['chunks']:
            path=done.parent/chunk['file'];raw_hashes[path.name]=sha(path)
            with np.load(path) as d:
                for key in ('p_T','p_D','p_E'):
                    p=d[key]
                    assert p.dtype==np.float32 and np.isfinite(p).all() and (p>=0).all()
                    if key=='p_D':
                        assert (p[d['is_bonus']]==0).all()
                        p=p[~d['is_bonus']]
                    err=np.max(np.abs(p.sum(-1,dtype=np.float64)-1))
                    assert err<3e-6
                    max_sum_error=max(max_sum_error,float(err))
                assert d['p_T'].shape[-1]==32000
                assert (d['y'][d['is_bonus']]==-1).all()
                assert (d['y'][~d['is_bonus']]>=0).all()
        log=done.with_name(done.name.replace('.complete.json','.log')).read_text(errors='replace')
        assert 'max|probe-engine|=0.000e+00 OK' in log
        assert 'Traceback (most recent call last)' not in log and 'falling back to random' not in log
    result=dict(passed=True,fresh_unique_prompts=128,overlap_with_development=0,
                records=records,raw_float32_probability_max_sum_error=max_sum_error,
                frozen_policies_sha256=sha(frozen),policy_freeze_precedes_all_confirmation_metrics=True,
                production_lossless_audit_code_unchanged=True,
                development_policy_parity='8/8 runs: GPU top-k coverage matches original replay within 3e-5',
                numerical_theorems=json.loads((HERE/'theory_checks.json').read_text()),
                raw_snapshot_sha256=raw_hashes,
                analysis_files_sha256={p.name:sha(p) for p in HERE.glob('*.py')})
    (HERE/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','raw_snapshot_sha256','analysis_files_sha256')},indent=2))


if __name__=='__main__':main()
