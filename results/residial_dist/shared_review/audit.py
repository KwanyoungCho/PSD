"""Final raw-data, code provenance, and numerical-floor audit."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    out=Path(__file__).parent
    campaign=Path('ssd/experiments/proxy_source_ablation/probe_replay_20260910')
    base=campaign/'out'
    meta=json.loads((campaign/'CAMPAIGN.json').read_text())
    done=sorted(base.glob('*.complete.json'))
    if len(done)!=8:raise ValueError('eight runs required')
    for p,sha in meta['measurement_sha256'].items():
        # Protocol documentation gained a disclosed post-hoc oracle control;
        # the collector, engine, and model execution runner did not change.
        if p.endswith('README.md'):continue
        assert digest(Path(p))==sha,p
    for name,record in meta['prior_environment']['datasets'].items():
        assert digest(Path(record['path']))==record['sha256'],name
    results=[];max_floor_mass=0.;floor_count=0;min_e=1.;max_norm=0.
    zero_target_residual_rows=0
    for path in done:
        r=json.loads(path.read_text());m=json.loads((base/r['manifest']).read_text())
        log=(base/path.name.replace('.complete.json','.log')).read_text()
        for bad in ('Traceback (most recent call last):','PROBE_FLUSH_FAILED','falling back to random','adding random tokens','NCCL timeout'):
            assert bad not in log,(path,bad)
        assert 'Final Decode Throughput:' in log
        assert 'max|probe-engine|=0.000e+00 OK' in log
        assert 'Generation:' in log and '--example' not in r['command']
        assert m['layers']==[56,79] and m['temperature']==1.0 and m['stride']==8
        assert set(map(int,m['seq_steps']))==set(range(1,33))
        sampled={sid:[] for sid in range(1,33)};chunks=[];nrows=0
        for c in m['chunks']:
            file=base/c['file']
            with np.load(file) as z:
                n=len(z['step'])//5;nrows+=len(z['step'])
                np.testing.assert_array_equal(z['position'],np.tile(np.arange(5),n))
                for sid,local in zip(z['seq_id'][::5],z['local_step'][::5]):sampled[int(sid)].append(int(local))
                assert z['p_E'].shape==(5*n,2,32000)
                e=z['p_E'][:,0];p=z['p_T'];q=z['p_D'];reject=~z['is_bonus']
                for a in (z['p_E'],p,q):assert a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
                max_norm=max(max_norm,float(np.abs(z['p_E'].sum(-1)-1).max()),
                             float(np.abs(p.sum(-1)-1).max()),float(np.abs(q[reject].sum(-1)-1).max()))
                mask=e<1e-38
                floor_count+=int(mask.sum());min_e=min(min_e,float(e.min()))
                max_floor_mass=max(max_floor_mass,float((p*mask).sum(-1).max()))
                zero_target_residual_rows+=int((np.maximum(p[reject]-q[reject],0).sum(-1)==0).sum())
                assert (q[~reject]==0).all() and (z['y'][~reject]==-1).all()
                assert (z['y'][reject]>=0).all() and (z['y'][reject]<32000).all()
            chunks.append(dict(file=c['file'],bytes=file.stat().st_size,sha256=digest(file)))
        for sid,values in sampled.items():assert values==list(range(0,m['seq_steps'][str(sid)],8))
        assert nrows==r['n_rows']==m['n_rows'] and sum(map(len,sampled.values()))==r['n_samples']
        replay=json.loads((out/'metrics'/path.name.replace('.complete.json','.json')).read_text())
        assert replay['replay_sha256']==digest(out/'replay.py')
        results.append(dict(dataset=r['dataset'],seed=r['seed'],steps=r['n_steps'],
                            samples=r['n_samples'],rows=nrows,chunks=chunks))
        print('[audited]',path.stem,flush=True)
    assert max_norm<1e-4
    result=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=results,
        max_probability_sum_error=max_norm,min_proxy_probability=min_e,
        proxy_entries_below_log_floor=floor_count,max_target_mass_below_proxy_log_floor=max_floor_mass,
        zero_target_residual_rows=zero_target_residual_rows,
        collector_engine_runner_hashes_unchanged=True,dataset_hashes_unchanged=True,
        replay_hashes_match=True,
        analysis_sha256={p.name:digest(p) for p in out.glob('*.py')})
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    meta.update(finished_utc=result['finished_utc'],totals=json.loads((out/'results.json').read_text())['totals'],
                report=str(out/'REPORT.md'),analysis_sha256=result['analysis_sha256'],
                final_protocol_sha256=digest(campaign/'README.md'),
                protocol_additions='rank permutation oracle added after partial inspection; continuous KL refinement uses training prompts only',
                validation=str(out/'validation.json'))
    (campaign/'CAMPAIGN.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','analysis_sha256')},indent=2))


if __name__=='__main__':main()
