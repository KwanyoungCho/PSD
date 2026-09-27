"""Check immutable selection, prompt provenance, snapshot hashes, and sampler code."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CAMPAIGN=ROOT/'ssd/experiments/proxy_source_ablation/probe_training_free_20260912'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    frozen=json.loads((HERE/'frozen.json').read_text())
    assert sha(HERE/'replay.py')==frozen['replay_sha256']
    for name,value in frozen['development_hashes'].items():assert sha(HERE/'development'/name)==value
    created=datetime.fromisoformat(frozen['created_utc']).timestamp()
    confirmation=list((HERE/'confirmation').glob('*.npz'));assert len(confirmation)==3
    assert all(p.stat().st_mtime>created for p in confirmation)
    provenance=json.loads((CAMPAIGN/'datasets.json').read_text());seen=set()
    for value in provenance['source_files'].values():
        path=Path(value['path']);assert sha(path)==value['sha256']
        # Original prompts 1..64 per dataset; include every nonempty source row.
        texts=[json.loads(line)['text'].strip() for line in path.read_text().splitlines() if line.strip()]
        for text in [x for x in texts if x][:64]:seen.add(hashlib.sha256(text.encode()).hexdigest())
    fresh=provenance['stages']['confirmation'];mapping=fresh['mapping']
    assert sha(Path(fresh['file']))==fresh['sha256']
    new={m['text_sha256'] for m in mapping};assert len(new)==96 and not new&seen
    counts={ds:sum(m['dataset']==ds for m in mapping) for ds in ['alpaca','c4','gsm','humaneval']}
    assert set(counts.values())=={24}
    snapshots=[]
    for stage in ['development','confirmation']:
        for done in sorted((CAMPAIGN/'out'/stage).glob('*.complete.json')):
            record=json.loads(done.read_text())
            for name,value in record['snapshot_sha256'].items():assert sha(done.parent/name)==value
            snapshots.append({'stage':stage,'file':done.name,'temperature':record['temperature'],
                              'numseqs':record['numseqs'],'steps':record['n_steps'],'samples':record['n_samples'],
                              'chunks':len(record['snapshot_sha256']),'max_probability_sum_error':record['max_probability_sum_error']})
            print('validated',stage,done.name,flush=True)
    assert len(snapshots)==5
    previous=json.loads((HERE.parent/'shared_review/followup_verify_checks.json').read_text())
    production={name:sha(ROOT/name)==value for name,value in previous['production_code_sha256'].items()}
    assert all(production.values())
    oldcampaign=json.loads((ROOT/'ssd/experiments/proxy_source_ablation/probe_replay_20260910/CAMPAIGN.json').read_text())
    integration={name:sha(ROOT/name)==value for name,value in oldcampaign['measurement_sha256'].items() if name.endswith('.py')}
    assert all(integration.values())
    for name in ['checks.json','runtime_checks.json','lossless_checks.json','runtime_snapshot_checks.json']:
        assert json.loads((HERE/name).read_text())['passed']
    live=[]
    for folder in ['live','live_clean']:
        for done in sorted((HERE/folder).glob('*.complete.json')):
            record=json.loads(done.read_text());path=done.with_name(done.name.replace('.complete.json','.json'))
            data=json.loads(path.read_text());log=path.with_suffix('.log').read_text(errors='replace')
            assert record['runtime_policy_sha256']==sha(HERE/'runtime_policy.py')
            assert data['output_tokens']==4096 and data['prompts']==16 and data['candidate_calls_by_K']
            assert 'Final Decode Throughput:' in log and 'Traceback (most recent call last):' not in log
            live.append({'folder':folder,'file':done.name,'arm':record['arm'],'seed':record['seed'],
                         'data_sha256':sha(path),'log_sha256':sha(path.with_suffix('.log'))})
    if (HERE/'live_clean_summary.json').exists():
        assert json.loads((HERE/'live_clean_summary.json').read_text())['passed']
        assert len(live)==14
    result={'passed':True,'frozen_sha256':sha(HERE/'frozen.json'),'frozen_utc':frozen['created_utc'],
      'confirmation_metric_mtimes_utc':{p.name:datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat() for p in confirmation},
      'development_replay_hashes_unchanged':True,'replay_source_unchanged_since_freeze':True,
      'fresh_unique_prompts':96,'fresh_counts':counts,'overlap_with_previous_prompts':0,
      'raw_snapshot_hashes_rechecked':True,'snapshots':snapshots,
      'live_records':live,'runtime_policy_sha256':sha(HERE/'runtime_policy.py'),
      'production_verification_code_unchanged':production,'integration_and_probe_code_unchanged':integration,
      'scope':'No engine source edits; experiment monkeypatch changes cache candidate graph only. Live results audited separately. Hashes/timing support provenance, not statistical independence of token rows.'}
    (HERE/'audit.json').write_text(json.dumps(result,indent=2))
    print('passed',flush=True)

if __name__=='__main__':main()
