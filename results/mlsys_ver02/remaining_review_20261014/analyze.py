"""Aggregate the ver02 incremental payload review without modifying raw."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'ssd/bench'))
from mlsys_tree_results import compare, summarize


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def telemetry(folder):
    campaign = read(folder / 'campaign.json')
    result = {}
    for job in campaign:
        if job['status'] != 'complete' or job['returncode'] or job['external_pids']:
            raise RuntimeError(f'Invalid campaign job: {job}')
        rows = [json.loads(s) for s in (folder/(job['name']+'.telemetry.jsonl')).read_text().splitlines()]
        by = {}
        for row in rows:
            for line in row['gpu'].splitlines():
                cols = [float(x) for x in line.split(',')]
                index = str(int(cols[0]))
                by.setdefault(index, []).append(cols[1:])
        result[job['name']] = dict(external_pids=job['external_pids'],
            gpu={k:dict(samples=len(v), clock_mean_mhz=statistics.mean(x[2] for x in v),
                        temperature_max_c=max(x[4] for x in v)) for k,v in by.items()})
    return result


def main():
    checks = read(HERE / 'checks/CHECKS.json')
    provenance = read(HERE / 'checks/CHECKS_PROVENANCE.json')
    assert len(checks) == 25 and sum(c['tests'] for c in checks) == 265
    assert all(c['returncode']==c['errors']==c['failures']==c['skipped']==0 for c in checks)
    assert provenance['source_unchanged']
    assert all(sha(ROOT/k)==v for k,v in provenance['source_hashes'].items())
    report = dict(check_suites=25, tests=265, errors=0, source_unchanged=True,
                  release_commit='0d5374289b8112db53c8f82ea7eafadc0c30a696', components=[], full={})
    for name, path in [('repeat1','/data/chokwans99/ver02-payload-component1-20261014.json'),
                       ('repeat2','/data/chokwans99/ver02-payload-component2-20261014/repeat2.json'),
                       ('repeat3','/data/chokwans99/ver02-payload-component3-20261014/repeat3.json')]:
        if not Path(path).exists(): continue
        raw = read(path)
        assert raw['status']=='complete' and raw['source_unchanged']
        assert all(provenance['source_hashes'][k]==v for k,v in raw['source_hashes'].items())
        dest = HERE/(name+'.json')
        dest.write_bytes(Path(path).read_bytes())
        cells = [{k:v for k,v in row.items() if k!='samples_ms'} for row in raw['cells']]
        report['components'].append(dict(name=name, raw_sha256=sha(path), cells=cells,
            usable_performance=name!='repeat2',
            caveat=('External GPU6 PID2792425 entered during measurement; excluded from performance evidence.'
                    if name=='repeat2' else 'No continuous GPU telemetry for initial exploratory component.'
                    if name=='repeat1' else 'Telemetry-monitored component run.')))
        if name=='repeat3':
            report['components'][-1]['telemetry']=telemetry(Path(path).parent)
    runtime_changed = None
    local=Path('/data/chokwans99/ver02-payload-full-l2-20261014')
    if (local/'campaign.json').exists():
        a,b=[read(local/(arm+'.json')) for arm in ('base','candidate')]
        report['excluded_local_comparison']=dict(reason='External GPU5 PID2794734 during candidate startup; source/output divergence also observed. No causal online speed claim.',
            campaign=read(local/'campaign.json'),base=summarize(a),candidate=summarize(b),comparison=compare(a,b))
    remote_sources=read(HERE/'REMOTE_SOURCE_HASHES.json')
    for model in ('l2','l3'):
        folder = Path(f'/data/chokwans99/ver02-payload-nova-{model}-20261014')
        if not (folder/'campaign.json').exists(): continue
        campaign = read(folder/'campaign.json')
        if len(campaign)!=2: continue
        a, b = [read(folder/(arm+'.json')) for arm in ('base','candidate')]
        assert a['status']==b['status']=='complete'
        assert a['args']==b['args'] or {k:v for k,v in a['args'].items() if k!='output'}=={k:v for k,v in b['args'].items() if k!='output'}
        assert a['paper_commit']==remote_sources['base']['commit']
        assert a['source_hashes']==remote_sources['base']['runtime']
        assert b['source_hashes']==remote_sources['candidate']['runtime']
        # The transported source-only repository has no .gitignore; Python
        # creates untracked bytecode caches. No tracked baseline source edits.
        assert all(line.startswith('?? ') and line.endswith('__pycache__/')
                   for line in a['paper_dirty'].splitlines())
        assert a['adapter_hashes']==b['adapter_hashes']
        assert len(a['outputs'])==len(b['outputs'])==960
        assert a['prompt_ids']==b['prompt_ids']
        assert a['question_indexes']==b['question_indexes']
        changed=[k for k,v in a['source_hashes'].items() if b['source_hashes'][k]!=v]
        assert changed==['ssd/ssd/engine/helpers/batch_tree_verify.py'], changed
        assert all(provenance['source_hashes'][k]==v for k,v in b['source_hashes'].items())
        report['full'][model]=dict(base=summarize(a),candidate=summarize(b),
            comparison=compare(a,b), runtime_changed=changed, telemetry=telemetry(folder),
            provenance='Nova/3090 snapshot commit; all 79 baseline runtime SHA256s equal original 0d53742 and candidate equals local tested source. Untracked bytecode caches retained in raw git status.',
            raw={arm:dict(path=str(folder/(arm+'.json')),sha256=sha(folder/(arm+'.json')))
                 for arm in ('base','candidate')})
        report['full'][model]['comparison']['same_step_events']=(
            [s['events'] for s in a['steps']]==[s['events'] for s in b['steps']])
    clean_component=any(c['name']=='repeat3' for c in report['components'])
    report['status']='complete' if len(report['full'])==2 and clean_component else 'partial'
    (HERE/'RESULTS.json').write_text(json.dumps(report,indent=2))
    print(report['status'], report['check_suites'], report['tests'])
    for k,v in report['full'].items(): print(k, v['comparison'])


if __name__=='__main__': main()
