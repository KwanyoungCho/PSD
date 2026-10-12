"""Audit recorded populations, coverage and matching implementation controls."""
import argparse,json,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'ssd/bench'))
from mlsys_step_metrics import performance_summary
p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args()
rows=json.loads((HERE/'MODEL_RESULTS.json').read_text());checks=[];hashes={};core_versions={}
for row in rows:
    path=HERE/row['path'];d=json.loads(path.read_text())
    core={k:v for k,v in d['runtime_source_sha256'].items() if k.startswith('ssd/ssd/') or k in ('ssd/bench/mlsys_coverage.py','ssd/bench/mlsys_step_metrics.py')}
    digest=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()
    core_versions.setdefault(digest,[]).append(row['name'])
    for c in d['cells']:
        m=c['metrics'];fresh=performance_summary(m,c['batch'])
        assert fresh==c['summary']['performance'],row['name']
        assert len(m['target_verify_times'])==len(m['decode_steps'])
        assert sum(s['emitted_tokens'] for s in m['decode_steps'])==m['decode_total_tokens']
        assert sum(len(o['token_ids']) for o in c['outputs'])==m['decode_total_tokens']
        assert all(s['batch_size']<=c['batch'] for s in m['decode_steps'])
        assert len(c['outputs'])==len(d['prompt_ids'])
    checks.append(dict(name=row['name'],cells=len(d['cells']),performance_recomputed=True,
                       actual_output_accounting=True,requests_completed=len(d['prompt_ids']),
                       dtype=[d.get('target_dtype'),d.get('draft_dtype')]))
    hashes[row['name']]=[hashlib.sha256(json.dumps([o['token_ids'] for o in c['outputs']]).encode()).hexdigest()
                         for c in d['cells']]
parity=[]
for name,value in hashes.items():
    if '_graph1_' in name:
        other=name.replace('_graph1_','_trim1_')
        if other in hashes:parity.append(dict(graph1=name,trim1=other,all_output_tokens_equal=value==hashes[other],
                                             passes=len(value)))
greedy=[]
for name,value in hashes.items():
    if '_greedy_' in name and '_host0_' in name:
        other=name.replace('_host0_','_host1_')
        if other in hashes:greedy.append(dict(control=name,prototype=other,
            all_output_tokens_equal=value==hashes[other],passes=len(value)))
chain=[]
for name,value in hashes.items():
    if '_chain_' in name and '_general_' in name:
        other=name.replace('_general_','_fast_')
        if other in hashes:chain.append(dict(general=name,fast=other,
            all_output_tokens_equal=value==hashes[other],passes=len(value)))
external=[]
for p in HERE.glob('*/campaign.json'):
    for row in json.loads(p.read_text()):
        if row.get('external_pids'):external.append(dict(path=str(p.relative_to(HERE)),name=row['name'],pids=row['external_pids']))
counts={kind:sum(r['kind']==kind for r in rows) for kind in ('profile','full')}
if a.require_complete:
    assert counts==dict(profile=58,full=32),counts
    assert not external,external
    assert len(core_versions)==1,core_versions
    assert len(parity)==16 and all(p['all_output_tokens_equal'] and p['passes']==2 for p in parity),parity
    assert len(greedy)==8 and all(p['all_output_tokens_equal'] and p['passes']==2 for p in greedy),greedy
    assert len(chain)==8 and all(p['all_output_tokens_equal'] and p['passes']==2 for p in chain),chain
    probe=json.loads((HERE/'greedy_host_micro.json').read_text())
    assert probe['status']=='complete' and len(probe['cases'])==32
    assert all(c['parity'] for c in probe['cases'])
    for model in ('llama2','llama3'):
        d=json.loads((HERE/f'{model}_prefix.json').read_text());assert d['status']=='complete' and len(d['cases'])==18
out=dict(status='complete' if counts==dict(profile=58,full=32) else 'partial',counts=counts,core_source_versions=core_versions,
         source_note='One on-disk inference implementation. Greedy host1 explicitly substitutes acceptance replay in memory via the benchmark adapter; source identity is not execution identity.',
         population_and_accounting=checks,graph_trim_output_controls=parity,
         greedy_adapter_output_controls=greedy,chain_fast_output_controls=chain,
         external_gpu_processes=external)
(HERE/'AUDIT.json').write_text(json.dumps(out,indent=2))
print('Audited',len(rows),'runs;',counts,'output controls',sum(p['all_output_tokens_equal'] for p in parity),'/',len(parity),'external',len(external))
