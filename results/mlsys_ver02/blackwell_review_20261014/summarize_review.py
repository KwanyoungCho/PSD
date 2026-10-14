"""Verify saved evidence against this checkout, including the final source hashes."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
# HERE = repository/results/mlsys_ver02/<review>

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    r=HERE/'checks_release'
    checks=json.loads((r/'full_final/CHECKS.json').read_text())
    provenance=json.loads((r/'full_final/CHECKS_PROVENANCE.json').read_text())
    mismatch=[k for k,v in provenance['source_hashes'].items() if sha(ROOT/k)!=v]
    regression=dict(suites=len(checks),tests=sum(x['tests'] for x in checks),
        failures=sum(x['failures'] for x in checks),errors=sum(x['errors'] for x in checks),
        skipped=sum(x['skipped'] for x in checks),source_unchanged=provenance['source_unchanged'],
        local_source_mismatch=mismatch,passed=bool(checks) and all(x['returncode']==0 for x in checks)
            and provenance['source_unchanged'] and not mismatch)
    m=r/'models';ledger=json.loads((m/'status.json').read_text())
    outputs={};model=[]
    for row in ledger:
        path=m/(row['name']+'.json');x=json.loads(path.read_text())
        diff=[k for k,v in x['source_hashes'].items() if sha(ROOT/k)!=v]
        ids=[v for g in x['groups'] for v in g['outputs']]
        outputs[row['name']]=ids
        model.append(dict(name=row['name'],status=x['status'],returncode=row.get('returncode'),
            output_count=len(ids),output_lengths=list(map(len,ids)),source_mismatch=diff,
            passed=x['status']=='complete' and row.get('returncode')==0 and len(ids)==4
                and not diff and all(0<len(v)<=32 for v in ids)))
    parity=[]
    for b in (1,2):
        a=outputs.get(f'ar_b{b}_t0',[]);z=outputs.get(f'tree_b{b}_t0',[])
        parity.append(dict(batch=b,matched=sum(x==y for x,y in zip(a,z)),
                           pairs=len(a),passed=len(a)==len(z)==4 and a==z))
    before_after=[]
    for v in (32000,128256):
        item=dict(vocab=v)
        for arm in ('before','after'):
            files=list((HERE/'checks_final').glob(f'proxy_*_{arm}.json'))
            for path in ('dense','flat'):
                vals=[row['median_ms'] for f in files for row in json.loads(f.read_text())['rows']
                      if row['vocab']==v and row['path']==path]
                item[f'{arm}_{path}_ms']=sum(vals)/len(vals)
        item['flat_additional_ms']=item['after_flat_ms']-item['before_flat_ms']
        item['flat_vs_dense_after_percent']=100*(1-item['after_flat_ms']/item['after_dense_ms'])
        before_after.append(item)
    result=dict(state='complete_local_review' if regression['passed'] and len(model)==6
        and all(x['passed'] for x in model+parity) else 'needs_attention',
        scope='SM86 local review, not actual SM120/70B release certification',
        regression=regression,model_functional=model,greedy_ar_parity=parity,
        normalization_component=before_after,
        pending=['Actual SM120 70B B1/B2/B8 and maximum warmup geometry',
                 'Server-specific Blackwell report/adapter integration',
                 'Qwama full generation on its declared target model',
                 'Merge into ongoing AL worktree after source-frozen campaign completes'])
    (HERE/'VERIFICATION.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    result=main();print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['state']=='complete_local_review' else 1)
