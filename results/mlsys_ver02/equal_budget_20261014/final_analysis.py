"""Final question-clustered analysis; never modifies experiment runtime/raw data.

Two server/seed replicas repeat the same 480 questions, not 960 independent
questions. Pooled intervals resample question IDs after averaging replicas.
"""
import csv
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze

BASE = Path('/data/chokwans99/equal-tree-paper-full-20261014')
FOLDERS = {
    'local': [BASE / n for n in ('local_45_paper', 'local_67_paper', 'local_45_retry_external')],
    'remote': [BASE / 'remote_mirror' / n for n in ('nova_23_paper', 'nova_45_paper', 'nova_67_paper', 'nova_23_retry_disk', 'nova_67_retry_disk')],
}
ARMS = analyze.EXPECTED
PAIRS = [(ARMS[0], n) for n in ARMS[1:]] + [
    ('last_c3_m32', 'frontier_c3_m32'), ('last_c8_m32', 'frontier_c8_m32'),
    ('last_c8_m64', 'frontier_c8_m64'),
    ('last_c3_m32', 'last_c8_m32'), ('frontier_c3_m32', 'frontier_c8_m32'),
    ('last_c8_m32', 'last_c8_m64'), ('frontier_c8_m32', 'frontier_c8_m64'),
    ('last_c3_m32', 'last_c8_m64'), ('frontier_c3_m32', 'frontier_c8_m64'),
]


def collect_run(r):
    qstats = analyze.question_stats(r)
    contributions, output_lengths, guard, source_pop, uid_order = defaultdict(list), defaultdict(list), set(), defaultdict(int), []
    depth_checks = dict(events=0, violations=0, al_violations=0, observed_max_depth=0, maximum_al=0)
    c, m = r['args']['siblings'], r['args']['p1_verify']
    theoretical_depth = min(8, (m - 1) // c + 1)
    for group in r['groups']:
        assert group['n'] == 1
        req = group['request_records'][0]
        q = str(req['question_id'])
        uid_order.append(req['paper_uid'])
        output_lengths[q].append(req['completion_tokens'])
        if req['finish_reason'] == 'context_limit_guard':
            guard.add(q)
        sums = np.zeros(3); total = 0
        for s in group['steps']:
            if s['cap'] or s['clipped']:
                continue
            assert s['batch'] == 1 and len(s['events']) == 1
            e = s['events'][0]; src = e['source']
            sums[src] += e['accepted_len']; total += 1; source_pop[src] += 1
            if src:
                depth_checks['events'] += 1
                depth_checks['violations'] += e['tree_max_depth'] > theoretical_depth
                depth_checks['al_violations'] += e['accepted_len'] > theoretical_depth + 1
                depth_checks['observed_max_depth'] = max(depth_checks['observed_max_depth'], e['tree_max_depth'])
                depth_checks['maximum_al'] = max(depth_checks['maximum_al'], e['accepted_len'])
        if total:
            contributions[q].append(sums / total)
    contrib = {q: np.mean(v, axis=0) for q,v in contributions.items()}
    for q in qstats:
        assert abs(contrib[q].sum() - qstats[q][0]) < 1e-10
    assert not depth_checks['violations'] and not depth_checks['al_violations']
    groups = {str(v['question_id']):v['group'] for v in r['input_metadata']}
    return dict(qstats=qstats, contributions=contrib, guard=guard, groups=groups,
        output_lengths={q:np.mean(v) for q,v in output_lengths.items()},
        depth_bound=theoretical_depth, depth_checks=depth_checks,
        source_population=dict(source_pop), uid_order=uid_order)


def paired_summary(a, b, keys, seed=143):
    keys = sorted(set(keys) & a.keys() & b.keys())
    va = np.array([a[k] for k in keys]); vb = np.array([b[k] for k in keys])
    valid = np.isfinite(va) & np.isfinite(vb); va, vb = va[valid], vb[valid]
    delta = vb - va
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(len(delta), size=(5000,len(delta)))].mean(1)
    return dict(questions=len(delta), before=float(va.mean()), after=float(vb.mean()),
        delta=float(delta.mean()), relative_percent=float(100*(vb.mean()/va.mean()-1)),
        ci95=np.quantile(draws,[.025,.975]).tolist(),
        positive_questions=int((delta>0).sum()), negative_questions=int((delta<0).sum()),
        equal_questions=int((delta==0).sum()))


def main():
    data={}; raw_manifest=[]; ref_hashes=None
    for host, folders in FOLDERS.items():
        data[host]={}; found={}
        for d in folders:
            for run in json.loads((d/'campaign.json').read_text()):
                if run.get('status')=='complete' and not run.get('returncode') and not run.get('external_pids'):
                    assert run['name'] not in found
                    found[run['name']]=d/(run['name']+'.json')
        assert set(found)==set(ARMS), (host,set(found))
        for name in ARMS:
            path=found[name]; blob=path.read_bytes(); sha=hashlib.sha256(blob).hexdigest()
            r=json.loads(blob); del blob
            analyze.validate(r)
            hashes=(r['source_hashes'],r['adapter_hashes'])
            if ref_hashes is None:ref_hashes=hashes
            assert hashes==ref_hashes, 'Source differs, including across servers'
            assert r['equal_budget']['calibration'] is False
            assert r['equal_budget']['root_weighting']=='weighted'
            x=collect_run(r);data[host][name]=x
            raw_manifest.append(dict(host=host,name=name,path=str(path),sha256=sha,
                seed=r['args']['seed'],turns=len(r['outputs']),questions=len(x['qstats']),
                output_tokens=sum(map(len,r['outputs'])),context_guard_questions=len(x['guard']),
                depth_bound=x['depth_bound'],depth_checks=x['depth_checks']))
            print('collected',host,name,len(x['qstats']),flush=True)
            del r;gc.collect()
    questions=sorted(data['local'][ARMS[0]]['qstats'])
    assert len(questions)==480
    for h in data:
        for x in data[h].values():assert sorted(x['qstats'])==questions
    guard_union=set().union(*(x['guard'] for d in data.values() for x in d.values()))
    clean=set(questions)-guard_union
    combined={n:{q:np.mean([data[h][n]['qstats'][q][0] for h in data]) for q in questions} for n in ARMS}
    comparisons={}
    for a,b in PAIRS:
        key=a+' -> '+b
        item={h:paired_summary({q:v[0] for q,v in d[a]['qstats'].items()},
                {q:v[0] for q,v in d[b]['qstats'].items()}, questions) for h,d in data.items()}
        item['paired_question_average_over_replicas']=paired_summary(combined[a],combined[b],questions)
        item['without_any_context_guard_question']=paired_summary(combined[a],combined[b],clean)
        comparisons[key]=item
    rows=[]; contributions={}; sensitivity={}
    for h,d in data.items():
        contributions[h]={};sensitivity[h]={}
        for n,x in d.items():
            contributions[h][n]=np.mean(list(x['contributions'].values()),axis=0).tolist()
            sensitivity[h][n]=dict(questions=len(clean),al=float(np.mean([x['qstats'][q][0] for q in sorted(clean)])))
            for q,v in x['qstats'].items():
                rows.append(dict(host=h,arm=n,question_id=q,group=x['groups'][q],
                    **dict(zip(analyze.NAMES,map(float,v))),
                    **dict(zip(('miss_contribution','p1_contribution','p2_contribution'),map(float,x['contributions'][q]))),
                    mean_output_tokens=float(x['output_lengths'][q]),context_guard=int(q in x['guard'])))
    with (HERE/'QUESTION_METRICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(state='complete',replicas=2,unique_questions=480,total_runs=14,total_turns=7840,
        interval_method='5000 paired question bootstrap draws; replicas averaged within each question; unadjusted exploratory 95% intervals; not a seed-generalization interval',
        comparisons=comparisons,phase_contributions=contributions,
        guard_sensitivity=dict(excluded_questions=sorted(guard_union),remaining=len(clean),arms=sensitivity),
        raw_manifest=raw_manifest,source_hashes=ref_hashes[0],adapter_hashes=ref_hashes[1],
        limitations=['Online contexts differ across methods.',
                    'No uniform-root-weighting or C3 protected-backbone ablation.',
                    'No target oracle for discarded nodes and no valid rerank replay.',
                    'Hardware and seed are confounded; one seed per server.',
                    'Forest trace records all exports, not hit-root identities.'])
    (HERE/'FINAL_ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',len(rows),'question-arm-replica rows',len(clean),'guard-free questions',flush=True)


if __name__=='__main__':main()
