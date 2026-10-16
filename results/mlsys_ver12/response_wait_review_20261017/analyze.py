"""Read existing traces only; never import an engine or start GPU work."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles


def elapsed(e):
    return (e['wall_end_ns']-e['wall_start_ns'])/1e6


def main():
    original = HERE.parent/'chain40_k10_20261016/RESPONSE_WAIT_DIAGNOSIS.json.gz'
    with gzip.open(original, 'rt') as stream:
        diagnosis = json.load(stream)
    runs = []
    for run in diagnosis['runs']:
        base = ('/data/chokwans99/duet-chain40-k10-20261016' if run['mode']=='chain'
                else '/data/chokwans99/duet-local-phase-20261015')
        groups, _, hashes = read_profiles(Path(base)/run['run'])
        details = []
        for row in run['rows']:
            key = row['epoch'], row['step_id']
            draft, target = groups['draft'][key], groups['target'][key]
            previous = groups['target'][(key[0], key[1]-1)]
            def one(events, label):
                found = [e for e in events if e['label']==label]
                assert len(found)==1, (key,label,len(found))
                return found[0]
            # The user's gap: end of the previous target forward including
            # its final head, to the next target Transformer launch. This
            # includes acceptance and request preparation, not just recv.
            points = [one(previous,'final_logits')['wall_end_ns'],
                one(previous,'verify_sample_accept')['wall_end_ns'],
                one(previous,'target_postprocess')['wall_end_ns'],
                one(target,'target_send_request')['wall_start_ns'],
                one(target,'target_send_request')['wall_end_ns'],
                one(target,'target_recv_response_wait')['wall_end_ns'],
                one(target,'graph_pre')['wall_start_ns']]
            assert all(b>=a for a,b in zip(points,points[1:])), (key,points)
            gap_names = ['previous_accept_and_kv_ms','previous_bookkeeping_postprocess_ms',
                'scheduler_and_next_request_input_ms','next_request_send_ms',
                'next_response_block_ms','next_response_decode_and_verify_setup_ms']
            partition = {name:(b-a)/1e6 for name,a,b in zip(gap_names,points,points[1:])}
            gap = (points[-1]-points[0])/1e6
            assert abs(sum(partition.values())-gap)<1e-8
            labels = Counter(e['label'] for e in draft)
            def ms(prefix):
                return sum(elapsed(e) for e in draft if e['label'].startswith(prefix))
            wait = next(e for e in target if e['label']=='target_recv_response_wait')
            # Elapsed interval intersection is not a measured counterfactual gain.
            restore = [e for e in draft if e['label']=='tree_kv_restore']
            restore_overlap = sum(max(0, min(e['wall_end_ns'], wait['wall_end_ns'])
                - max(e['wall_start_ns'], wait['wall_start_ns']))/1e6 for e in restore)
            target_detail = {}
            for label in ('verify_setup','tree_wire_parse_validate','tree_proxy_topology_prepare',
                          'tree_parent_q_select','tree_verify_meta_cpu',
                          'tree_verify_mask_prepare','tree_verify_attention_buffers'):
                target_detail[f'next_{label}_in_gap_ms'] = sum(
                    max(0,min(e['wall_end_ns'],points[-1])-max(e['wall_start_ns'],points[-2]))/1e6
                    for e in target if e['label']==label)
            for label in ('verify_accept_prep','chain_accept','tree_accept_walk','tree_kv_commit'):
                target_detail[f'previous_{label}_ms'] = sum(elapsed(e) for e in previous if e['label']==label)
            details.append(dict(row,**partition,
                **target_detail,
                target_forward_to_forward_gap_ms=gap,
                after_accept_to_next_forward_gap_ms=(points[-1]-points[1])/1e6,
                previous_tree_restore_present=bool(labels['tree_kv_restore']),
                tree_kv_restore_ms=ms('tree_kv_restore'),
                tree_kv_restore_overlap_response_ms=restore_overlap,
                tree_hit_pack_ms=ms('tree_hit_pack_served'),
                tree_hit_rerank_inclusive_ms=ms('tree_hit_rerank'),
                tree_hit_q_gather_ms=ms('tree_hit_parent_q_gather'),
                draft_send_fused_ms=ms('draft_send_fused'),
                draft_send_q_ms=ms('draft_send_parent_q')+ms('draft_send_chain_q'),
                target_metadata_read_ms=sum(elapsed(e) for e in target
                    if e['label']=='target_response_meta_read')))
        summary = []
        for state in (1,2,0,'all'):
            subset = [r for r in details if state=='all' or r['state']==state]
            fields = [k for k in subset[0] if k.endswith('_ms')]
            summary.append(dict(state=state,n=len(subset),
                previous_tree_restore_steps=sum(r['previous_tree_restore_present'] for r in subset),
                **{k:mean(r[k] for r in subset) for k in fields}))
        runs.append(dict(mode=run['mode'],run=run['run'],summary=summary,
                         trace_sha256=hashes,rows=details))
    source_files = ['ssd/ssd/engine/step.py','ssd/ssd/engine/speculator_async.py',
        'ssd/ssd/engine/draft_runner.py','ssd/ssd/engine/helpers/p2_tree.py',
        'ssd/ssd/engine/verifier.py','ssd/ssd/engine/model_runner.py',
        'ssd/ssd/engine/helpers/batch_tree_draft.py','ssd/ssd/utils/async_helpers/nccl_pack.py']
    result = dict(status='complete',mode='existing_trace_and_static_code_review',
        gpu_experiments_run=False,runtime_changed=False,
        sample_policy='Original diagnostic populations unchanged; profile ON, no new 2x exclusion',
        limits='Nested spans and cross-device spans are not additive. These are observational profile intervals, not isolated kernel times or verified improvement amounts.',
        diagnosis_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        reviewed_source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
        runs=runs)
    raw = json.dumps(result,ensure_ascii=False).encode()
    (HERE/'DETAILS.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    summary = dict(result,runs=[{k:v for k,v in r.items() if k!='rows'} for r in runs])
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    for run in runs:
        for s in run['summary']:
            if s['state'] in (1,2):
                keys=['state','n','target_forward_to_forward_gap_ms',*gap_names,
                    'response_wait_ms','previous_ps_overlap_response_ms',
                    'request_receive_ms','tree_kv_restore_ms','chain_or_tree_lookup_ms',
                    'tree_hit_pack_ms','tree_hit_q_gather_ms','target_q_receive_ms']
                print(run['mode'],run['run'],{k:round(s[k],4) if isinstance(s[k],float) else s[k] for k in keys},flush=True)


if __name__=='__main__':
    main()
