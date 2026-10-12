"""Attribute target/draft spans against actual paper-checkout diagnostic runs.

CUDA event spans include launch gaps. CPU spans can wait for earlier GPU work;
neither is labelled pure kernel time. Never sum nested labels or phase medians.
"""
import collections
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent


def stats(v):
    if not v:
        return None
    s = sorted(v)
    return dict(n=len(s), mean=statistics.mean(s), median=statistics.median(s),
                p10=s[int(.1*(len(s)-1))], p90=s[int(.9*(len(s)-1))])


def groups(events, marker):
    out = []
    for e in events:
        if e['label'] == marker and e.get('step_id') == 1:
            out.append(collections.defaultdict(lambda: collections.defaultdict(list)))
        if out and e.get('step_id') is not None:
            out[-1][e['step_id']][e['label']].append(e)
    return out


def ends(labels, names, key, how):
    vals = [e[key] for name in names for e in labels.get(name, [])]
    return how(vals) if vals else None


def analyze(raw):
    d = json.loads(raw.read_text())
    if d['status'] != 'complete':
        raise ValueError(f'Incomplete: {raw}')
    name = raw.stem
    fs = R/'profiles'/name
    tf = list(fs.glob('*target_rank0*.json')); df = list(fs.glob('*draft*.json'))
    assert len(tf) == len(df) == 1, (name, tf, df)
    te = json.loads(tf[0].read_text()); de = json.loads(df[0].read_text())
    unified = name == 'tree_unified'
    tg = groups(te, 'target_send_request')
    dg = groups(de, 'batch_kv_restore' if unified else 'draft_recv_request')
    rg = d['warm_groups']+d['groups']
    assert len(tg) == len(dg) == len(rg), (name, len(tg), len(dg), len(rg))
    samples = collections.defaultdict(list)
    labels_out = collections.defaultdict(list)
    timeline = []
    counts = collections.Counter()
    for gi, (g, target, draft) in enumerate(zip(rg, tg, dg)):
        if gi < len(d['warm_groups']):
            continue
        caps = {sid for lab in (target, draft) for sid, x in lab.items()
                if any('capture' in label for label in x)}
        excluded = caps | {sid+1 for sid in caps}
        for s in g['steps']:
            sid = s['events'][0]['step_id']
            # A uniform steady window, not selective outlier deletion.
            if s['batch'] != d['args']['batch'] or s['cap'] or s['clipped'] or sid <= 19 or sid in excluded:
                counts['excluded'] += 1
                continue
            counts['included'] += 1
            t, dr = target[sid], draft[sid]
            source = s['events'][0]['source'] if s['batch'] == 1 else 'mixed_batch'
            def add(k, value):
                if value is not None:
                    samples[k].append(value)
                    samples[f'{k}/source{source}'].append(value)
                    if source in (1,2):samples[f'{k}/hit'].append(value)
                    if s['batch']==1:
                        vk=s['events'][0]['valid_k']
                        samples[f'{k}/source{source}/vk{vk}'].append(value)
            add('verify_wall_ms', sum(s['verify'])*1000)
            add('step_wall_ms', s['seconds']*1000)
            for device, lab in [('target', t), ('draft', dr)]:
                for k, events in lab.items():
                    labels_out[f'{device}/{k}/cuda_span_ms'].append(sum(e['cuda_ms'] for e in events))
                    labels_out[f'{device}/{k}/host_ms'].append(sum((e['cpu_dispatch_end_ns']-e['cpu_dispatch_start_ns'])/1e6 for e in events))
            mapping = {
                'target_pre_ms': (t, ['batch_target_pre' if unified else 'graph_pre']),
                'target_post_ms': (t, ['batch_target_post' if unified else 'graph_post']),
                'target_accept_commit_ms': (t, ['batch_tree_accept','batch_tree_commit'] if unified else ['verify_sample_accept']),
                'target_tree_walk_ms': (t, ['batch_tree_accept'] if unified else ['tree_accept_walk']),
                'target_chain_accept_ms': (t, ['chain_accept']),
                'draft_glue_ms': (dr, ['batch_glue' if unified else 'glue']),
                'draft_p1_replay_ms': (dr, ['batch_p1_replay' if unified else 'p1_graph_replay']),
                'draft_p2_replay_ms': (dr, ['batch_p2_replay' if unified else 'p2_graph_replay']),
                'target_wait_ms': (t, ['target_spec_wait']),
            }
            for k, (lab, names) in mapping.items():
                ev = [e for n in names for e in lab.get(n, [])]
                if ev:
                    add(k, sum(e['cuda_ms'] for e in ev))
            if unified:
                p1s = ends(dr,['batch_p1_total'],'wall_start_ns',min)
                p1e = ends(dr,['batch_p1_total'],'wall_end_ns',max)
                p2s = ends(dr,['batch_p2_total'],'wall_start_ns',min)
                p2e = ends(dr,['batch_p2_total'],'wall_end_ns',max)
                ready = ends(dr,['batch_proxy_receive'],'wall_end_ns',max)
            else:
                # phase1_build is absent on paper tree-hit steps; phase2_build
                # ends BEFORE the tree replay. Use explicit envelope boundaries.
                p1s = ends(dr,['p1_root_build','phase1_build'],'wall_start_ns',min)
                p1e = ends(dr,['proxy_wait'],'wall_start_ns',min)
                p2s = ends(dr,['proxy_wait'],'wall_end_ns',max)
                p2e = ends(dr,['merge_cache'],'wall_end_ns',max)
                ready = None
            for k, a, b in [('draft_p1_total_ms',p1s,p1e),('draft_p2_total_ms',p2s,p2e)]:
                if a is not None and b is not None:
                    assert b >= a, (name, sid, k)
                    add(k,(b-a)/1e6)
            send = ends(t,['proxy_send_enqueue'],'wall_end_ns',max)
            final = ends(t,['batch_target_final_logits' if unified else 'final_logits'],'wall_end_ns',max)
            nxt = ends(target.get(sid+1,{}),['target_send_request'],'wall_start_ns',min)
            # Original path has no independent receiver-ready event: use sender
            # enqueue completion as an explicitly labelled proxy-ready estimate.
            if ready is None:
                ready = send
            for k, deadline, finish in [('p1_proxy_slack_ms',ready,p1e),('p2_final_slack_ms',final,p2e),('p2_next_request_slack_ms',nxt,p2e)]:
                if deadline is not None and finish is not None:
                    add(k,(deadline-finish)/1e6)
            if p1s is not None and p2e is not None and final is not None:
                timeline.append(dict(group=gi-len(d['warm_groups']),step=sid,source=source,
                                     p1_start=p1s,p1_end=p1e,p2_start=p2s,p2_end=p2e,
                                     proxy_ready_estimate=ready,target_final=final,next_request=nxt,
                                     target_pre_start=ends(t,['batch_target_pre' if unified else 'graph_pre'],'wall_start_ns',min),
                                     target_pre_end=ends(t,['batch_target_pre' if unified else 'graph_pre'],'wall_end_ns',max),
                                     target_post_start=ends(t,['batch_target_post' if unified else 'graph_post'],'wall_start_ns',min),
                                     target_post_end=ends(t,['batch_target_post' if unified else 'graph_post'],'wall_end_ns',max)))
    return dict(raw=str(raw.relative_to(R)), kwargs=d['kwargs'], args=d['args'], counts=counts,
                summary={k:stats(v) for k,v in samples.items()},
                labels={k:stats(v) for k,v in labels_out.items()}, timeline=timeline)


def main():
    arms={}
    for folder in ('tree_r0','tree_r1','chain','greedy'):
        for raw in sorted((R/folder).glob('*_profile.json')):
            if json.loads(raw.read_text()).get('status')=='complete':
                arms[raw.stem]=analyze(raw)
    out=dict(scope='Separate 24-prompt diagnostic runs. Exclude first19/cap/clip/tail/capture+next. CUDA spans include launch gaps; nested/overlapping spans must not be added. Phase envelopes include preparation, not just model forward. Cross-device CPU timeline anchors are approximate.', arms=arms)
    (R/'COMPONENTS.json').write_text(json.dumps(out,indent=2))
    for name,a in arms.items():
        print(name,{k:round(v['median'],4) for k,v in a['summary'].items() if '/' not in k})

if __name__=='__main__':main()
