"""Rebuild round-2 tables from successful, uncontended full-corpus runs."""
import json
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_run(path):
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict) or obj.get('status') != 'complete':
        return None
    if obj['args']['limit'] != 0 or len(obj['question_indexes']) != 480:
        raise ValueError(f'Not a full-corpus run: {path}')
    manifest = json.loads((path.parent/'campaign.json').read_text())
    run = next((r for r in manifest if r['name'] == path.stem),None)
    if run is None:
        return None  # Output was flushed, but worker cleanup is still running.
    if run.get('external_pids') or run.get('returncode') != 0:
        raise ValueError(f'Contended or failed run: {path}')
    cell, = obj['cells']
    metrics = cell['metrics']
    outputs = sum(len(o['token_ids']) for o in cell['outputs'])
    count = outputs if obj['args']['mode'] != 'ar' else metrics['decode_total_tokens']
    tps = count/metrics['decode_total_time']
    events = metrics.get('phase_events', [])
    steps = defaultdict(list)
    for e in events: steps[e['step_id']].append(e)
    dense = sum(e['verify_width']+1 for e in events)
    useful = sum(e['valid_k']+1 for e in events)
    return dict(file=str(path.relative_to(ROOT)), tps=tps,
                wall_tps=outputs/cell['summary']['wall_s'],
                al=cell['summary']['al_including_recovery'],
                hit=cell['summary']['cache_hit'],
                output_tokens=outputs, tree_events=cell['summary']['tree_verify_events'],
                events=len(events), steps=len(steps),
                no_proposal_rows=sum(e['valid_k']==0 for e in events),
                logical_dense_queries=dense, useful_queries=useful,
                mean_step_ms=1000*metrics['decode_total_time']/len(steps) if steps else None,
                seed=cell['seed'],batch=cell['batch'], temperature=cell['temperature'],
                prompt_sha256=obj['prompt_sha256'], source_sha256=obj['source_sha256'])


def summarize(rows):
    if not rows: return {'n':0}
    result = {'n':len(rows),'runs':rows}
    for key in ['tps','wall_tps','al','hit']:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(mean=st.mean(values),sd=st.stdev(values) if len(values)>1 else None)
    return result


def paired_gain(a,b):
    by_seed = {r['seed']:r for r in b}
    gains = [(r['tps']/by_seed[r['seed']]['tps']-1)*100
             for r in a if r['seed'] in by_seed]
    return dict(n=len(gains), mean=st.mean(gains) if gains else None,
                sd=st.stdev(gains) if len(gains)>1 else None)


def write_validation_summaries():
    comparison = {}
    for model in ['llama2','llama3']:
        directory = ROOT/f'{model}_tree_full'
        paths = dict(chain=directory/f'{model}_duet-chain_p10_t0_full.json',
                     tree=directory/f'{model}_duet-tree_p11_t0_full.json',
                     host=ROOT/f'{model}_tree_host'/f'{model}_tree_host.json')
        # The host campaign uses its own explicit run name.
        if not paths['host'].exists():
            candidates = [p for p in (ROOT/f'{model}_tree_host').glob(f'{model}_*.json')
                          if json.loads(p.read_text()).get('status')=='complete']
            if len(candidates)!=1:
                raise ValueError(f'Expected one successful host comparison for {model}')
            paths['host'] = candidates[0]
        cells = {k:json.loads(p.read_text())['cells'][0] for k,p in paths.items()}
        def outputs(cell):
            return [o['token_ids'] for o in cell['outputs']]
        tree_out = outputs(cells['tree'])
        host_out = outputs(cells['host'])
        chain_out = outputs(cells['chain'])
        assert len(tree_out)==len(host_out)==len(chain_out)==480
        events_equal = (cells['tree']['metrics']['phase_events']==
                        cells['host']['metrics']['phase_events'])
        audit = json.loads((directory/'chain_tree_audit.json').read_text())['divergences']
        tree = read_run(paths['tree'])
        host = read_run(paths['host'])
        comparison[model] = dict(
            host_exact_outputs=sum(x==y for x,y in zip(tree_out,host_out)),
            host_events_exact=events_equal,
            host_gain_pct=(host['tps']/tree['tps']-1)*100,
            tree_vs_chain_exact_outputs=sum(x==y for x,y in zip(tree_out,chain_out)),
            hf_divergences=len(audit),
            hf_max_token_logit_gap=max(abs(d['ar_logit']-d['duet_logit']) for d in audit),
            hf_both_top2=sum(d['ar_token'] in d['hf_top_ids'][:2] and
                            d['duet_token'] in d['hf_top_ids'][:2] for d in audit),
            hf_both_top5=sum(d['ar_token'] in d['hf_top_ids'][:5] and
                            d['duet_token'] in d['hf_top_ids'][:5] for d in audit))
    (ROOT/'validation').mkdir(exist_ok=True)
    (ROOT/'validation/greedy_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    with (ROOT/'RUN_INVENTORY.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['directory','run','status','recorded_status','returncode','foreign_gpu_pids','seconds'])
        for p in sorted(ROOT.glob('*/campaign.json')):
            for run in json.loads(p.read_text()):
                # Two historical startup crashes left raw status="running"
                # despite a recorded ended timestamp and returncode=1.
                status = 'failed' if run['returncode']!=0 else run['status']
                writer.writerow([p.parent.name,run['name'],status,run['status'],run['returncode'],
                                 json.dumps(run.get('external_pids',[])),
                                 round(run['ended']-run['started'],2)])


def main():
    result={}
    for model in ['llama2','llama3']:
        groups={}
        for directory,variants in [(f'{model}_full',['base','packed','mixed']),
                                   (f'{model}_final',['candidate','ssd'])]:
            for variant in variants:
                rows=[]
                for f in sorted((ROOT/directory).glob(f'{model}_{variant}_s*.json')):
                    row=read_run(f)
                    if row: rows.append(row)
                groups[variant]=summarize(rows)
        for name in ['packed','mixed','candidate']:
            groups[name]['gain_vs_base_pct']=paired_gain(
                groups[name].get('runs',[]),groups['base'].get('runs',[]))
        groups['candidate']['gain_vs_ssd_pct']=paired_gain(
            groups['candidate'].get('runs',[]),groups['ssd'].get('runs',[]))
        groups['greedy_tree']=[]
        for directory in [f'{model}_tree_full',f'{model}_tree_host']:
            for f in sorted((ROOT/directory).glob(f'{model}_*.json')):
                row=read_run(f)
                if row: groups['greedy_tree'].append(row)
        result[model]=groups
    (ROOT/'NUMBERS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Raw-result-derived round-2 numbers','',
           'All main rows: 480 first-turn inputs, B=8, T=0.7, max output=64. TPS mean ± sample SD; n=3 only after all three runs complete.', '',
           '| Model | Variant | n | Decode TPS | AL | Hit |',
           '|---|---|---:|---:|---:|---:|']
    for model,groups in result.items():
        print(model)
        for variant,g in groups.items():
            if isinstance(g,dict) and g.get('n'):
                print(variant,g['n'],g['tps'],g.get('gain_vs_base_pct',''))
                sd=g['tps']['sd']
                tps=f"{g['tps']['mean']:.2f}"+(f" ± {sd:.2f}" if sd is not None else '')
                lines.append(f"| {model} | {variant} | {g['n']} | {tps} | {g['al']['mean']:.4f} | {g['hit']['mean']:.4f} |")
    lines += ['', '## Greedy tree / chain (B=1, one full-corpus run per row)', '',
              '| Run | Decode TPS | AL | Tree events |', '|---|---:|---:|---:|']
    for groups in result.values():
        for r in groups['greedy_tree']:
            lines.append(f"| {r['file']} | {r['tps']:.2f} | {r['al']:.4f} | {r['tree_events']} |")
    (ROOT/'NUMBERS.md').write_text('\n'.join(lines)+'\n')
    write_validation_summaries()


if __name__=='__main__':main()
