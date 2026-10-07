"""Rebuild the raw experiment inventory and per-cell numbers, retaining failures."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    inventory=[];cells=[]
    for manifest in sorted(ROOT.glob('*/campaign.json')):
        for run in json.loads(manifest.read_text()):
            path=manifest.parent/(run['name']+'.json')
            result=json.loads(path.read_text()) if path.exists() else {}
            status=('timeout' if run.get('timeout') else 'failed' if run['returncode'] else
                    'complete' if result.get('status')=='complete' else 'incomplete')
            record=dict(directory=manifest.parent.name,name=run['name'],status=status,
                raw_status=result.get('status','missing'),returncode=run['returncode'],
                foreign_gpu_pids=run['external_pids'],git_commit=result.get('git_commit'),
                raw=str(path.relative_to(ROOT)),cells=len(result.get('cells',[])))
            inventory.append(record)
            for index,c in enumerate(result.get('cells',[])):
                events=c['metrics'].get('phase_events',[])
                times=c['metrics'].get('target_verify_times',[])
                row=dict(**record,cell=index,batch=c['batch'],temperature=c['temperature'],seed=c['seed'],
                    prompts=len(result['prompt_ids']),input_cap=result['args']['input_cap'],
                    output_cap=result['args']['max_new_tokens'],mode=result['args']['mode'],
                    p1_tree=result['args']['p1_tree'],preemptions=c['metrics'].get('preemptions',0),
                    batches_observed=sorted({e['batch_size'] for e in events}),
                    physical_verify_widths=sorted({e['physical_verify_width'] for e in events if 'physical_verify_width' in e}),
                    target_verify_median_ms=statistics.median(times)*1000 if times else None,
                    target_peak_allocated_gib=c['target_peak_allocated_bytes']/2**30,
                    **c['summary'])
                cells.append(row)
    (ROOT/'NUMBERS.json').write_text(json.dumps(dict(inventory=inventory,cells=cells),indent=2))
    if inventory:
        with (ROOT/'RUN_INVENTORY.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
    lines=['# Raw 실험 집계','',
        '완료된 campaign만 성능 근거로 사용한다. 실패 이전의 cell은 진단 자료다. '
        '각 cell의 입력 수와 길이, batch 및 phase 설정이 다른 결과를 직접 비교하지 않는다.','',
        '| 폴더 | 모드 | P1 tree | B | T | Seed | 문항 | 출력 cap | Decode TPS | AL | Tree events | Preempt |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        if c['status']!='complete':continue
        al=c['al_including_recovery'];al='—' if al is None else f'{al:.4f}'
        lines.append(f"| [{c['directory']}]({c['raw']}) | {c['mode']} | {c['p1_tree']} | {c['batch']} | {c['temperature']} | {c['seed']} | {c['prompts']} | {c['output_cap']} | {c['decode_tps']:.2f} | {al} | {c['tree_verify_events']} | {c['preemptions']} |")
    lines+=['','Seed가 여러 개인 한 프로세스 실행에서는 target RNG만 cell마다 재설정한다. '
        'Draft RNG는 프로세스 시작 시 SSD_SEED로 초기화된 뒤 계속 진행한다. '
        '첫 cell은 lazy graph capture 비용을 포함할 수 있으며, 후속 cell과 분리해 본다.','']
    (ROOT/'NUMBERS.md').write_text('\n'.join(lines))
    critical=[]
    for model in ['llama2','llama3']:
        for mode in ['tree','chain']:
            path=ROOT/f'{model}_full1'/f'{model}_{mode}_full.json'
            if not path.exists():continue
            d=json.loads(path.read_text())
            if d['status']!='complete':continue
            c=d['cells'][-1];m=c['metrics'];events=m['phase_events']
            vt,st=m['target_verify_times'],m['target_step_times']
            entry=dict(model=model,mode=mode,cell=len(d['cells'])-1,
                temperature=c['temperature'],batch=c['batch'],
                verify_median_ms=statistics.median(vt)*1000,step_median_ms=statistics.median(st)*1000,
                verify_mean_ms=statistics.mean(vt)*1000,step_mean_ms=statistics.mean(st)*1000,
                nonverify_median_ms=None,counts=[len(st),len(vt)],
                mean_live_queries_per_request=statistics.mean(e['valid_k']+1 for e in events),
                al_by_source={str(k):statistics.mean(e['accepted_spec_len']+1 for e in events if e['source']==k) for k in [0,1,2]},
                hit_by_source={str(k):sum(e['source']==k for e in events)/len(events) for k in [0,1,2]})
            if mode=='tree':
                fixed=max(d['engine_kwargs']['duet_p1_tree_verify_nodes'],d['engine_kwargs']['duet_p2_tree_verify_nodes'])
                entry['dense_queries_per_request']=statistics.mean(e.get('physical_verify_width',fixed)+1 for e in events)
            critical.append(entry)
    (ROOT/'critical_path.json').write_text(json.dumps(critical,indent=2))
    print('campaign jobs',len(inventory),'complete',sum(r['status']=='complete' for r in inventory),'cells',len(cells))
if __name__=='__main__':main()
