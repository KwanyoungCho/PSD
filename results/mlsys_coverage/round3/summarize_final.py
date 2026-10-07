"""Build review tables/plots from complete campaigns only; keep both passes."""
import json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(directory,name):
    manifest=ROOT/directory/'campaign.json'
    records=json.loads(manifest.read_text()) if manifest.exists() else []
    if not any(r['name']==name and r['status']=='complete' for r in records):return None
    d=json.loads((ROOT/directory/(name+'.json')).read_text())
    return d if d['status']=='complete' else None

def extract(d):
    out=[]
    for c in d['cells']:
        ev=c['metrics']['phase_events'];steps={}
        for e in ev:steps.setdefault(e['step_id'],[]).append(e)
        physical=live=0
        for events in steps.values():
            e=events[0];b=e['batch_size'];cap=1<<(b-1).bit_length()
            physical+=e.get('physical_batch_queries',cap*(e.get('physical_verify_width',e['verify_width'])+1))
            live+=sum(e['valid_k']+1 for e in events)
        out.append(dict(seed=c['seed'],batch=c['batch'],temperature=c['temperature'],**c['summary'],
            target_verify_median_ms=1000*statistics.median(c['metrics']['target_verify_times']),
            live_queries=live,physical_queries=physical if d['args']['mode']=='duet-tree' else None,
            query_utilization=live/physical if physical and d['args']['mode']=='duet-tree' else None))
    return out

def main():
    comparison={};full={};p2={};packed_greedy={};placement={}
    for model in ['llama2','llama3']:
        comparison[model]={}
        for mode in ['tree_eager_helpers','tree_optimized','chain','tree_packed']:
            directory='packed_comparison' if mode=='tree_packed' else 'comparison'
            d=load(directory,f'{model}_{mode}')
            if d:
                comparison[model][mode]=[c for c in extract(d) if c['temperature']==.7]
                if mode=='tree_packed':packed_greedy[model]=[c for c in extract(d) if c['temperature']==0]
        full[model]={mode:extract(d) for mode in ['tree','chain']
            if (d:=load(f'{model}_full1',f'{model}_{mode}_full'))}
        d=load(f'{model}_p2',f'{model}_p2tree_full');p2[model]=extract(d) if d else []
    for mode in ['tree_optimized','chain']:
        d=load('placement_pix',f'llama3_{mode}')
        if d:placement[mode]=extract(d)
    summary=dict(comparison=comparison,full=full,p2=p2,packed_greedy=packed_greedy,placement=placement)
    (ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    lines=['# 최종 비교 수치','',
        '같은 GPU 0,1에서 순차 실행한 B8/T0.7/480 first turns/input512/output64 결과. '
        '첫 번째 cell에는 미리 생성되지 않은 graph의 capture가 포함된다. '
        '두 번째 cell은 같은 engine을 다시 사용한다. 독립 반복의 평균/신뢰구간이 아니다.','',
        '| 모델 | 경로 | 첫 pass TPS | 두 번째 pass TPS | 두 번째 AL | 두 번째 cache hit | Target verify 중앙값 ms | Query 사용률 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for model,modes in comparison.items():
        for name,cells in modes.items():
            a,b=cells
            utilization='—' if b['query_utilization'] is None else f"{100*b['query_utilization']:.1f}%"
            lines.append(f"| {model} | {name} | {a['decode_tps']:.2f} | {b['decode_tps']:.2f} | {b['al_including_recovery']:.4f} | {100*b['cache_hit']:.2f}% | {b['target_verify_median_ms']:.2f} | {utilization} |")
    lines+=['','Query 사용률 = 실노드+각 root query 수 / 실제 model query 수. '
        'Chain의 물리 query 수는 이 field로 기록되지 않으므로 해당 비율은 tree끼리만 비교한다.','',
        '## 세 target seed cell의 AL 평균 (모델별 전체 입력 실험)','',
        '| 모델 | T | Tree AL | Chain AL | 차이 | Tree hit | Chain hit |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model,modes in full.items():
        if len(modes)!=2:continue
        for t in [0.,.7]:
            groups={mode:[c for c in cells if c['temperature']==t] for mode,cells in modes.items()}
            mean=lambda mode,key:statistics.mean(c[key] for c in groups[mode])
            ta,ca=mean('tree','al_including_recovery'),mean('chain','al_including_recovery')
            lines.append(f"| {model} | {t} | {ta:.4f} | {ca:.4f} | {100*(ta/ca-1):+.2f}% | {100*mean('tree','cache_hit'):.2f}% | {100*mean('chain','cache_hit'):.2f}% |")
    lines+=['','각 seed의 AL을 산술평균했다. Target RNG는 재설정하지만 draft RNG는 계속 진행한다. '
        'Full 비교 일부는 서로 다른 GPU pair에서 동시 실행했으므로 TPS 주장은 위 순차 비교를 사용한다.','']
    if packed_greedy:
        lines+=['## Packed tree greedy (B8, 480 first turns)','',
            '| 모델 | 첫 T0 pass TPS | 두 번째 T0 pass TPS | 두 번째 AL |',
            '|---|---:|---:|---:|']
        for model,cells in packed_greedy.items():
            a,b=cells
            lines.append(f"| {model} | {a['decode_tps']:.2f} | {b['decode_tps']:.2f} | {b['al_including_recovery']:.4f} |")
        lines+=['','같은 engine에서 T0.7 측정 후 실행했다. T0 전용 acceptance graph의 첫 capture는 새로 발생할 수 있다.','']
    if placement:
        lines+=['## Llama3 GPU 배치 비교 (B8/T0.7)','',
            '| 경로 | GPU0·1 NODE 두 번째 TPS | GPU4·5 PIX 첫 TPS | GPU4·5 PIX 두 번째 TPS | PIX 두 번째 AL |',
            '|---|---:|---:|---:|---:|']
        for mode,cells in placement.items():
            a,b=cells;node=comparison['llama3'][mode][1]
            lines.append(f"| {mode} | {node['decode_tps']:.2f} | {a['decode_tps']:.2f} | {b['decode_tps']:.2f} | {b['al_including_recovery']:.4f} |")
        lines+=['','코드와 CLI는 같지만 GPU pair/NUMA 위치 및 실행 시점이 다르다. '
            '운영 배치의 영향으로 해석하며 PCIe 한 요소만의 인과 효과라고 주장하지 않는다.','']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines))
    if all(len(x)==4 for x in comparison.values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained',sharey=True)
        modes=['tree_eager_helpers','tree_optimized','tree_packed','chain']
        labels=['Tree\n(eager helpers)','Tree\n(graph helpers)','Tree\n(packed queries)','Chain\n(packed queries)']
        colors=['#99a4ad','#3971ad','#269a8e','#d49b35']
        ceiling=max(c['decode_tps'] for items in comparison.values() for m in modes for c in items[m])*1.17
        for ax,(model,items) in zip(axes,comparison.items()):
            values=[items[m][1]['decode_tps'] for m in modes]
            bars=ax.bar(labels,values,color=colors)
            ax.scatter(range(4),[items[m][0]['decode_tps'] for m in modes],marker='x',color='black',label='First pass')
            ax.bar_label(bars,fmt='%.1f',padding=3);ax.set_ylim(0,ceiling)
            ax.tick_params(axis='x',labelsize=9)
            ax.set_title('LayerSkip-Llama2-7B + AMD-135M' if model=='llama2' else 'LayerSkip-Llama3-8B + Qwama-0.5B',fontsize=10)
            ax.set_ylabel('Decode tokens / second');ax.spines[['top','right']].set_visible(False)
        axes[0].legend(fontsize=7,loc='upper left')
        fig.suptitle('B=8, T=0.7, all 480 first turns; bars = second pass in same engine',fontsize=11)
        for ext in ['png','pdf']:fig.savefig(ROOT/f'throughput_comparison.{ext}',dpi=180)
    print((ROOT/'SUMMARY.md').read_text())
if __name__=='__main__':main()
