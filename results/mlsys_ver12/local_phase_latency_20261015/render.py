"""Readable local latency tables, W scaling figures and fresh native timelines."""
import gzip
import json
from pathlib import Path
import sys
import shutil
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'ssd/tools/duet_timeline'))
from export_profile_timelines import export
from build_model_timelines import main as build


def main():
    with gzip.open(HERE/'SAMPLES.json.gz','rt') as f: runs=json.load(f)
    by={r['name']:r for r in runs}
    # Keep every arm and current outcome inspectable without opening raw traces.
    phase_rows=[]; node_rows=[]
    for run in runs:
        for item in run['summary']:
            for phase in ('ds','ps'):
                if phase not in item:continue
                for r in item[phase]['rounds']:
                    phase_rows.append(dict(run=run['name'],state=item['state'],phase=phase,
                        round=r['round'],steps=item['steps'],
                        **{k+'_'+stat:r[k][stat] for k in ('width','model_ms','head_ms','forward_ms')
                           for stat in ('mean','p50','p95','max')}))
                node_rows.append(dict(run=run['name'],state=item['state'],population=phase+'_all_roots',
                                      **item[phase]['all_root_nodes']))
            node_rows.append(dict(run=run['name'],state=item['state'],population='actual_hit_tree',
                                  **item['hit_tree_nodes']))
    for name,rows in [('FORWARD_ROUNDS.csv',phase_rows),('TREE_NODES.csv',node_rows)]:
        with (HERE/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
    text=['# 로컬 모델별 DS/PS 시간 예산과 입력 폭 진단','',
          '이 결과는 지정한 출발 설정의 짧은 중간 구간 진단이다. 전체 데이터셋 품질이나 최적 파라미터를 확정한 결과가 아니다.',
          '', '조건·집계식·계측 한계: [README.md](README.md). 모든 시간은 ms이며 root 노드 자체는 자손 수에서 제외한다.',
          '', '그림: [연속 두 step timeline PDF](timelines/ALL_MODELS.pdf), [HTML](timelines/index.html), [폭·노드 분포 PDF](DIAGNOSTICS.pdf).',
          '전체 설정·source별 세 forward: [FORWARD_ROUNDS.csv](FORWARD_ROUNDS.csv). 전체 후보와 hit tree 노드 분포: [TREE_NODES.csv](TREE_NODES.csv).', '']
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    spec=dict(schema='duet-timeline-inputs-v1',skip_initial_steps=10,datasets=[])
    states={1:'P1 hit',2:'P2 hit',0:'Miss','all':'전체'}
    with PdfPages(HERE/'DIAGNOSTICS.pdf') as pdf:
        for model in ('7b','13b'):
            if model+'_base' not in by:continue
            base=by[model+'_base'];text += [f'## LayerSkip-Llama2-{model.upper()} + llama-68m','',
                f"Exit={base['args']['exit_layer']} (0-based), target GPU={base['args']['target_tp']}, K1/K2=3/3, DS 위치당 root=3, PS root=9, W1/W2=9/9, C=3, M1/M2=40.", '',
                '| 현재 step | 표본 수 | Step | DS forward 1회 | DS 총 계산창 | DS 남은 여유 | PS forward 1회 | PS 총 계산창 | PS 남은 여유 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in base['summary']:
                keys=['step_ms','ds_forward_once_ms','ds_compute_window_ms','ds_margin_ms','ps_forward_once_ms','ps_compute_window_ms','ps_margin_ms']
                text.append('| '+states[r['state']]+' | '+str(r['steps'])+' | '+' | '.join(fmt(r['metrics'][k]['mean']) for k in keys)+' |')
            text += ['', '계산창은 이미 사용한 forward 시간을 포함한다. DS 여유는 proxy **송신 종료** 기준이며 도착을 직접 측정한 값이 아니다. PS 여유는 정리 후 다음 명령 수신 준비까지 반영한다.', '',
                     '| 현재 step | 비중 (%) | AL | 진단 TPS | hit tree 자손 평균 | p95 | 최대 |',
                     '|---|---:|---:|---:|---:|---:|---:|']
            for p,r in zip(base['performance'],base['summary']):
                nodes=r['hit_tree_nodes'];text.append('| '+states[r['state']]+' | '+' | '.join(fmt(v) for v in (p['share_pct'],p['al'],p['tps'],nodes['mean'],nodes['p95'],nodes['max']))+' |')
            text += ['', '| 현재 step | DS 여유 하위5% | DS 음수 step | PS 여유 하위5% | PS 음수 step |',
                     '|---|---:|---:|---:|---:|']
            for r in base['summary']:
                text.append('| '+states[r['state']]+' | '+' | '.join(
                    f"{r['metrics'][p+'_margin_ms']['p05']:.3f} | {r['negative_margin_steps'][p]}/{r['steps']}" for p in ('ds','ps'))+' |')
            text += ['', f"전체 hit rate: {fmt(base['performance'][-1]['hit_rate_pct'])}%. Miss는 hit tree 통계 대상이 아니다.", '',
                     '| 설정 W1/W2 | DS forward 평균 | PS forward 평균 | DS 전체 후보 자손 평균 / p95 / 최대 | PS 전체 후보 자손 평균 / p95 / 최대 |',
                     '|---|---:|---:|---|---|']
            selected=[by[model+'_'+a] for a in ('base','ds2','ds3','psbase','ps2','ps3') if model+'_'+a in by]
            for run in selected:
                r=run['summary'][-1]
                cells=[f"{run['args']['ds_width']}/{run['args']['ps_width']} ({run['name']})",fmt(r['metrics']['ds_forward_once_ms']['mean']),fmt(r['metrics']['ps_forward_once_ms']['mean'])]
                for phase in ('ds','ps'):
                    d=r[phase]['all_root_nodes'];cells.append(' / '.join(fmt(d[k]) for k in ('mean','p95','max')))
                text.append('| '+' | '.join(cells)+' |')
            text += ['', '각 후보의 자손 수는 0개인 유효 후보도 포함한다. 패딩 root는 제외한다. 아래 입력 폭 그래프는 첫 forward와 후속 forward를 분리한다.', '']
            if model+'_psbase' in by:
                text += ['7B의 DS 비교는 GPU4·5의 base를, PS 비교는 GPU6·7의 psbase를 기준으로 한다. 두 기준의 절대 latency 차이를 W 효과로 해석하지 않는다.', '']
            text += ['| W를 9→27로 변경 | 9의 forward 평균 | 27의 forward 평균 | 증감 |',
                     '|---|---:|---:|---:|']
            for phase,ref,arm in [('ds','base','ds3'),('ps','psbase' if model+'_psbase' in by else 'base','ps3')]:
                if model+'_'+arm not in by:continue
                key=phase+'_forward_once_ms'
                before=by[model+'_'+ref]['summary'][-1]['metrics'][key]['mean']
                after=by[model+'_'+arm]['summary'][-1]['metrics'][key]['mean']
                text.append(f'| {phase.upper()} | {before:.3f} | {after:.3f} | {(after/before-1)*100:+.2f}% |')
            text += ['', '실제 모델 forward 평균이며, source·문맥의 구성은 생성 경로에 따라 달라진다. 동일 입력 tensor를 고정한 단일 커널 A/B가 아니다. 폭·round·prefix page별 분해는 AUDIT.json의 by_shape에서 확인한다.', '',
                     '| 현재 step | phase | 1회차 행 수 / latency | 2회차 행 수 / latency | 3회차 행 수 / latency |',
                     '|---|---|---:|---:|---:|']
            for row in base['summary']:
                for phase in ('ds','ps'):
                    cells=[states[row['state']],phase.upper()]+[
                        f"{r['width']['mean']:.2f} / {r['forward_ms']['mean']:.3f}" for r in row[phase]['rounds']]
                    text.append('| '+' | '.join(cells)+' |')
            text += ['', 'DS의 W는 첫 forward를 제한하지 않는다. 실제 첫 입력은 tree-hit context bucket×위치당 root3이고, 후속 입력은 min(bucket root 수,W)다. PS는 root9개를 유지하지만 graph의 물리적 폭은 W다. 따라서 W=27 설정이 모든 단계의 실제 계산량을 정확히3배로 만든다는 뜻은 아니다.', '']
            text += ['준비·선택·정리의 나머지 비용을 현재 K=3 값으로 고정한 계산창이다. K를 늘리면 round별 샘플링·선택 비용도 늘어나므로 계산창을 모델 forward 시간만으로 나눈 값을 새 K로 사용하면 안 된다.', '',
                     '| 현재 step | DS graph 1round | DS 비-forward 합 | PS graph 1round | PS 비-forward·정리 합 |',
                     '|---|---:|---:|---:|---:|']
            for perf,row in zip(base['performance'],base['summary']):
                text.append('| '+states[row['state']]+' | '+' | '.join(fmt(v) for v in (
                    perf['ds_round_ms'],row['metrics']['ds_nonforward_ms']['mean'],
                    perf['ps_round_ms'],row['metrics']['ps_nonforward_ms']['mean']))+' |')
            text.append('')
            fig,axes=plt.subplots(2,2,figsize=(13,8))
            for col,(phase,arms) in enumerate((('ds',('base','ds2','ds3')),('ps',('base','ps2','ps3')))):
                if phase=='ps' and model+'_psbase' in by:arms=('psbase','ps2','ps3')
                if not all(model+'_'+a in by for a in arms):continue
                for ri in range(3):
                    rr=[by[model+'_'+a]['summary'][-1][phase]['rounds'][ri] for a in arms]
                    axes[0,col].plot([9,18,27],[r['forward_ms']['mean'] for r in rr],'-o',label=f'forward {ri+1}')
                axes[0,col].set(title=phase.upper()+' model + head per forward',xlabel='Configured W',ylabel='Mean ms',xticks=[9,18,27]);axes[0,col].legend();axes[0,col].grid(alpha=.25)
                for arm in arms:
                    hist=by[model+'_'+arm]['summary'][-1][phase]['root_node_histogram'];xx=sorted(map(int,hist));yy=np.cumsum([hist[str(n)] for n in xx]);yy=yy/yy[-1]
                    axes[1,col].step(xx,yy,where='post',label=arm)
                axes[1,col].set(title=phase.upper()+' all real roots: descendants CDF',xlabel='Descendants per root',ylabel='Cumulative fraction',ylim=(0,1.02));axes[1,col].legend();axes[1,col].grid(alpha=.25)
            fig.suptitle(f'LayerSkip Llama2-{model.upper()} + llama-68m | K1=K2=3, C=3 | local RTX4090')
            fig.tight_layout();pdf.savefig(fig);fig.savefig(figures/f'{model}_width_nodes.png',dpi=150);plt.close(fig)
            text.append(f'![입력 폭과 노드 분포](figures/{model}_width_nodes.png)')
            text += ['', '실제 입력 폭 / 문맥 / 상태별 각 round의 model·head·forward 평균과 분위수는 SUMMARY.json에 모두 보존했다.', '']
            folder=Path('/data/chokwans99/duet-local-phase-20261015')/(model+'_base')
            spec['datasets'].append(dict(model_id='llama2_'+model+'_68m',model_name=f'LayerSkip Llama2-{model.upper()} + llama-68m',
                setting=f'Exit {base["args"]["exit_layer"]}; K1/K2 3/3; DS roots/pos 3, PS roots 9; W1/W2 9/9; C3; M40',
                measurement='Fresh local orion RTX4090; B1 BF16 T0.7; 12 fixed turns x 2; first10/terminal/cap excluded; GPU forward markers enabled',
                runs=[dict(outcomes=str(folder/'run.json'),profile_dir=str(folder))]))
    audit=json.loads((HERE/'AUDIT.json').read_text())
    text += ['## 검증과 한계','',
             '| 모델 | 계측 OFF→ON 출력 | 공통 step | 평균 step 차이 (ms) | 문항 평균 차이의95% bootstrap 구간 |',
             '|---|---|---:|---:|---|']
    for model,r in audit['probe_comparison'].items():
        text.append(f"| {model} | {'24/24 동일' if r['outputs_equal'] else '불일치'} | {r['paired_steps']} | {r['mean_step_difference_ms']:+.3f} | {r['uid_bootstrap95_ms'][0]:+.3f} ~ {r['uid_bootstrap95_ms'][1]:+.3f} |")
    text += ['', 'Marker kernel을 포함한 실측이다. 독립 실행의 변동과 계측 비용을 완전히 분리한 수치는 아니며, 구간이0을 포함한다고 비용이0이라는 뜻은 아니다.12개 UID로 묶어 bootstrap했고 두 반복을24개 독립 문항으로 취급하지 않았다.',
             '', '| 실행 | 유효 step | 준비 영향으로 추가 제외 | GPU 외부 작업 관측 | telemetry 최대 간격 (s) |',
             '|---|---:|---:|---:|---:|']
    for r in runs:
        t=audit['telemetry'][r['name']]
        text.append(f"| {r['name']} | {r['eligibility']['included']} | {r['eligibility']['skipped'].get('preparation',0)} | {len(t['foreign'])} | {t['maximum_sampling_gap_s']:.2f} |")
    if '7b' in audit['repeat_comparison']:
        r=audit['repeat_comparison']['7b']
        text += ['', f"7B base의 worker 인계 중 telemetry에 약36초 공백이 있어 같은 설정을 별도 `base_repeat`로 다시 확인했다. 기존 base/반복 출력은 {'동일' if r['outputs_equal'] else '불일치'}, 공통 step 평균 {r['before_mean_step_ms']:.3f}→{r['after_mean_step_ms']:.3f}ms. 기존 base를 삭제하거나 더 빠른 반복으로 교체하지 않았다."]
    text += ['', '13B 첫 실행에서는 두 번째 target GPU의 `_pack_verify_inputs_kernel` 최초 컴파일에 최대 약51초가 걸렸다. CPU 컴파일 구간과 겹치는 step을 직접 식별해 제외했고, latency 크기로 잘라내지 않았다. 모델·입력폭마다 제외 개수는 위 표와 AUDIT.json에 남았다.',
             '', '7B target1장·13B target2장이라 두 target의 절대 latency 차이를 모델 크기 하나의 효과로 해석하지 않는다. proxy replica/별도 stream은 이 진단에서 OFF이며 target과 draft는 서로 다른 GPU에서 병렬 실행한다.',
             '', 'M=40이 생성 상한에는 걸리지 않아도 target graph는41 query, DS 첫 forward는 최대123행으로 패딩한다. 실제 hit tree는 그보다 작다. 이 결과는 후속 M 축소·K 조정의 근거이며, 지금 설정이 최적이라는 결론은 아니다.']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n')
    spec_path=HERE/'TIMELINE_INPUTS.json';spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    dest=HERE/'timelines'
    # Only this campaign's generated bundle; historical canonical timelines
    # and all source profiles remain untouched.
    if dest.exists():shutil.rmtree(dest)
    export(spec_path,dest)
    build(['--manifest',str(dest/'SOURCES.json'),'--output',str(dest)])


if __name__=='__main__':main()
