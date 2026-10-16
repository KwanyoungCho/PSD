"""Detailed native timelines and survival curves matching the user's example."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RAW=Path('/data/chokwans99/duet-chain40-k10-20261016')
sys.path.insert(0,str(ROOT/'ssd/tools/duet_timeline'))
from export_profile_timelines import export
from build_model_timelines import main as build

LABEL={1:'Draft-source hit (P1)',2:'Proxy-source hit (P2)',0:'Cache miss (Re-draft)'}
COLORS={1:'#4C78A8',2:'#EF553B',0:'#72B7B2'}


def curve(ax,run):
    for r in run['survival']:
        ax.plot([p['x'] for p in r['points']],[100*p['probability'] for p in r['points']],
            '-o',color=COLORS[r['state']],lw=1.8,ms=4,label=f"{LABEL[r['state']]} (n={r['steps']})")
    ax.set(title=f"LayerSkip Llama2-{run['model'].upper()} + llama-68m\nChain;40 roots/phase; K1=K2=10",xlabel='Accepted length (or longer)',
        ylabel='Probability (%)',xlim=(.6,11.4),ylim=(0,103),xticks=range(1,12))
    ax.yaxis.set_major_formatter(PercentFormatter(100));ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.25);ax.legend(loc='upper right',fontsize=9)


def main():
    with gzip.open(HERE/'SAMPLES.json.gz','rt') as f:runs=json.load(f)
    images=HERE/'figures';images.mkdir(exist_ok=True)
    csv_rows=[];round_rows=[]
    text=['# Chain: phase별40후보, K1=K2=10 측정','',
          '같은 로컬 서버·모델·입력으로 chain 경로를 측정했다. 설정·집계·AL 정의는 [README.md](README.md). 모든 시간은 ms다.',
          '', '[모든 timeline PDF](timelines/ALL_MODELS.pdf) · [HTML](timelines/index.html) · [AL 확률 PDF](AL_SURVIVAL_MODELS.pdf) · [확률·95% 구간 CSV](AL_SURVIVAL.csv)',
          '', '추가 분석: [PS 초과와 다음 step의 response wait](RESPONSE_WAIT.md).',
          '', 'DS는11개 위치에4개×7+3개×4로 총40개, PS는 점수로 총40개를 선택한다. 각각40행×10회 forward로 후보당 길이10의 chain을 만든다. Target 실제 검증은10개 draft token이며 AL 최대는 correction/bonus 포함11이다.', '']
    spec=dict(schema='duet-timeline-inputs-v1',skip_initial_steps=10,datasets=[])
    with PdfPages(HERE/'AL_SURVIVAL_MODELS.pdf') as pdf:
        for run in runs:
            model=run['model'];rows=run['summary'];overall=rows[-1]
            fig,ax=plt.subplots(figsize=(7.5,5.1));curve(ax,run);fig.tight_layout();pdf.savefig(fig)
            fig.savefig(images/f'{model}_al_survival.png',dpi=180);fig.savefig(images/f'{model}_al_survival.pdf');plt.close(fig)
            for population,key in [('steady','survival'),('all_nonterminal','all_nonterminal_survival')]:
                for r in run[key]:
                    for point in r['points']:
                        csv_rows.append(dict(model=model,population=population,source=LABEL[r['state']],n=r['steps'],**point))
            text += [f'## Llama2-{model.upper()} + llama-68m','',
                     f"B1, BF16, T0.7, exit {run['args']['exit_layer']} (0-based), target GPU{run['args']['target_tp']}장.12입력×2회. 유효 step {run['eligibility']['included']}개.",
                     '', '| 현재 step | n | 비중/전체 hit (%) | AL | Profile TPS |', '|---|---:|---:|---:|---:|']
            for r in rows:
                rate=r['hit_rate_pct'] if r['state']=='all' else r['share_pct']
                text.append(f"| {r['label']} | {r['n']} | {rate:.3f} | {r['al']:.3f} | {r['tps']:.3f} |")
            text += ['', '| 현재 step | Step | DS forward1회 | DS 계산창 | DS 여유 | PS forward1회 | PS 계산창 | PS 여유 |',
                     '|---|---:|---:|---:|---:|---:|---:|---:|']
            fields=['step_ms','ds_forward_once_ms','ds_compute_window_ms','ds_margin_ms',
                    'ps_forward_once_ms','ps_compute_window_ms','ps_margin_ms']
            for r in rows:text.append('| '+r['label']+' | '+' | '.join(f'{r[k]:.3f}' for k in fields)+' |')
            text += ['', '계산창은 이미 사용한 forward 합을 포함한다. **음수 PS 여유는 다음 target 요청이 시작될 때 draft가 아직 준비를 마치지 못했음을 뜻한다.** 계산창도 음수라면 현재 고정된 비-forward 비용 아래 deadline 내 forward 여유가 없다는 뜻이다.',
                     '', '| 현재 step | DS 전체 | PS 전체 | DS deadline 초과 | PS deadline 초과 | DS 여유 하위5% | PS 여유 하위5% |',
                     '|---|---:|---:|---:|---:|---:|---:|']
            for r in rows:
                text.append(f"| {r['label']} | {r['ds_total_ms']:.3f} | {r['ps_total_ms']:.3f} | {r['ds_negative_margin_steps']}/{r['n']} | {r['ps_negative_margin_steps']}/{r['n']} | {r['ds_margin_p05_ms']:.3f} | {r['ps_margin_p05_ms']:.3f} |")
                for phase in ('ds','ps'):
                    for i,v in enumerate(r[phase+'_rounds_mean_ms']):
                        round_rows.append(dict(model=model,state=r['label'],phase=phase,round=i+1,n=r['n'],input_rows=40,forward_ms=v))
            text += ['', 'PS 전체는 proxy 수신 완료→cache merge 완료다. PS 여유는 그 뒤의 정리까지 마친 수신 준비 시각을 사용한다. Forward1회는 model+LM head graph replay이며, 준비·샘플링을 포함한 전체 round 시간과 다르다.',
                     '', '| 현재 step | P(AL≥2) | P(AL≥4) | P(AL≥8) | P(AL≥10) | P(AL≥11) |',
                     '|---|---:|---:|---:|---:|---:|']
            for r in run['survival']:
                text.append('| '+LABEL[r['state']]+' | '+' | '.join(f"{100*r['points'][x-1]['probability']:.2f}%" for x in (2,4,8,10,11))+' |')
            text += ['',f'![AL이 x 이상일 확률](figures/{model}_al_survival.png)',
                     '', '각 선은 해당 source 안에서의 조건부 확률이다. x=1은100%이며 AL에는 correction/bonus가 포함된다. 두 반복을 묶은 UID bootstrap 구간과 초기 구간 포함 민감도 곡선은 AL_SURVIVAL.csv와 SUMMARY.json에 있다.',
                     '', '제외 내역: `'+json.dumps(run['eligibility']['skipped'])+'`.', '']
            folder=RAW/model
            spec['datasets'].append(dict(model_id='chain_'+model+'_68m',model_name=f'LayerSkip Llama2-{model.upper()} + llama-68m',
                setting=f"Chain; DS/PS roots40/40; K1/K2 10/10; exit{run['args']['exit_layer']}",
                measurement='Fresh local RTX4090; B1 BF16 T0.7;12 inputs x2; first10/terminal/cap/preparation excluded; separately timed native forwards',
                runs=[dict(outcomes=str(folder/'run.json'),profile_dir=str(folder))]))
    fig,axes=plt.subplots(1,len(runs),figsize=(7.5*len(runs),5.1),squeeze=False)
    for ax,run in zip(axes[0],runs):curve(ax,run)
    fig.tight_layout();fig.savefig(HERE/'AL_SURVIVAL_MODELS.png',dpi=180);plt.close(fig)
    for name,rows in [('AL_SURVIVAL.csv',csv_rows),('FORWARD_ROUNDS.csv',round_rows)]:
        with (HERE/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text += ['## 비교 범위','',
             '이번 그래프는12개 입력의 진단 결과이며 전체 Spec-Bench의 AL 곡선이 아니다. 직전 tree 실험은 root 수·K·M이 달랐으므로 이번 결과와의 차이를 chain/tree 구조 하나의 효과로 해석하지 않는다.',
             '', 'P1/P2/miss는 서로 다른 step 집합이다. 따라서 P1 조건부 AL이 높다는 관측만으로 같은 prefix에서 P1 생성 방식이 더 우수하다는 인과 결론을 내리지 않는다. Miss는69/99개 step으로 표본이 작아 꼬리 확률을 특히 조심해서 해석한다.',
             '', '이번 설정에서는 forward1회가 두 모델 모두 약0.214ms이고10회 합은 phase당 약2.14ms다. 그러나 준비·샘플링·계획·중간 대기를 포함한 DS/PS 전체는 약9~10ms다. PS 여유 평균은7B −2.130ms,13B −1.099ms이며 다음 target 요청 시점을 넘긴 비율은 각각79.25%,78.24%다. 따라서 K1=K2=10은 요청한 측정 조건이며, latency 관점의 최적 설정으로 확정하지 않는다.',
             '', '기존 엔진의 native chain 경로를 그대로 사용했다. 계산 자체와 각 round의 CPU 준비/plan/샘플링을 timeline에서 구분한다. Proxy replica/별도 stream은 OFF이고 target/draft는 다른 GPU에서 병렬 실행한다.']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n')
    spec_path=HERE/'TIMELINE_INPUTS.json';spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    dest=HERE/'timelines'
    if dest.exists():shutil.rmtree(dest)
    export(spec_path,dest);build(['--manifest',str(dest/'SOURCES.json'),'--output',str(dest)])


if __name__=='__main__':main()
