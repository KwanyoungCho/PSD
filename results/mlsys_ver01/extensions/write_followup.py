"""Append the questioning/re-audit results to the single handover document."""
import json
from pathlib import Path

HERE=Path(__file__).parent
d=json.loads((HERE/'FOLLOWUP_AUDIT.json').read_text())
parts=[r'''
<a id="extensions-followup-review"></a>

#### 2026-10-13 재검토: stream·packed·mixed-miss의 해석과 남은 작업

**이전 결론의 범위를 정정한다.** Proxy side stream의 고정 입력 검사는 해당 B8 조건에서 이득을 보이지 않았다. Packed tree는 shape에 따라 이미 이득이 있고, pipeline 전체의 padding 제거는 완료되지 않았다. Mixed-miss 첫 실행 TPS에는 뒤늦은 graph 준비가 섞였으므로 그 표로 정책 자체의 이득이 없다고 결론 내리지 않는다. 이번 재검토에서는 production runtime을 바꾸지 않고 bench에 capture 시각 기록과 두 번째 전체 순회를 추가했다.

**Git 배포 경계:** 사용자 요청으로 `MLSys-ver0.1@f12d0f1da6a2e4a26e9cb11ddcca54c189736d19`까지만 origin에 push하고 원격 ref를 확인했다. 새 모델·B>1 확장과 이 추가 검토는 미커밋 작업이므로 그 push에 포함되지 않는다. 논문 브랜치는 `a82f7d2` 그대로다.

##### A. Proxy 별도 stream을 끄면 DUET overlap도 없어지는가?

아니다. 구분해야 할 실행은 다음 두 가지다.

| 겹치는 작업 | 별도 proxy stream OFF | ON |
|---|---|---|
| 별도 GPU의 draft P1/P2와 target verification | 유지 | 유지 |
| 같은 target GPU의 proxy head/점수 계산과 exit 이후 layer | 순서대로 실행 | 별도 stream에서 동시 실행 |

OFF도 exit 직후 proxy를 보내고 target 후반부를 실행한다. Draft는 proxy를 받은 뒤 P2를 수행하므로 target 후반부와의 기본 overlap이 유지된다. B1 기존 경로를 이번에 끄거나 바꾼 것도 아니다.

ON은 같은 GPU에 일을 추가로 겹치므로 두 연산이 각각 느려질 수 있다. B8 Llama3 고정 입력에서 serial proxy3.981ms/post5.956ms가 side stream에서는7.435ms/9.957ms로 늘었고, 전체 target 호출의 개선으로 이어지지 않았다(전체 Llama3 24.387→24.470ms). 두 stream의 시간을 합산해 전체 시간으로 보고하지 않는다. 자원 경쟁은 관측과 일치하는 해석이며 SM/HBM counter로 병목 종류를 확정한 것은 아니다. Proxy 완료가 늦으면 draft P2 시작에도 불리할 수 있다.

과거 round5에서도 B1 proxy stream은 TPS +0.3~1.5%, B8은 −2.6~7.3%였다. 이번에만 갑자기 방향이 뒤집힌 것이 아니다. 근거: `PSD-mlsys-coverage/results/mlsys_coverage/round5/REPORT.md` 6절. 현재 권고는 **시험한 B>1 기본 경로는 OFF 유지**, 조건을 바꾸면 다시 같은 입력의 proxy 완료시각·후반 layer·target 전체를 비교하는 것이다.

##### B. Packed tree는 어디까지 구현됐는가?

`FixedPlanPackedTree.run`은 실제 query 합을8 단위로 정렬하고 `(총 query bucket, page bucket)`별 graph를 재사용한다. 같은 bucket 안의 tree 모양은 mask/metadata tensor로 전달하므로 topology가 달라질 때마다 plan을 다시 만들지 않는다. 그러나 **padded 비교 기준도 이미 `(B,width,pages)`별 plan을 재사용한다.** “재계획 제거”는 초기 packed 구현에 대한 개선이다.

현재 남은 비용은 다음과 같다.

1. Transformer/LM head는 packed query로 실행하지만 logits를 `index_select(...).view(B,width,V)`로 다시 펼친다. **Proxy ladder·수락 처리에는 padded 폭이 남는다.**
2. 고정 plan을 위해 query 하나를 attention sequence 하나로 표현한다. 예를 들어 원래8개 요청의 prefix 정보를48개 query용 page table로 반복 표현할 수 있다. KV tensor 자체를48번 복사한다는 뜻은 아니지만 metadata/attention 실행 형태가 달라진다.
3. CPU layout·ancestor mask 구성, metadata 복사, logits gather 비용이 남는다. Query 수를 줄여도 model weight를 읽는 비용 등 모든 비용이 같은 비율로 줄지는 않는다. 정확한 병목 비율은 추가 profiling 대상이다.

고정 입력 target 호출에서 Llama3 half-short는24.320→23.236ms(−4.45%), one-hit는24.240→21.143ms(−12.78%)였다. Llama2 one-hit도19.226→17.077ms(−11.18%)였다. 따라서 “tree는 효과 없음”이 아니라 **일부 shape에서는 효과가 있지만 전체 pipeline 최적화는 아직 남아 있음**이 정확하다. Half-short Llama3에는 logits/argmax 차이가 있으므로 bitwise 동일한 경로로 주장하지 않는다.

자연 입력의 실제 packing 양도 다시 집계했다. 아래 평균은 각 arm에서 조건을 만족한 step의 값으로, 서로 다른 생성 경로이므로 순수 속도 인과 비교가 아니다.

| 모델 | 실행 | 실제 query | Target 물리 query | Proxy/수락의 dense query 폭 |
|---|---|---:|---:|---:|
''']
for model,cells in d['padding'].items():
    for arm,c in cells.items():
        parts.append(f"| {model} | {arm} | {c['actual_queries']:.2f} | {c['physical_queries']:.2f} | {c['dense_proxy_queries']:.2f} |\n")
parts.append(r'''
다음 구현 검토 순서는 (a) proxy 점수/수락까지 실제 query offset을 전달하여 dense 복원을 줄이기, (b) 요청 단위 prefix 재사용을 유지하는 ragged ancestor attention, (c) metadata/mask/gather 비용 분리, (d) shape·page 범위를 포함한 graph 준비다. 어느 것도 아직 성능 개선이 입증됐다고 표시하지 않는다. Chain의 CUDA graph/CPU sync 제거 효과와 chain packing만의 효과도 구분해야 한다.

##### C. Mixed-miss: 이전 개선, K2, 실제 추가 비용

과거 round2 **packed chain**, 두 모델 모두4090, B8/T.7/K4·2/exit21,480입력·3개 seed에서 Llama3 TPS647.58→672.26(+3.81%), Llama2 651.70→651.04(−0.10%)였다. 출처는 `PSD-mlsys-coverage/results/mlsys_coverage/MERGE_REVIEW.md` 21.3절이다. 그 집계 정의는 이번 full-B/non-cap/clip 정의와 다르다. 이번 K2 sensitivity는 padded tree이며 Llama3 GPU도3090이다. 알고리즘 효과를 비교하려면 이 차이를 먼저 고정해야 한다. 이전 full_v2에는 packed+AR도 시험했지만 그 역시 lazy 준비가 있는 첫 순회 결과다.

**K2는 proxy를 받은 뒤 P2에서 draft forward를 수행하는 round 수**다. Miss 개수 threshold가 아니다. 현재 `duet_jit_short=True`는 cache miss의 JIT chain 깊이도 K2로 정하므로 K2=1/2/3 sweep은 P2와 fallback을 함께 바꾼다. 인과 분석에서는 두 값을 분리해야 한다.

Mixed-miss AR adapter는 `0<miss<B`일 때 miss 행의 proposal을0개로 만들고 JIT forward를 생략한다. 그 행은 target에서 새 token1개를 얻는다. Hit 행의 tree는 유지하고 all-miss/startup은 JIT를 유지한다. Miss q buffer도0길이로 만들어 사용하지 않는 전체 miss 분포를 계산하지 않는다. 다만 이전 step에서 시작한 glue/P1/P2가 draft worker를 사용 중이면 그것이 끝나야 다음 요청을 처리한다. AR가 그 진행 중인 cache 구축을 취소하는 구현은 아니다.

**실제로 발견한 추가 비용:** 첫48개 warm 입력은 짧은 입력에 편중돼 긴 prefix/page·tree 폭 조합을 충분히 준비하지 못했다. 기존 K2=2 첫 순회에서 median의5배보다 느린7~8개 step이 eligible decode 시간의 Llama2 wait16.41%/AR23.02%, Llama3 wait22.49%/AR22.43%를 차지했다. 이것들을 제거해서 유리한 TPS를 만들지 않고, graph 생성의 CPU 시각을 직접 기록하고 동일 엔진을 유지한 채 전체480개를 다시 순회했다.

실험: nova GPU2/3 RTX3090, CPU0–7/32–39, Llama3-8B/Qwama0.5B, B8/T.7/K1=4/K2=2/exit index21/tree8·4/C3. Wait와AR 각각 warm48→전체480 첫 순회→전체480 둘째 순회. CUDA profiling/events/추가 synchronize는 넣지 않았고 이미 이름 붙은 capture 구간의 CPU 시작·끝만 기록했다. 첫 순회는 과거 실험과 runtime hash·설정·출력·수락 event 동일성을 검사했다.

| 순회 | Wait AL → AR AL | Wait hit → AR hit | Wait TPS → AR TPS | 변화 / group-bootstrap 95% CI |
|---|---|---|---|---|
''')
for name,label in [('measured_first','첫480'),('measured_second','둘째480')]:
    w=d['repeat']['wait'][name]['metrics'];v=d['repeat']['ar'][name]['metrics'];c=d['comparisons'][name];lo,hi=c['tps_percent_ci']
    parts.append(f"| {label} | {w['al']:.3f} → {v['al']:.3f} | {100*w['hit']:.2f}% → {100*v['hit']:.2f}% | {w['tps']:.2f} → {v['tps']:.2f} | {c['tps_percent']:+.2f}% / [{lo:+.2f}, {hi:+.2f}]% |\n")
parts.append('''
| 실행/순회 | Eligible step 수 | Median / 최대 step | 5×median 초과 step 수 | 이 step들의 시간 비중 | 이 중 capture와 겹친 수 |
|---|---:|---|---:|---:|---:|
''')
for arm in ('wait','ar'):
    for name,label in [('measured_first','첫480'),('measured_second','둘째480')]:
        c=d['repeat'][arm][name];m=c['metrics']
        parts.append(f"| {arm}/{label} | {m['steps']} | {m['median_step_ms']:.2f} / {m['max_step_ms']:.2f} ms | {m['slow_steps']} | {100*m['slow_time_fraction']:.2f}% | {c['capture']['slow_steps_overlapping_capture']} |\n")
parts.append('\n전체 decode 기준 raw TPS도 보존했다: ')
parts.append('; '.join(f"{label} wait {d['repeat']['wait'][name]['metrics']['raw_decode_tps']:.2f} → AR {d['repeat']['ar'][name]['metrics']['raw_decode_tps']:.2f}" for name,label in [('measured_first','첫 순회'),('measured_second','둘째 순회')])+'.\n')
w=d['repeat']['wait']['measured_second']['metrics'];v=d['repeat']['ar']['measured_second']['metrics']
parts.append(f"\n둘째 순회의 평균 step 시간은 **{w['mean_step_ms']:.3f}→{v['mean_step_ms']:.3f}ms({100*(v['mean_step_ms']/w['mean_step_ms']-1):+.2f}%)**로 줄었지만 AL도 **{100*(v['al']/w['al']-1):+.2f}%** 감소했다. 그 결과 eligible TPS는 **{d['comparisons']['measured_second']['tps_percent']:+.2f}%**로 사실상 중립이며 CI가0을 포함한다. 이 설정에서는 AR가 대기를 실제로 줄였지만 포기한 token 수를 보상할 정도의 안정적 TPS 이득은 확인하지 못했다. 이는 파라미터가 최적인 상태의 일반 결론이 아니다.\n")
parts.append('\n둘째 순회 full-B/non-cap/clip step 중 capture와 겹친 수: '+', '.join(f"{arm} {d['repeat'][arm]['measured_second']['capture']['eligible_steps_overlapping_capture']}개" for arm in ('wait','ar'))+'. 첫 순회에서 느린7개 step은 양 arm 모두 capture와 겹쳤고, 모두 step CPU 구간의 절반 이상을 capture 구간이 차지했다. Capture 구간에는 GPU 준비 대기 등이 포함되며 실제 겹친 구간의 합집합을 계산했다.\n')
parts.append(r'''
둘째 순회는 같은 엔진의 후속 실행으로, 독립 process/seed 반복이 아니다. Target seed는 다시 설정하지만 draft RNG 상태는 이어진다. 따라서 첫→둘째 TPS 차이 전부를 capture 제거 효과로 환산하지 않는다. 모든 느린 step을 그대로 분모에 포함했다. 두 번째 순회에서도 부분 batch 등 다른 조건에는 새 capture가 남을 수 있으므로 전체 serving shape의 prewarm 완료를 주장하지 않는다. 위 CI는 입력 group bootstrap이며 GPU/seed 반복 불확실성은 포함하지 않는다.

![Capture 재감사](extensions/figs/05_capture_audit.png)

또한 기존48입력 profile에서 현재/직전 capture step을 모두 제외하고 target·draft 완료시각을 대조했다. GPU event를 host monotonic clock에 대응시킨 근사이므로 sub-ms 정밀한 cross-GPU 인과로 읽지 않는다.

| 모델 | P1 종료 − proxy 계산 종료, median | P2 종료 − 다음 target 요청 시작, median | 표본 수 |
|---|---:|---:|---:|
''')
for model,c in d['timeline'].items():
    parts.append(f"| {model} | {c['p1_end_minus_proxy_end']['median_ms']:+.3f} ms | {c['p2_end_minus_next_request']['median_ms']:+.3f} ms | {c['p1_end_minus_proxy_end']['n']} |\n")
parts.append(r'''
양수는 draft 작업이 뒤늦게 끝났다는 뜻이다. 특히 Llama3의45개 대응 step 모두 P1/P2 종료가 이 두 시점보다 늦었다. 따라서 현재 예산에서 AR가 miss JIT를 없애도 **진행 중인 cache 구축 대기**가 남을 수 있다. 이 분석은 온라인 TPS와 다른 소규모 instrumented run이다. P1/P2 시간을 더해서 target 시간과 단순 비교한 TPS 예측이 아니다.

정책 채택은 같은 상태에서 `절약 시간 / wait 시간 > 포기하는 기대 token / wait 기대 token`인지, 그리고 최종 파라미터의 반복 전체 실행에서도 이득이 유지되는지로 판단해야 한다. 아직 K1/K2/exit/tree 예산의 최적값이나 miss-count threshold를 찾았다고 하지 않는다. 새 결과가 이득을 보이더라도 모든 모델/tree에 일반화하지 않는다.

##### D. 완료된 검증과 미완료 범위

| 항목 | 이번에 확인한 범위 | 남은 사항 |
|---|---|---|
| 새 모델 지원 | 두 dense 모델 pair, SSD 호환 smoke, B1/B2/B8와 T0 등 targeted 회귀 | 최적 parameter·튜닝 SSD와의 최종 비교,70B/AWQ/TP4 일반화 |
| B>1 tree 실행기 | 요청 독립성, tree attention/KV/수락, B1 기존 출력 회귀,234회 targeted 검사 | 저장소 전체 test 통과는 아님; 기존 분석 module 누락1건, 추가 B/context 범위 |
| 결합 kernel / bulk metadata | 같은 출력/event에서 draft P1/P2 직접 latency 감소 | 남은 CPU metadata와 phase deadline 최적화 |
| 불필요 target 계산 제거 | Proxy/accept 직접 latency 감소, 난수·수락 검사 | Target transformer 자체 가속으로 주장하지 않음 |
| Packed tree | Plan 재사용, 일부 shape에서 target latency 개선 | Dense proxy/accept 잔존, metadata와 attention 형태, 수치 차이 검사 |
| Mixed-miss AR | Zero-proposal exact 처리, 실제 JIT 생략,graph 준비 비용 재감사 | 파라미터/독립 반복/packed와의 상호작용, 기본 정책 미채택 |
| B>1 proxy side stream | 같은 입력 target latency 비교 | 현재 조건에서는 기본OFF; 자원 경쟁 종류 정밀 분리 미완료 |

따라서 **승인된 기능의 이식·시험한 구간의 기능/latency 검증은 했지만, 전체 시스템의 최적화가 모두 끝난 것은 아니다.** Miss DS형 구성과 root/위치/tree AL 정책은 요청대로 별도 후속 검토다.

재현: `followup_audit.py --raw-root <기존 raw root>` → `write_followup.py`. 원시 기록은 `nova_latest/review_followup`, 결과는 [FOLLOWUP_AUDIT.json](extensions/FOLLOWUP_AUDIT.json). 최초 source/raw archive는 그대로 보존하고 이번 추가 bench·분석·로그는 `FOLLOWUP_SOURCE.tar.gz`/`FOLLOWUP_RAW.tar.gz`와 `FOLLOWUP_ARTIFACTS.json`에 별도로 보존한다. 새 runtime 변경이나 기본 옵션 채택은 없다.
''')
report=HERE.parent/'REPORT.md';s=report.read_text();marker='<a id="extensions-followup-review"></a>'
later_marker='<a id="measurement-and-layout-review"></a>'
later=s[s.index(later_marker):] if later_marker in s else ''
if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
report.write_text(s+''.join(parts)+('\n'+later if later else ''))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11,6.5))
for i,arm in enumerate(('wait','ar')):
    for j,name in enumerate(('measured_first','measured_second')):
        c=d['repeat'][arm][name];rows=c['capture']['rows'];ax=axes[i,j]
        ax.plot([1000*r['seconds'] for r in rows],linewidth=.8,color='#007c91')
        ix=[k for k,r in enumerate(rows) if r['captures']]
        ax.scatter(ix,[1000*rows[k]['seconds'] for k in ix],s=20,color='#d95f02',label='Overlaps graph capture',zorder=3)
        ax.set_yscale('log');ax.set_ylim(20,4000);ax.set_ylabel('Step wall latency (ms)')
        ax.set_xlabel('Eligible step index (none removed)')
        ax.set_title(f"{arm.upper()} / {'First' if j==0 else 'Second'} 480 prompts; {c['metrics']['tps']:.2f} TPS")
        ax.grid(axis='y',alpha=.2)
        if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('Mixed-miss graph preparation audit: Llama3 / RTX 3090, B8\nSecond pass reuses the engine; not an independent seed/process repetition',fontsize=12)
fig.tight_layout(rect=(0,0,1,.91));dest=HERE/'figs';dest.mkdir(exist_ok=True)
fig.savefig(dest/'05_capture_audit.png',dpi=180);fig.savefig(dest/'05_capture_audit.pdf');plt.close(fig)
print('Updated',report)
