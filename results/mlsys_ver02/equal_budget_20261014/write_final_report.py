"""Render the reviewed full AL analysis with traceable tables and caveats."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
R=json.loads((P/'RESULTS.json').read_text());F=json.loads((P/'FINAL_ANALYSIS.json').read_text())
assert all(x['status']=='complete' for x in R.values()) and F['state']=='complete'
N=['chain_m32','last_c3_m32','frontier_c3_m32','last_c8_m32','frontier_c8_m32','last_c8_m64','frontier_c8_m64']
L=dict(zip(N,['Chain C1/M32','마지막 깊이 C3/M32','전체 말단 C3/M32','마지막 깊이 C8/M32','전체 말단 C8/M32','마지막 깊이 C8/M64','전체 말단 C8/M64']))
def q(h,n,k):return R[h]['arms'][n]['question_quality']['all'][k]['mean']
def avg(n,k):return float(np.mean([q(h,n,k) for h in R]))
def tab(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def comparison(a,b):return F['comparisons'][a+' -> '+b]['paired_question_average_over_replicas']
def ci(x):return f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
def shape(n,ph,key):return float(np.mean([R[h]['arms'][n]['shape'][str(ph)][key]['mean'] for h in R]))
def rootwork(n,ph,rank):return float(np.mean([R[h]['arms'][n]['forest']['by_priority_rank'][f'p{ph}_r{rank}']['processed_mean'] for h in R]))
parts=['# 동일 생성 예산의 tree 선택: 최종 AL 분석',
'작성: 2026-10-14. 대상: 이번 서버와 eslab18의 q-only tree 비교 14개 full 실행. **전체 실행·원시 결과 검증·최종 분석 완료.** 기존 자동 진행 보고서보다 이 문서가 최종 해석의 기준이다.',
'**결론: C=3, 검증 상한 M=32가 현재 가장 실용적인 기준이다.** 전체 말단 경쟁의 AL은 chain 형태 대비 두 반복 평균 +7.44%였다. 마지막 깊이 경쟁도 +5.91% 개선됐다. 전체 말단과 마지막 깊이의 우열은 이번 표본으로 확정할 수 없다. C=8/M32는 깊이가 잘려 악화됐고, M64로 늘리면 회복되지만 C3/M32를 넘는 추가 AL 이득은 확인되지 않았다.',
'[전체 그림 PDF](figs/ALL_FIGURES.pdf) · [수치 CSV](FINAL_TABLE.csv) · [추가 통계와 원시 파일 해시](FINAL_ANALYSIS.json) · [문항별 지표](QUESTION_METRICS.csv)',
'## 1. 무엇을 검증했는가',
'이번 비교는 같은 root 선정 규칙 아래 **후속 tree를 어떻게 확장할 것인가**를 검증한다. 기존 논문 DUET 대 SSD/Mirror-SD 비교가 아니며, root 점수 자체의 우열이나 calibration 제거 전후를 단독으로 검증하는 실험도 아니다.',
tab(['항목','실제 조건'],[
['모델','LayerSkip-Llama2-7B + AMD-Llama-135m, full BF16'],
['입력','논문 입력 ID에 맞춘 Spec-Bench 480문항/560턴, 6개 하위 과제 각각80문항'],
['출력/문맥','출력 상한1024, 문맥 상한2048, 입력 절단 없음'],
['온도/배치/exit','T=0.7, B=1, exit21'],
['P1/P2','K1=K2=8, 각 phase root8개, 매 forward 입력8행'],
['생성 예산','각 phase 실제 활성 입력64행; P1+P2 본문128행'],
['작은/큰 분기','C=3과 C=8; C=1은 chain 모양 대조군'],
['검증 상한','선택된 root 아래 최대 M=32; C8은 M=64 추가'],
['Miss','기존 JIT chain K2=8; miss tree나 mixed-miss AR로 바꾸지 않음'],
['반복','local4090 seed16101, eslab18 RTX3090 seed26101; 각 서버에7개 비교군'],
['집계','턴 내 step 평균 → 문항 내 턴 평균 → 480문항 평균'],
]),
'MT-Bench는 기존 논문 실행에 맞춰 각 raw 턴을 독립 입력으로 사용한다. 앞선 생성 응답을 다음 턴에 붙이는 평가와 구분한다. 모든 비교군은 같은 unified tree 실행기이며, C1 대조군도 chain 전용 fast path가 아닌 **tree 실행기의 chain 모양**이다.',
'7×2=14개 실행, 총7,840턴이다. 같은480문항을 반복했으므로 독립 문항이6,720개라는 뜻은 아니다. 유효 검증 step은 총'+f"{sum(a['population']['eligible_verification_events'] for h in R.values() for a in h['arms'].values()):,}"+'개이지만 통계의 재표집 단위는 문항이다. 저장공간 부족2건과 외부 GPU 작업이 관측된1건은 최초 결과를 제외하고 재실행 결과만 사용했다.',
'**동일 생성 입력 수 ≠ 동일 전체 계산량/latency.** 본문 draft forward는64행으로 같지만, 샘플링 자식 수는 phase당64C개이고 glue·tree 관리·CPU export·target 검증 입력은 다를 수 있다. Chain은 M32 설정이어도 root별 생성 노드가8개여서 실제 hit 검증도8노드다. C3 tree는 보통24~28노드를 검증한다.',
'## 2. 실제 수식과 경쟁 범위',
r'''후속 노드 \(u\)의 선택 점수는 다음이다.

\[
S(u)=s_i\prod_{j\in\operatorname{path}(i\to u)}q(x_j\mid\operatorname{prefix}_j).
\]

- \(s_i\): 아직 확정되지 않은 verification outcome 후보 root \(i\)의 점수.
- P1: **draft 문맥 도달 확률 × 해당 root 토큰의 draft 확률**. 여러 위치의 후보에서 전체 상위8개를 택한다.
- P2: 현재 개선 root 점수인 **위치 점수 × 정규화한 \(e(1-q)\)**. 위치는 기존 \(\hat h\), \(\bar h\) 기반 혼합(0.75/0.25), 토큰 정규화는 full vocabulary를 사용한다. Tree 문맥에서는 구현된 ladder의 종료 질량을 사용하므로 단순 chain hazard와 동일한 코드라고 부르지 않는다. P1에서 이미 선택한 동일 위치·토큰은 제거한다.
- Calibration 수락 확률·gain 표는 사용하지 않았다. \(q\)는 실제 T=0.7의 draft 확률이다.

이 점수는 **계산 우선순위의 대리값**이다. 실제 target의 수락 확률 또는 기대 AL과 같다는 정리는 아니다. 실제 target acceptance/correction은 native verifier가 별도로 수행한다.''',
tab(['방식','각 round에서 경쟁하는 후보','주경로 보호'],[
['Chain C1','root마다 유일한 다음 노드; W=R이므로 root별1개씩 전진','각 root가8회 전진'],
['마지막 깊이 Last','직전 깊이의 미확장 노드 중 root 구분 없이 상위8개','특정 root에 최소1개를 배정하지 않음'],
['전체 말단 Frontier','이전 깊이에 남은 미확장 노드까지 포함해 상위8개','이전에 밀린 후보가 나중에 다시 경쟁 가능'],
]),
'사용자의 **q-global은 Last**다. Last도 root마다 주경로를 보장하지 않는다. 한 root가 어떤 깊이에서 입력8개 경쟁에 못 들면 다음 round에는 그 root의 이전 깊이 노드가 경쟁에서 빠진다. Frontier는 그 노드를 나중에 다시 확장할 수 있다. 생성이 끝난 뒤 점수 rerank는 하지 않고 native 생성 순서 prefix만 검증한다.',
'## 3. 전체·P1·P2·miss 결과',
'다음은 두 서버/seed의 문항 평균을 동일 가중치로 평균한 요약이다. AL은 **확정 root/target 토큰1개를 포함**한다. P1/P2/miss AL은 각 source에 해당한 조건부 값이며, source가 관측된 문항의 분모가 방법마다 다르다.',
tab(['방법','전체 AL','P1-hit AL','P2-hit AL','Miss AL','Hit (%)','Chain 대비 AL'],[
[L[n],*[f'{avg(n,k):.3f}' for k in ('al','p1_al','p2_al','miss_al')],f"{avg(n,'hit_percent'):.2f}",f"{100*(avg(n,'al')/avg(N[0],'al')-1):+.2f}%"] for n in N]),
'![전체 AL](figs/01_overall_AL.png)',
'![조건부 AL](figs/02_phase_AL.png)',
'서버별 수치도 같은 방향이다. 서버와 seed를 동시에 바꾸었으므로 둘의 효과를 분리 추정하거나 서버 간 TPS를 직접 비교하지 않는다.',
tab(['서버/seed','방법','전체 AL','P1 AL','P2 AL','Miss AL','Hit (%)'],[
[('local/16101' if h=='local' else 'eslab18/26101'),L[n],*[f'{q(h,n,k):.3f}' for k in ('al','p1_al','p2_al','miss_al')],f"{q(h,n,'hit_percent'):.2f}"] for h in R for n in N]),
'## 4. 어느 차이까지 믿을 수 있는가',
'같은 문항의 두 반복을 먼저 평균하고, 문항 ID480개를 paired bootstrap 5,000회 재표집했다. 아래95% 구간은 다중 비교 보정 전의 탐색적 구간이다. 임의의 새 seed·모델까지 일반화하는 구간이 아니다.',
tab(['전 → 후','AL 차이','상대 변화','95% 구간'],[
[L[a]+' → '+L[b],f"{comparison(a,b)['delta']:+.4f}",f"{comparison(a,b)['relative_percent']:+.2f}%",ci(comparison(a,b))]
for a,b in [(N[0],N[1]),(N[0],N[2]),(N[1],N[2]),(N[3],N[4]),(N[3],N[5]),(N[4],N[6]),(N[2],N[6])]]),
'- **C3 tree 대 chain:** 두 서버 각각의 AL 차이 구간도0보다 높다. Last는 local +0.161 / eslab18 +0.245, Frontier는 +0.186 / +0.325다. 개선 방향은 반복됐다.\n- **Frontier 대 Last:** C3/M32 평균 +0.0524지만 구간이0을 포함한다. 전체 말단이 확실히 우월하다고 확정할 수 없다. 반대로 전체 말단이 깊이를 망쳐 불리하다는 주장도 이번 결과로 지지되지 않는다.\n- **C8/M64 대 C3/M32:** Frontier끼리 차이 +0.0018, 약+0.05%다. local에서는 +0.0340, eslab18에서는 −0.0304로 방향도 달랐다. 검증 상한을2배로 늘려 더 좋은 AL을 얻었다는 근거가 없다.',
'![Paired 비교](figs/04_paired_effects.png)',
'## 5. C8/M32가 나빠진 이유: 검증 prefix가 깊이를 자른다',
r'''이번 구현에서 한 부모를 확장하면 C개 자식이 생성 순서대로 저장된다. 깊이 d인 노드 하나를 prefix 안에 남기려면, 앞선 d−1개 조상 확장에서 생성한 형제 묶음도 모두 앞에 있다. 따라서 필요한 최소 prefix 길이는

\[
M\ge C(d-1)+1.
\]

즉, K회 생성 후 검증에 남는 최대 draft 깊이는

\[
D_{\mathrm{served}}\le\min\left(K,\left\lfloor\frac{M-1}{C}\right\rfloor+1\right).
\]

이는 **이번 고정 C·생성 순서 prefix 선택의 구조적 상한**이다. 모든 tree 알고리즘이나 다른 pruning 방식의 상한은 아니다. 다른 분기를 함께 확장하면 실제 깊이는 더 낮아질 수 있다.

- C8/M32: 깊이≤4. 따라서 cache hit의 AL≤5. K1/K2를8로 늘렸어도 깊이8까지 검증할 수 없다.
- C8/M64: 깊이≤8.
- C3/M32: prefix만의 상한은11이므로 이번 K=8을 막지 않는다.
- Miss는 별도 chain K=8이므로 C8/M32여도 AL≤5라는 제한을 적용하지 않는다.''',
'이 상한을 모든 유효 hit event에 대조했으며 위반은0건이었다. 따라서 C8/M32 결과로 “넓게 만드는 것이 본질적으로 나쁘다” 또는 “8회 depth를 충분히 시험했다”고 결론내리면 안 된다.',
r'''이 식은 파라미터를 고를 때도 바로 쓸 수 있다. 생성 순서 prefix를 유지하면서 깊이 K인 노드가 **하나라도 포함될 가능성**을 열어 두려면

\[
C\le\left\lfloor\frac{M-1}{K-1}\right\rfloor \quad(K>1)
\]

이어야 한다. K=8, M=32에서는 C≤4가 필요하다. C=8로 깊이8까지 남기려면 최소 M≥57이 필요하다. 다른 분기도 예산을 쓰므로 이는 필요조건이지 충분조건이나 최적 C를 정하는 공식은 아니다. 실제로 C2/C4는 이번 full에서 비교하지 않았다.''',
tab(['방법','P1 hit 깊이','P2 hit 깊이','P1 hit 노드','P2 hit 노드','P1 깊이 끝 도달','P2 깊이 끝 도달'],[
[L[n],f"{shape(n,1,'tree_max_depth'):.2f}",f"{shape(n,2,'tree_max_depth'):.2f}",f"{shape(n,1,'valid_k'):.2f}",f"{shape(n,2,'valid_k'):.2f}",f"{100*shape(n,1,'reached_depth_ceiling'):.1f}%",f"{100*shape(n,2,'reached_depth_ceiling'):.1f}%"] for n in N]),
'표의 구조 통계는 실제 hit event 가중 평균을 각 서버에서 구한 뒤 평균했다. 앞의 문항 가중 AL과 가중 방식이 다르다. “깊이 끝 도달”은 수락한 경로 길이가 해당 tree의 최대 깊이와 같았다는 뜻이며, 뒤를 늘리면 추가 토큰을 반드시 수락한다는 뜻은 아니다.',
'Frontier C8에서 M32→M64로 바꾸면 전체 AL은2.925→3.694, 약26.3% 증가한다. P1 깊이는3.24→5.91, P2는2.64→3.94로 늘었다. **상한을 완화하면 AL이 크게 회복된다는 온라인 증거**다. 다만 생성 경로·context도 달라져 동일 tree를 두 번 검증한 counterfactual 실험은 아니다.',
'C8/M64의 P1 AL4.661은 C3/M32의4.528보다 높지만, P2 AL3.440은3.586보다 낮다. 두 source가 다른 조건부 집합이므로 이를 그대로 더해 P1=C8/P2=C3 혼합의 이득을 예측할 수 없다. 그 조합은 후속 검증 후보이다.',
'![깊이와 상한](figs/03_depth_and_cap.png)',
'## 6. 후보 점수에 따른 계산 배분은 실제로 이루어졌는가',
'총'+f"{sum(a['forest']['calls'] for h in R.values() for a in h['arms'].values()):,}"+'개 exported phase forest(워밍업·마지막 step 작업 포함)에서 물리 입력뿐 아니라 **활성 입력도 모두64행**임을 확인했다. Root 기록 누락 및 생성 수=처리 입력 수×C 불일치는0건이다. 상위 점수 root에 더 많은 forward 행이 배정됐다.',
tab(['방식','P1 1위 / 8위 root 입력 수','P2 1위 / 8위 root 입력 수'],[
[L[n],f'{rootwork(n,1,0):.2f} / {rootwork(n,1,7):.2f}',f'{rootwork(n,2,0):.2f} / {rootwork(n,2,7):.2f}'] for n in N[:3]]),
'Chain은 각 root8행으로 균등하지만 C3에서는 상위 root에 더 많은 계산을 쓴다. 특히 P2에서 분배 차이가 크다. 이는 **원하는 예산 배분이 실행됐다는 증거**다. 후보 점수를 균등하게 바꾼 대조군이 없으므로, AL 개선 중 얼마가 root 점수 가중 때문인지 단독 인과 효과는 아직 모른다. C3에 root별 주경로를 강제로 보장한 별도 대조군도 없다.',
'![Root 예산 배분](figs/05_root_budget_allocation.png)',
'## 7. Hit는 낮아졌는데 왜 전체 AL은 올랐는가',
'Chain→Frontier C3/M32에서 평균 hit는81.88%→77.42%로 낮아졌다. P1 비중은48.10%→27.54%, P2는33.77%→49.88%, miss는18.12%→22.58%로 바뀌었다. 반면 hit 시 AL은 P1 4.033→4.528, P2 2.899→3.586으로 높아졌다.',
'Root 수식은 비교군 사이에서 고정했지만, tree 구조·수락 경로가 달라지면 다음 context와 위치 후보 집합, P1/P2 중복 제거 결과도 달라진다. 따라서 온라인 cache hit까지 같아야 하는 실험은 아니다. 이 결과만으로 root 수식이 나빠졌다고 해석하거나 조건부 P1/P2 AL 변화 전체를 순수 tree 효과라고 해석하지 않는다.',
'아래는 각 턴에서 source별로 확정한 토큰 수를 전체 step 수로 나눈 뒤 문항·반복 평균한 **정확한 AL 기여분**이다. 세 열을 더하면 전체 AL이 된다. 전체 평균 hit×전체 평균 조건부 AL을 곱하면 일반적으로 이 합과 일치하지 않는다.',
tab(['방법','Miss 기여','P1 기여','P2 기여','합계 AL'],[
[L[n],*[f'{v:.4f}' for v in np.mean([F['phase_contributions'][h][n] for h in R],axis=0)],f"{avg(n,'al'):.4f}"] for n in N[:3]]),
'P1 기여가 줄고 P2 기여가 늘어나는 방향이다. Miss 처리 자체는 바꾸지 않았는데 miss AL도2.493→2.616으로 달라졌다. 이는 miss가 발생한 문맥·문항 분포가 달라진 효과를 포함하며 miss 알고리즘 개선으로 보고하지 않는다.',
'## 8. 데이터셋별로도 같은가',
tab(['하위 과제','Last C3−Chain','Frontier C3−Chain','Frontier C8/M32−Chain','Frontier C8/M64−Chain'],[
[g,*[f"{np.mean([R[h]['arms'][n]['question_quality'][g]['al']['mean']-R[h]['arms']['chain_m32']['question_quality'][g]['al']['mean'] for h in R]):+.3f}" for n in [N[1],N[2],N[4],N[6]]]]
for g in sorted(k for k in R['local']['arms'][N[0]]['question_quality'] if k!='all')]),
'C3는 두 반복 평균으로6개 과제 모두에서 chain보다 높았다. C8/M32는 QA/RAG에서 크게 하락했지만 summarization에서는 소폭 높았다. 따라서 폭을 늘리는 선택의 손익은 데이터셋에 따라 달라지며, 전체 평균만으로 모든 문항에 같은 최적 구조가 있다고 주장하지 않는다. 하위 과제 수치는 탐색적 평균이며 각각의 유의성을 검증했다는 뜻은 아니다.',
'![과제별 차이](figs/06_task_AL.png)',
'## 9. 문맥 종료·통계·정확성의 검토 범위',
'- 각 arm480문항/560턴, 입력 UID 순서와 입력 token ID, 출력 상한, runtime 및 adapter hash를 검증했다. 두 서버 사이의 소스 hash도 일치한다.\n- 모든 arm에서 무디코딩 턴은0개다. 생성 도중 context_limit_guard로 끝난 턴은 남아 있다. 어느 arm에서든 문맥 보호 종료가 발생한34문항을 공통 제외한446문항에서도 Frontier C3의 chain 대비 이득은+0.252 AL, 약+7.16%, 95% 구간[+0.138,+0.361]로 유지됐다. 이 제외 결과는 민감도 분석이며 원래 full 결과를 교체하지 않는다.\n- 출력 상한 도달 또는 실제 반환량보다 긴 수락 step은 AL 집계에서 제외했다. 원시 출력·실행 기록은 보존했다.\n- P1/P2 조건부 AL은 해당 source를 관측한 문항만 계산한다. 분모는 `RESULTS.json`과 `QUESTION_METRICS.csv`에 있다.\n- 마지막 깊이/전체 말단 변경은 미래 확장 선택에 적용했고 original q와 ordered-WOR 샘플 순서를 유지했다. 사후 점수 pruning으로 proposal을 바꾸지 않았다. 기존 회귀·proposal q·graph parity와 소규모K8/C1·3·8 검증을 통과했다. 이 full AL 비교 자체가 losslessness의 통계적 증명은 아니다.\n- 추가 분석의 paired 비교·문항/턴 가중·source 기여 합계·cap 제외·forest 예산 검사는5개 CPU 테스트로 통과했다.\n- 실제 target 분포에서 버려진 노드를 재검증한 oracle 결과는 없다. Trace는 실제 hit root의 identity와 완전한 미선택 subtree logits를 연결하지 않으므로 “이 노드를 남겼다면 AL이 얼마 더 증가했다”는 정확한 사후 counterfactual은 만들 수 없다.',
'## 10. 무엇을 채택하고 무엇을 더 확인할 것인가',
tab(['판정','내용','근거/남은 검증'],[
['유지할 방향','Calibration 없이 root 점수×draft 경로 확률','이 구성에서 C3 tree의 AL 개선 확인. Calibration 대비 직접 우위는 미측정'],
['우선 기준','C3 / M32','Chain보다 AL 상승, C8/M64와 사실상 같은 AL을 더 적은 검증 노드로 달성'],
['단일 후보를 고른다면','Frontier C3/M32를 후속 기준 후보로 권고','관측 평균은 Last보다 높음. Last 대비 우위 확정 아님; 간결성을 우선하면 Last C3도 근거 있음'],
['현재 비권고','C8 / M32','생성 순서 prefix가 깊이를≤4로 제한; 두 반복 모두 크게 하락'],
['기본값으로 확대할 근거 부족','C8 / M64','C3/M32보다 추가 AL 이득 확인 안 됨'],
['별도 검증 필요','후보 점수 가중 vs 균등 가중','Contribution의 독립 효과를 분리하는 핵심 대조'],
['별도 검증 필요','C3 주경로 보호 / Last / Frontier','현재 C1 대조만으로 주경로 보호 여부의 단독 효과를 분리 못함'],
['후속 후보','C2, P1/P2 서로 다른 C, B>1','이번 full에 포함하지 않았음'],
['보류','Expand-then-rerank','AL 개선 여지는 있지만 실제 수익과 stochastic 정확성 모두 미검증'],
]),
'Rerank를 하려면 어떤 tree가 좋은지와 그 tree를 어떤 확률 법칙으로 샘플링·검증할지를 함께 설계해야 한다. 이미 샘플한 형제의 점수만 보고 상위 노드를 남기면 verifier가 가정한 proposal 법칙이 달라질 수 있다. 부모 연결·형제 순서를 보존하는 것만으로 모든 사후 pruning이 자동으로 정당화되는 것도 아니다. 우선 샘플링 전 확장 예산/자식 수 결정, 또는 별도 정확성 증명이 가능한 선택 방식으로 한정하는 것이 타당하다.',
'이번 결과만으로 “모든 파라미터가 최적” 또는 “AL 증가가 TPS 증가로 이어진다”고 주장하지 않는다. K8/8과 exit21은 depth 비교를 위한 고정 조건이다. Phase 시간 예산·overlap 최적화와 B>1은 후속 대상이며, 이 보고서를 만들면서 production 기본값을 추가로 변경하지 않았다.',
'## 11. 보조 timing과 재현 자료',
'아래 timing은 실제 온라인 입력 shape·경로·CPU export가 달라지는 실행의 step 가중 관측값이다. AL 중심 실험에 남긴 참고 자료이며 개별 kernel/target/draft 최적화 효과나 서버 간 속도 우위의 근거로 사용하지 않는다. 질문 가중 AL과 step 가중 TPS의 모집단이 다르다.',
tab(['서버','방법','Step 가중 AL','Verify ms','Step ms','관측 TPS'],[
[h,L[n],*[f"{R[h]['arms'][n]['auxiliary_timing'][k]:.3f}" for k in ('al','verify_ms','step_ms','tps')]] for h in R for n in N]),
'재현 파일: `PLAN.md`, `PAPER_PROTOCOL.json`, `MODEL_SHA256.json`, `SOURCE_SHA256.json`, 각 campaign manifest, 원시 JSON/forest gzip/telemetry. `FINAL_ANALYSIS.json.raw_manifest`는 선택한14개 원시 JSON의 경로와 SHA256을 담는다. 실패·외부 간섭 실행은 최종 데이터에 들어 있지 않다.',
'''```bash
cd /home/chokwans99/PSD-MLSys-ver0.1/results/mlsys_ver02/equal_budget_20261014
/home/chokwans99/PSD/ssd/.venv/bin/python -B final_analysis.py
/home/chokwans99/PSD/ssd/.venv/bin/python -B -m unittest -v test_analysis test_final_analysis
/home/chokwans99/PSD/ssd/.venv/bin/python -B make_final_figs.py
/home/chokwans99/PSD/ssd/.venv/bin/python -B write_final_report.py
```''',
'`RESULTS.json`은 기존 `analyze.py`가14개 전체 forest를 검증한 결과다. `FINAL_ANALYSIS.json`은 추가 paired 비교·공통 문맥 종료 민감도·source 기여분·prefix 깊이 상한을 독립적으로 대조한 결과다. Figures는 두 파일에서 생성했다.',
'실험은 `MLSys-ver0.2@76c98f9` 이후 고정된 q-only 연구 작업 트리로 수행했다. 이후 별도 브랜치 `fix/mlsys-ver02-blackwell-review`에 반영한 수치 일관성·설정·workspace 수정이 이번 데이터에 적용됐다고 소급 표시하지 않는다. 그 수정과 연구 코드를 통합한 뒤에는 별도 회귀 검증이 필요하다.',
]
(P/'FINAL_REPORT.md').write_text('\n\n'.join(parts)+'\n')
print('Wrote FINAL_REPORT.md')
