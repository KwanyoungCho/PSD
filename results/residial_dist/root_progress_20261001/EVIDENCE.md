# Root 후보 보고서의 근거와 부록

## 이번에 추가한 작업

- 2026-10-01: 기존 9월 confirmation NPZ를 CPU로 재집계. 새 GPU 생성/모델 학습/정책 선택은 수행하지 않음.
- 산출물: [REPORT.md](REPORT.md), [summary.json](summary.json), [tables.csv](tables.csv), PNG/PDF 그림3개, [재생성 코드](make_report_assets.py).
- 96개 prompt(Alpaca/C4/GSM/HumanEval 각24개), seed271, T=1/.7/.5.
- 저장 step 수: 1,144/1,033/935, 합계3,112. Full generation의 매8번째 step을 저장한 자료. Cap256, AWQ70B/TinyLlama, exit index56, B1, 실제 K4, P1 off.
- 전체 Spec-Bench/full dataset 평가나 dense 모델 평가가 아님. 서로 다른 T에 같은 96개 prompt를 사용.

## 2×2 표의 정확한 정의

- 대상 root는 `(위치, token)` 쌍. 실제 첫 거절 확률 $h_i$, 해당 위치의 true residual $R_i$, bonus 위치의 $R_K=p_K$를 이용해 $C(S)=\sum_{(i,v)\in S}h_iR_i(v)$ 평가.
- **고정 분배:** 4개 correction 위치 + bonus 위치에 정확히 top3씩, 총15개. 저장된 각 source의 local top17 pool에서 앞3개를 사용. 관측 draft token 제외는 양쪽에 동일 적용.
- **기존 위치 배분:** 각 위치의 top17 점수를 그 top17 합으로 정규화하고 기존 $\widehat h_i$를 곱한 뒤 전역 top15 선택.
- 정책 이름만으로 고정 quota 여부를 판단하지 않음. 균등 h × 전역 top15와 위치당3개는 다르지만, 초기 exit_probe의 unif는 실제로 round-robin 선택이며 K+1=5/root15에서 위치당3개임. 이전 초안에서 초기 unif도 고정 분배가 아니라고 한 설명은 정정. 초기 M64/다른 자료의 수치를 이번 M17/96-prompt 표와 혼합하지 않음.
- 기존 배열 `local::{source}`와 `row::h`의 곱을 합해 fixed3 산출. Dynamic 값은 기존 `policy::{source}__topm__original`을 사용. 정답 p는 평가에만 사용.
- Point estimate는 step→prompt→dataset 균등 평균. 각 dataset 안에서 prompt를 paired bootstrap4,000회(seed20261001). CI는 pointwise/탐색적이며 다중비교 보정 결과가 아님.
- 기존 CSV의18개 정책/온도별 평균과 오차1e−12 이내 일치 확인. 개별 CI는 재집계 bootstrap seed 때문에 과거 CI와 소폭 다를 수 있음.
- 저장 replay의 전역 top15 결과이며 GPU top17 전송 후 앞15개와 동점 순서가 다를 수 있음. 큰 차이의 구조적 진단에 사용하고, 작은 token 식 이득은 별도 wire 검사 우선.

### 같은 배분에서 residual과 proxy의 차이

단위 %p, residual−proxy. 아래 CI는 새 집계의 pointwise95% 구간.

| T | 고정 배분 | 기존 위치 배분 |
|---|---:|---:|
| 1.0 | −2.556 [−3.481,−1.721] | −3.278 [−4.296,−2.345] |
| .7 | −3.905 [−4.943,−2.937] | −3.767 [−4.699,−2.877] |
| .5 | −3.563 [−4.493,−2.682] | −3.831 [−4.633,−3.055] |

- Residual+기존 배분 대 proxy+고정 배분: T1 +2.356 [1.105,3.602], T.7 −.772 [−2.174,.642], T.5 −.735 [−2.062,.577].
- T1의 조합 개선을 residual 자체의 우위로 설명하면 안 됨. 배분을 맞추면 proxy가 더 높음.
- 9/10 shared-review 자료에도 실제 fixed3 통제가 있음: 전체 prompt 평균 residual56.55%, proxy60.27%, 기존 배분60.06%/64.01%. 이 128-prompt 자료와 이번96-prompt의 절대값을 섞지 않음.

## 언제 residual이 유리하거나 불리했는가

근거: [direct_comparison/REPORT.md](../direct_comparison/REPORT.md), 별도 새128 prompts, T1.

- Global15 coverage residual61.297%, proxy65.618%, 차이−4.321%p.
- Step 승패: residual 승40.50%, 패42.21%, 동률17.29%. 평균 기여는 승+1.029%p, 패−5.350%p.
- **다른 지표인** rejection-event 가중 local top3 분해:
  oracle headroom13.016 − 후보 삭제10.665 − 순위 손실7.360 = −5.009%p.
- 전체 residual support 누락20.658%는 위 budget 기준 삭제 손실10.665%p와 다름. 꼬리 질량도 포함하므로 혼용하지 않음.
- Z<.1인 위치는 rejection-event 질량의2.094%. Z≥.6에서도 local residual−proxy는−3.742%p. 정규화 증폭만으로 전체 패배를 설명할 수 없음.
- $\delta<.2,Z\geq.4$ 구간은 local+0.109%p, $\delta\geq.2,Z<.4$는−9.527%p. 사후 oracle 조건이며 실행 가능한 gate/유의한 양의 threshold를 발견한 결과가 아님.
- 실제 오차 방향을 보존한 oracle $e_\gamma=(1-\gamma)e+\gamma p$에서 γ=.5의 gap+0.999%p, γ=1의+2.627%p. 양쪽 source와 h를 같은 $e_\gamma$로 재계산한 통제. p를 사용하므로 배포 정책이 아님.
- 초기 all-layer 그림은 뒤 layer에서 residual 역전이 보이는 자료이나, exit 지연/남은 draft 시간까지 고려한 최적 layer 결론은 아님.

### 조건별 승패와 해석

128-prompt 독립 검증, T1/exit56. 아래는 bonus를 제외한 local top3의 reject-event 가중 평균이며, 전역 root15의 prompt 균등 평균과 구분.

| 조건 | 전체 reject-event 질량 비중 | Local residual−proxy |
|---|---:|---:|
| $\delta<.2,\ Z\geq.4$ | 12.149% | +0.109%p |
| $\delta<.2,\ Z<.4$ | 14.895% | −3.666%p |
| $\delta\geq.2,\ Z\geq.4$ | 48.747% | −4.451%p |
| $\delta\geq.2,\ Z<.4$ | 24.208% | −9.527%p |

- 낮은 오차/큰 mass에서 손실이 줄어드는 경향. 첫 행은 거의 동률인 작은 양의 평균이며, 유의한 우위로 판정한 결과가 아님.
- $\delta,Z$를 계산하려면 target p가 필요. 실행 중 쓸 수 있는 e,q 기반 entropy/peak/argmax/TV 등의 13개 feature로 단일 임계값 및 Ridge 선택기를 별도 검증했지만, 모두 always-proxy를 선택해 추가 이득0.
- 이는 시험한 선택기가 실패했다는 뜻이며, 모든 gate가 불가능하다는 증명은 아님.
- 실제 $Z<.1$ 구간이 차지하는 reject-event 질량은2.094%에 불과. “draft가 target을 잘 맞추는 드문 상황만 제거하면 해결된다”는 해석도 지지되지 않음.
- Entropy로 나눈 결과: proxy가 더 평평한 $H(e)-H(p)>.05$ 구간(48.208%)에서 local−7.405%p, 더 뾰족한 $<-.05$ 구간(42.676%)에서도−3.406%p. 평탄화만의 문제로 설명할 수 없음.
- h만 참값으로 바꾸면 전역 residual−proxy는−6.640%p. Source를 정확한 p로 바꾸고 hhat를 유지하면+2.562%p. 현재 residual의 상대적 열세를 뒤집는 데 source의 확률·순위 품질이 중요함.

### 후보 교환의 정확한 승패 조건과 후속 측정

[training_free/THEORY.md §2](../training_free/THEORY.md), [REPORT.md §5](../training_free/REPORT.md), [diagnostics.json](../training_free/diagnostics.json) 근거.
여기는 앞의128개와 다른96개 확인 prompt, T1/.7/.5의 local top3 분석.

- Residual 후보를 $S_r$, proxy 후보를 $S_e$라 하면 새로 넣은 집합 $I=S_r\setminus S_e$, 뺀 집합 $O=S_e\setminus S_r$.
- $a=[p-q]_+$, $b=[e-q]_+$, $\eta=b-a$, $Z=\sum a>0$일 때

$$
\Delta=C(S_r)-C(S_e)=R(I)-R(O)=\frac{\Gamma-E}{Z},
\quad
\Gamma=b(I)-b(O),\quad E=\eta(I)-\eta(O).
$$

- **정확한 판정:** $\Gamma>E$면 승, $\Gamma<E$면 패. $\Gamma$는 추정 교환 이득, $E$는 그 이득에 섞인 점수 오차. 이는 상한이나 경향이 아닌 항등식.
- 실제 이득이 있으려면 proxy 후보와 다른 후보를 선택할 가치가 있어야 하고, 그 가치를 proxy 오차가 훼손하지 않아야 함. 큰 Z만으로 개선이 보장되지는 않음.

| T | 추정 교환 이득 $\Gamma/Z$ | 교환 오차 $E/Z$ | 실제 차이 $\Delta$ | 뺀 후보 오차 $\eta(O)/Z$ |
|---|---:|---:|---:|---:|
| 1.0 | +3.834%p | +7.362%p | −3.528%p | −6.422%p |
| .7 | +2.223%p | +8.800%p | −6.576%p | −8.467%p |
| .5 | +1.183%p | +8.453%p | −7.270%p | −9.029%p |

- 가장 큰 오차 항은 **뺀 후보의 과소평가**. “새로 넣은 token이 과대평가됐다”만으로 설명하면 중요한 원인을 놓침.
- T1 local 사건 비중: 승35.61% / 패24.56%, 승리 시+7.39%p / 패배 시−25.08%p. T.7도 승리 시+5.61%p / 패배 시−35.37%p.
- 위 수치는96-prompt local 사건 가중이며,128-prompt global step 승40.50% / 패42.21%와 분모·자료가 다름.
- 개선 방향: 필요한 proxy 후보를 과도하게 버리는 손실을 줄이는 것. $e(1-q)$는 이를 완화하는 후보식이지만, 이 진단 자체가 proxy-only 대비 추가 이득의 증명은 아님.

### 초기 “73%에서 이긴다” 분석의 정확한 범위

- 그림 생성 코드와 원본12개 JSON(34,150 step)을 다시 확인. 생성 코드는 [make_figs.py](../make_figs.py), 누적 방식은 [exit_probe.py](../../../ssd/ssd/engine/helpers/exit_probe.py)의 bins 부분.
- x축 overlap은 $1-\mathrm{TV}(p,q)=1-Z$. 오른쪽일수록 draft가 target에 가깝고 정규화 전 residual mass가 작음.
- y축은 $\sum_v R_v\widehat R_v-\sum_v R_v\widetilde e_v$. $\widehat R,\widetilde e$ 모두 관측 draft token을 제외한 뒤 전체 어휘에서 재정규화. 실제 top-k 후보 선택이나 전역 예산은 이 내적에 사용하지 않음.

| Overlap 구간 | 그림의 가중 비중 | 내적 차이 residual−proxy |
|---|---:|---:|
| 0–.25 | 8.20% | +.01770 |
| .25–.50 | 22.07% | +.01811 |
| .50–.75 | 42.55% | +.00939 |
| .75–1 | 27.19% | −.04413 |

- **가중 방식:** step마다 $w_i=h_i/\sum_{j<K}h_j$로 거절 위치 가중치를 정규화한 뒤 step 누적, run별 구간 평균을 계산하고12개 run 균등 평균. 따라서 “27% of reject mass”를 전체 생성에서 발생한 거절 횟수의27%로 해석하지 않음.
- **허용되는 결론:** 이 지표의 구간 평균에서는 높은 draft–target overlap 구간에 음의 차이가 나타남. 작은 residual 신호에 비해 proxy 오차가 클 수 있다는 실패 기전과 부합.
- **허용되지 않는 결론:** 실제 root 후보 성능이 이 구간에서만 나쁘다거나, 나머지 구간의 모든 context에서 좋다는 주장. 그림 제목 “We only lose where the draft is already good”는 실제 후보 정책의 보편적 승패 결론으로는 과도함.
- 초기 보고서의 합계−.00256은 구간 비중의 run 평균과 gap의 run 평균을 곱한 근삿값. 각 run 안에서 곱한 뒤 평균한 값은−.0031839. 평균들의 곱과 곱의 평균은 다르므로 이를 정확한 전체 손익 분해로 재사용하지 않음.
- 보고용 권장 캡션: **“높은 draft–target overlap 구간에서 proxy residual의 분포 내적 열세 관측. 실제 후보 coverage의 조건별 우열은 별도 검증.”**


- [초기 REPORT §6](../REPORT.md), [06_where_we_lose.png](../figs/06_where_we_lose.png)는 draft–target overlap $1-\mathrm{TV}(p,q)$를 네 구간으로 나눈 **분포 내적** 진단.
- 세 양의 평균 구간이 차지하는 가중 질량은 합72.8%, overlap≥.75인 음의 평균 구간은27.2%.
- 이는 “전체 context의73%에서 top-k 후보가 이긴다”는 뜻이 아니며, 각 구간 안의 모든 context가 이겼다는 뜻도 아님.
- 초기 문서의 overlap은 target p를 필요로 하므로 early-exit 시점의 관측 가능한 gate로 그대로 사용할 수 없음.
- 초기 문서의 “작은 residual이면 필연적으로 실패”, “이 구간만 피하면 해결” 같은 강한 해석은 후속 coverage/선택기 검증으로 뒷받침되지 않음. 교수님 보고에는 위 조건별 coverage 및 정확한 손실 분해를 우선 사용.
- 초기 all-layer 비교는 dataset별 평균 교차 layer가 약74–77임을 보여줌. 이 수치를 모든 모델의 보편적 threshold나 실제 P2 실행 불가능 경계로 사용하지 않음.

### 추가: 같은 자료에서 두 지표를 직접 대조

2026-10-01 사용자 요청에 따라 [DISTANCE.md](DISTANCE.md)와 그림2종을 추가. 이전 설명에서 필요하다고 한 비교 중 **후속128개 입력의 동일 context에서 내적/coverage를 대조하는 작업**을 완료.

- Confirmed source rows33–64, 4datasets×32prompts, seed7/T1/exit56. 저장1,509step의6,036correction위치 사용.
- x축 TV(p_T,p_D)를 .25 간격으로 분할, 같은 pooled h 가중치 적용. 저장GPU top3 coverage와 원시 full-vocab 분포의 CPU alignment를 동일 step/seq_id에 연결.
- TV 구간별 coverage 차이: −13.619/−3.814/−3.884/−1.944%p. Alignment 차이: −.04433/+.01065/+.01815/+.01976.
- 따라서 높은 overlap에서만 alignment가 음수인 초기 부호 패턴은 후속 입력에서도 나타남. 같은 context의 top3 coverage는 모든 구간의 평균이 음수이므로 지표 차이가 실제로 중요함을 직접 확인.
- Step 내 h 정규화 민감도에서도 같은 부호 패턴 유지. 초기34,150step 자체의1:1재분석은 아니며 가중 방식별 수치는 분리.
- 네 구간 coverage를 거절 질량으로 합치면 기존 direct-comparison local gap와1e−12 이내 일치. CI는 prompt 단위 paired bootstrap4,000회, pointwise/탐색적.
- [TV–coverage PNG](04_tv_coverage.png) · [PDF](04_tv_coverage.pdf), [동일 overlap 축의 coverage/alignment PNG](05_overlap_coverage_alignment.png) · [PDF](05_overlap_coverage_alignment.pdf).
- 재현: [make_distance_figures.py](make_distance_figures.py), [distance_summary.json](distance_summary.json), [distance_tables.csv](distance_tables.csv).

## 데이터 규모와 신뢰 범위

2026-10-01 점검. 아래는 root 후보 연구의 범위이며 tree 후속 캠페인의 표본 수를 합산하지 않음.

| 단계 | 서로 다른 prompt | 생성/계측 범위 | 독립성 |
|---|---:|---|---|
| 초기 Fig.6 포함 all-layer probe | 128개, 4 datasets 각32개 | 3seed, 12run, 기록34,150step | 반복 seed는 새 prompt가 아님 |
| 9/10 shared-review / 이후 development | 기존128개 | 2seed, 22,990step 중2,995step 저장 | 새 prompt 독립 검증으로 세지 않음 |
| 9/12 direct comparison confirmation | 새128개, 각32개 | seed7, 11,616step 중1,509step 저장; correction6,036위치 | 개발 prompt와 텍스트 중복0 |
| training-free confirmation | 또 다른96개, 각24개 | seed271, T1/.7/.5; 23,854step 중3,112step 저장 | 이전 prompt와 중복0; 세 T는 같은96개 |

- 네 dataset은 Alpaca/C4/GSM/HumanEval의 **부분집합**. 사용자 요청의 full-dataset root 검증이 완료된 상태가 아님.
- 같은 prompt의 여러 step/seed/T를 독립 prompt로 세지 않음. 후속 주 결과의 CI는 dataset별 prompt cluster를 paired bootstrap4,000회 재표집.
- 128-prompt 직접 비교와96-prompt training-free 각각 정책 선택을 먼저 동결하고 새 prompt 결과를 평가. [direct audit](../direct_comparison/audit.json), [training-free audit](../training_free/audit.json)에서 중복/동결 순서/분포 유효성 확인.
- 새 독립 prompt를 쓴 주 정책 비교와, 정답 p로 사후 분할한 조건표의 근거 수준은 다름. $\delta<.2,Z\geq.4$의+0.109%p는 검증된 gate나 유의한 우위가 아닌 기술통계.

### 평균 coverage 열세는 초기부터 일관됨

모두 residual−proxy, global root15/T1/exit56/P1 off. 다른 자료·M·가중 방식이므로 숫자를 하나의 통합 추정치로 합치지 않음.

| 자료 | 평균 차이 | 근거 |
|---|---:|---|
| 초기12run | −3.833%p | 12개run 모두 음수; 원본 JSON 재확인 |
| 9/10 자료를 직접 비교 코드로 재평가 | −3.957%p | 95% CI [−4.556,−3.373] |
| 독립 새128 prompts | −4.321%p | 95% CI [−5.164,−3.526]; 네dataset 모두 음수 |
| 또 다른96 prompts, T1 | −3.278%p | 이번 보고서 paired CI [−4.296,−2.345] |

- “초기에는 residual이 전체적으로 이겼는데 후속에서 졌다”는 결과 변화가 아님. **초기에도 전체 coverage는 열세였고, 수정한 것은 Fig.6에서 전체 정책의 조건별 승패를 과도하게 추론했던 해석.**
- Fig.6은 분포 내적, 후속 원인 분석은 local top3 coverage. 전자는 확률값 전체를 곱해 합하고 후자는 선택 집합에 들어간 실제 R 질량을 합하므로, 같은 분포에서도 부호가 달라질 수 있음.
- 가중 방식도 다름: Fig.6은 step 내 정규화된 거절 위치 가중치와 run 균등 평균; 후속 조건표는 저장 위치의 h를 모아 가중; 후속 주 coverage는 prompt/dataset 균등 평균.
- Initial global 비교의 local pool은 M64, 후속 주 비교는 실제 설정에 맞춘 M17. 다만 Fig.6 내적 자체에는 top-M 후보 선택이 없으므로 Fig.6의 부호 차이를 M 변경 때문이라고 단정하지 않음.
- 기존 실행과 후속 실행의 prompt/seed/계측도 다름. 각 요인이 관측 차이에 기여한 양을 완전히 분리한 실험은 아님. 초기34,150step에서 동일 context별 두 지표와 동일 가중치를 교차 계산한 검증은 아직 제시되지 않았음.

### 보고 가능한 결론과 남은 범위

- **반복 근거가 강한 결론:** 이 AWQ70B/TinyLlama, exit56, chain/P1-off 설정에서 기존 residual 후보의 평균 coverage가 proxy-only보다 낮음. 큰 차이는 새 prompt와 여러 task에서 반복.
- **진단으로 사용할 결과:** 후보 삭제·순위 오류의 손실 분해, 높은 overlap에서 작은 residual 신호가 불안정해질 수 있다는 기전, oracle 정확도 개선에서의 역전. 보편적 발생 빈도나 배포 가능한 gate로 확대하지 않음.
- **아직 확정할 수 없는 결론:** overlap .75 등의 보편적 경계, +.109%p 조건부 이득, e(1-q)의 작은 추가 이득, dense full model/다른 모델 쌍/P1-on에서의 우열, 실제 Mirror-SD 시스템 대비 성능.
- 신뢰 범위를 넓힐 후속 검증: full dataset와 여러 generation seed, dense 모델에서 동결된 두 점수를 같은 context에 적용; Fig.6 내적과 실제 top-k coverage를 동일 overlap 구간/동일 가중치로 함께 보고 prompt CI 제시. 이것은 별도 검증 계획이며 현재 완료 결과로 표기하지 않음.

## P1과 실제 cache hit 자료

### 고정 draft 집합 참조

- [shared_review/REPORT.md §7](../shared_review/REPORT.md#7-ds-보완-효과의-정확한-의미와-측정): T1, M64. 4 correction 위치에서 y 제외 q top3, 총 DS12. Bonus q가 없어 DS bonus root는 없음.
- P2는 DS 중복을 제외하고 추가 root를 채움. DS12+PS3: residual57.28%, proxy57.84%. DS12+PS15:69.10%/70.45%.
- 동일 DS12 이후 proxy−residual: PS3에서+.56%p CI[+.31,+.81], PS15에서+1.35%p CI[+1.07,+1.65].
- 실제 P1의 readiness, 중단 시점, dynamic K, subtree 재사용을 재현하지 않음. Root 개수 동일과 실행 비용 동일도 다름.
- 초기 문서의 P1 hit .726/P2 .037 언급은 당시 관측 기록이며, 보존된 동일2×2/P1-on 반복 결과로 간주하지 않음.

### 실제 생성 로그

- [9/8 P1-off 로그](../../../ssd/experiments/proxy_source_ablation/temp1_3arm_20260908/logs_onlyproxy/): seed42, K4, root15, 기존 위치 배분. 로그의 P2 hit residual.597 / proxy.639 / draft.556. P1 hit는0.
- 로그에 기록된 입력은 HumanEval/Alpaca/GSM/**UltraFeedback** 각32개, 총128개. 이후 C4를 쓰는 분포 측정 자료와 같은 dataset으로 쓰지 않음. 단일 seed/소수점3자리 로그이므로 CI나 정밀한 전체 비교는 제공하지 못함.
- [무간섭 반복](../training_free/LIVE_CLEAN.md): source=e 고정, 기존 배분/정규화 대 개선 배분/정규화. 16 prompts, seeds903/904, T.7, cap256, P1 off. 전체 step 가중 hit81.835419→82.389937%, +.554518%p.
- Seed별 hit 변화+2.1743/−1.1824%p. 두 seed만으로 안정적인 online 이득 확정 불가. TPS도 합산−1.615%였으므로 coverage와 TPS를 등치하지 않음.
- [이전 pilot](../training_free/LIVE.md)은 e(1-q) 조합의 hit82.130%, 기존 matched proxy80.853%, 개선 배분 proxy85.544%. 다만 GPU 분석 작업과 시간 중첩, 16 prompts×2seed, 조합 변화가 섞여 있어 token 식 단독의 확정 근거로 채택하지 않음.

## Residual 오차 상한의 증명

분포 p,q,e의 전체 vocabulary에 대해

$$
a_v=[p_v-q_v]_+,\quad b_v=[e_v-q_v]_+,\quad
Z=\sum_v a_v>0,\quad\widehat Z=\sum_v b_v>0,\quad
\delta=\operatorname{TV}(p,e).
$$

1. 같은 q를 빼고 양의 부분을 취하는 연산의 단조성/비팽창성으로
   $[a_v-b_v]_+\leq[p_v-e_v]_+$. 따라서 $\sum_v[a_v-b_v]_+\leq\delta$.
2. $Z\geq\widehat Z$이면 $b_v/\widehat Z\geq b_v/Z$이므로

$$
\begin{aligned}
\operatorname{TV}(a/Z,b/\widehat Z)
&=\sum_v[a_v/Z-b_v/\widehat Z]_+\\
&\leq Z^{-1}\sum_v[a_v-b_v]_+
\leq\delta/Z.
\end{aligned}
$$

3. $Z<\widehat Z$이면 a,b를 바꾸어 같은 논증으로 $\delta/\widehat Z$.
   TV≤1과 결합해 $\min(1,\delta/\max(Z,\widehat Z))$를 얻음.

- Z=0이면 이 prefix에서 draft draw에 대해 평균한 거절 확률0. Zhat=0이면 추정 residual 정규화가 정의되지 않아 fallback 필요.
- 이 bound는 y 제외 전 전체 분포 기준. 실제 후보 제외/Top-M 정규화 뒤 분포에 무조건 같은 bound를 쓰지 않음.
- 작은 bound는 residual 근사의 충분한 정확도 조건이지 proxy와의 순위 비교 결과가 아님.
- 같은 full-vocab top-k 문제의 oracle 집합을 S*, 실제 proxy top-k를 Se, residual top-k를 Sr, 상한을 B라 하면
  $R(S^*)-R(S_r)\leq2B$. 따라서 $R(S^*)-R(S_e)>2B$면 residual coverage 우위의 충분조건.
  현재 e,q만으로 이 oracle gap을 정확히 알 수 있다는 뜻은 아님.
- e=p이면 residual top-k는 실제 R의 top-k이므로 최적. 엄격한 개선은 proxy top-k와 다른 집합을 고르고 그 R 질량이 더 클 때 발생.

## 단순 token 수정식의 근거

$$
[e-q]_+\leq e(1-q)\leq e,\qquad
e(1-q)=[e-q]_++\min(e,q)-eq.
$$

- q∈[0,1]에서 성립. e≥q일 때 왼쪽 차이는 q(1−e)≥0, e<q이면 왼쪽 residual이0이라 성립.
- q 고정, e>0/q<1인 좌표에서 $\partial\log(e(1-q))/\partial\log e=1$.
  기존 양의 residual은 $e/(e-q)$로 상대 민감도가 커질 수 있음.
- 이 성질은 **비정규화 점수의 민감도**에 관한 것. 정규화 후 TV, top-k coverage 또는 항상 더 좋은 cache hit의 증명은 아님.
- e=p에서도 일반적으로 true residual과 다름. 독립 V~e,Y~q의 $P(V=v,Y\ne v)$ 해석은 가능하지만 실제 SD rejection 사건과 동일하지 않음.
- 현재 본문은 T=.7에서 사전에 동결했던 β=1 결과만 주표로 제시. 과거 T1의 선택식은 β=.5, T.5는 floor였으므로 모든 T의 녹색 막대를 e(1-q)라고 부르면 안 됨.
- 이번 CPU 재집계는 β=1 고정 결과도 모든 T에 저장. 같은 기존 배분에서 proxy 대비 T1+.005%p, T.7−.001%p, T.5−.146%p이며 모두 CI에0 포함.
- T1의 개선 배분 아래 β=1은 저장 replay에서+.199%p였지만, 이 arm은 기존 wire 검사 대상이 아님. 사후 추가 비교이고 동점 민감도 재검증 전이므로 우위 결론에서 제외.

### 기존 위치 수식 고정·세 점수 비교

- 2026-10-02: 기존 confirmation CSV와10/1 재집계 JSON의9개 평균이1e−12 이내 일치함을 확인하고 주 보고서에 명시적3자 표를 추가.
- 정확한 키는 residual / proxy / complement_power1 각각의 topm + original 조합. 모든 source에서 같은 $\widehat h=H(\min(1,e(y)/q(y)))$ 사용.
- 공통 정규화 규칙은 위치별 해당 source의 top17 점수 합으로 나누는 것. 규칙은 동일하지만 source가 바뀌면 분모 값/전역 선택도 바뀔 수 있음.
- $e(1-q)$−proxy의 paired95% CI, 단위 %p: T1 +.005 [−.217,+.197], T.7 −.001 [−.098,+.103], T.5 −.146 [−.485,+.145]. 우위 또는 엄밀한 동등성을 증명한 결과가 아님.
- 추정 위치 확률과 정규화 변경을 함께 적용한 조합의 이득을 token 식 단독의 이득으로 설명하지 않음.
- 기존 위치식은 세 방법에 동일한 식을 쓴다는 의미. T별 sampling 확률을 사용한 저장 replay이며, 원래 BF16 runtime/wire의 bit-level 재현은 아님.
- [새 비교 그림](06_token_original_position.png) · [PDF](06_token_original_position.pdf). 재생성 함수는 make_report_assets.py의 plot_original_position.

## 위치 확률 혼합의 근거와 단독 대체의 한계

관측 draft token y에서 $\widehat\alpha_i=\min(1,e_i(y_i)/q_i(y_i))$.
고정 prefix의 Y~q에 대한 기대값은

$$
\bar\alpha_i
=\sum_vq_i(v)\min(1,e_i(v)/q_i(v))
=\sum_v\min(e_i(v),q_i(v)).
$$

- q(v)=0인 항은 마지막 합으로 정의하면0. $\bar\alpha=1-TV(e,q)$.
- 실제 y에 조건부인 추정과, y를 평균한 추정은 다름. 뒤 prefix도 앞 token에 의존하므로 평균들의 곱을 전체 경로의 정확한 첫 거절 확률로 해석하지 않음.
- 완벽한 e=p=(.9,.1), q=(.5,.5), 실제 y=A일 때 $\widehat\alpha=1$, $\bar\alpha=.6$.
  첫 위치의 정확한 거절 확률은0인데 overlap 단독은.4를 부여. **Proxy가 정확해도 단독 평균화가 현재 token 정보를 잃을 수 있음.**
- 반대로 e 오차 때문에 실제 거절 가능 위치의 hhat가0이 되면, hbar가 양수인 경우 혼합으로 그 위치를 복구할 수 있음.
- $\widetilde h=(1-\omega)\widehat h+\omega\bar h$는 두 정규화된 위치분포의 convex mixture이므로 유효한 가중치 분포. Bonus도 포함해 합이1.
- 위 혼합은 $\alpha$ 혼합 후 누적곱과 다름. $\omega=.25$는 development에서 선택; 최적값의 보편적 증명 없음. 실제 target verification α는 그대로.

단위 %p, pointwise95% CI. Source=e/top-M 정규화 고정.

| T | overlap 단독−기존 | 25% 혼합−기존 | 25% 혼합−overlap 단독 |
|---|---:|---:|---:|
| 1.0 | −1.366 [−2.197,−.536] | +1.612 [1.075,2.185] | +2.978 [2.452,3.518] |
| .7 | +.913 [−.049,1.899] | +2.345 [1.624,3.107] | +1.432 [.801,2.058] |
| .5 | +1.909 [.721,3.176] | +2.334 [1.394,3.340] | +.425 [−.260,1.068] |

- 단독 사용을 아직 시험하지 않았던 것이 아님. `expected_rejection`으로 기존2,640조합에 포함돼 있었으며 이번에 표/그림으로 별도 정리.
- 단독 규칙의 단순성은 장점이지만 모든 온도에서 기존보다 좋다는 결과는 없음. T=.5에서 혼합이 필수라고 주장할 근거도 부족.
- Online hook의 현재 지원 weight는 `original`, `expected_mix...`, `uniform...`임.
  단독 hbar를 live 실행하려면 동일 의미의 `expected_mix1`로 설정할 수 있으나 해당 live 반복 결과는 아직 없음.

## 사용할 그림과 피해야 할 해석

| 그림 | 보여주는 내용 | 범위 |
|---|---|---|
| [01_factorial](01_factorial.png) / [PDF](01_factorial.pdf) | 요청한 source×배분의2×2 | 새 CPU 집계, P1 off, coverage |
| [02_position](02_position.png) / [PDF](02_position.pdf) | 기존/overlap 단독/25% 혼합 | Source/정규화 동일, 위치만 변경 |
| [03_token_wire](03_token_wire.png) / [PDF](03_token_wire.pdf) | T.7의 배분 효과와 e(1-q)의 작은 추가 효과 | 실제17→15 방식의 재구성 logits 검사 |
| [direct comparison](../direct_comparison/comparison.png) | 원래 residual 열세, 삭제/순위 손실, oracle 보정 | 다른128 prompts; oracle를 구현 성과로 쓰지 않음 |
| [초기 layer coverage](../figs/01_coverage.png) | 뒤 layer에서 residual이 상대적으로 좋아지는 경향 | 초기 topM64/run 가중; 현재96-prompt 표와 절대값 혼합 금지 |
| [기존 runtime confirmation](../training_free/runtime_confirmation.png) | 전체 조합 대 동일 배분 proxy | T별 source식이 다르며 e(1-q) 고정 비교가 아님 |

초기 `06_where_we_lose.png`는 분포 내적 진단이다. 이를 top-k hit나 전체 손실의 원인별 비중으로 다시 명명하지 않음.

## 남은 실험

1. 동일 root15/forward 예산의 actual cache hit 2×2, P1 off/on 각각. 계측은 가볍게 하고 준비 완료/중복 제거/P1·P2 hit를 별도 기록.
2. Token score=e와 e(1-q), 위치 weight=hhat/hbar/mix를 교차한 controlled live 반복. 정규화와 후보 temperature를 고정해 혼동 방지.
3. Same-context actual-wire 재검증: 새로운 fixed3 arm, hbar 단독, T1/.5의 β=1. 현재 저장 replay와 GPU 동점 순서가 같다고 가정하지 않음.
4. 독립 full dataset와 여러 seed, dense target, 추가 exit layers/model pairs. Test 결과를 보고 계수를 다시 골랐다면 새 독립 검증 필요.
5. 실제 Mirror-SD system 비교: 별도 구현/실행 배치/학습된 draft까지 맞추어 검증. 본 보고서의 proxy+고정 배분은 그 시스템 benchmark가 아님.

## 재생성과 원본 보존

```bash
OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/root_progress_20261001/make_report_assets.py
```

- 입력 NPZ는 9/27 raw archive 또는 현재 서버에서 복원 필요. Git만 clone하면 이 입력들은 없음.
- 코드는 새 디렉터리만 출력하며 이전 frozen 보고서/실험 파일을 수정하지 않음.
- Source hash와 집계 조건은 `summary.json`. 이번 figure의 CI는 통계 표본 오차이며 precision/wire/model 변경 불확실성을 포함하지 않음.
