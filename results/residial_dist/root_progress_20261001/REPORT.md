# DUET Root 후보 선정 연구 진행 보고

- **목표:** correction token을 미리 준비해 cache hit 향상. 토큰 점수와 위치별 예산 배분의 효과를 분리 평가.
- **핵심 결과:** 기존 위치 배분의 이득 확인. Residual은 proxy 오차로 후보를 잃는 문제가 있으며, 현재 가장 일관된 개선은 위치 확률 보정.
- **주요 조건:** LayerSkip70B AWQ / TinyLlama1.1B, exit56, chain K=4, root15, P1 off. 4 datasets의 별도 검증 96 prompts를 T=1/.7/.5에서 평가.
- **지표 구분:** 아래 표는 저장된 동일 경로에서 계산한 **후보 coverage**. 실제 cache hit는 마지막 항목에 별도 기재. 세 온도의 prompt는 동일하며 총 288개 독립 prompt가 아님.

## 1 DUET과 Mirror-SD의 후보 선정 비교

- Mirror-SD는 early-exit proxy의 top-κ 후보를 이용해 분기를 준비. 원문의 해당 설계에는 DUET의 첫 거절 위치 확률에 따른 전역 예산 재배분식이 명시되어 있지 않음. [원문 §3.1](https://arxiv.org/html/2510.13161v2#S3.SS1)
- 아래는 **동일 DUET 엔진의 후보 정책 분해 비교**. Proxy+고정 배분을 비교 기준으로 사용하며, Mirror-SD 전체 시스템 재현 결과와 구분.
- 2×2 구성: 토큰 점수 `{[e−q]₊, e}` × 배분 `{위치마다 3개, 기존 위치 확률에 따른 전역 15개}`. Bonus를 포함한 5개 위치, 총 root 수 동일.

| T | Residual 고정 배분 | Proxy 고정 배분 | Residual 기존 위치 배분 | Proxy 기존 위치 배분 |
|---|---:|---:|---:|---:|
| 1.0 | 56.85% | 59.41% | 61.76% | **65.04%** |
| 0.7 | 75.14% | 79.05% | 78.28% | **82.04%** |
| 0.5 | 82.27% | 85.83% | 85.10% | **88.93%** |

- **위치 배분 효과:** 두 토큰 점수 모두 고정 배분보다 개선. T=1에서 residual +4.91%p, proxy +5.63%p.
- **조합 간 비교:** residual+기존 배분은 proxy+고정 배분 대비 T=1에서 +2.36%p, 95% CI [+1.11,+3.60]. T=.7/.5는 각각 −0.77/−0.73%p이며 두 CI 모두 0 포함.
- **토큰 점수만 비교:** 같은 배분에서는 세 온도 모두 기존 residual이 proxy보다 낮음. 위치 배분의 이득을 residual 수식 자체의 우위로 해석하지 않음.
- 그림: [2×2 비교 PNG](01_factorial.png) · [PDF](01_factorial.pdf). 저장된 top15 replay 재집계이며 실제 wire 동점 재현과 구분.

**언제 이기고 지는가 — target와 draft의 차이를 기준으로**

- **볼 실험:** direct_comparison의 T1/exit56 확인 실행. Alpaca/C4/GSM8K/HumanEval 각각 기존1–32행 다음인 **33–64행 입력32개**, 총128개. “새 프롬프트”는 같은 dataset의 다른 입력을 뜻함.
- x축은 $d=\mathrm{TV}(p_T,p_D)$, y축은 같은 위치에서 후보3개를 골랐을 때의 correction coverage. 거절 확률 h로 가중하며 bonus 제외. 앞의96-prompt 전역 root15 표와 구분.

| Target–draft 차이 d | Residual coverage | Proxy coverage | Residual−proxy |
|---|---:|---:|---:|
| [0,.25): 가까움 | 52.57% | 66.19% | **−13.62%p** |
| [.25,.50) | 48.13% | 51.94% | −3.81%p |
| [.50,.75) | 50.78% | 54.66% | −3.88%p |
| [.75,1]: 멂 | 65.64% | 67.59% | −1.94%p |

- **관측:** draft가 target에 가까운 구간에서 coverage 손실이 가장 큼. 나머지 구간도 평균 열세. [원하는 축의 coverage 그림](04_tv_coverage.png) · [PDF](04_tv_coverage.pdf).
- **이전 Fig.6과의 관계:** 같은128개 자료에서 alignment는 이전처럼 높은 overlap 구간만 음수. 그러나 top3 coverage는 네 구간 모두 음수. **같은 context·가중치에서도 지표에 따라 부호가 달라짐**을 확인. [두 지표 대조 그림](05_overlap_coverage_alignment.png).
- **원인 분해:** oracle 개선 여지 **+13.016%p − 후보 삭제10.665%p − 순위 손실7.360%p = −5.009%p**. 작은 residual 신호에 비해 proxy 오차가 클 수 있고, 큰 mass에서도 삭제·순위 오류가 남음.
- **개선 가능한 조건:** target을 이용해 실제 proxy 오차의 방향을 유지하며 오차를50% 줄인 oracle 실험에서는 전역 residual−proxy가 −4.32→+1.00%p로 전환. 배포 가능한 보정 성과나 보편적 threshold는 아님.
- **개별 승패:** 별도 전역 root15 비교는 승40.50%/패42.21%; 승리 step 평균+2.54%p, 패배 step 평균−12.68%p. 구간 평균이 음수라고 모든 step에서 지는 것은 아님.
- 입력 출처·CI·alignment 정의·가중 방식은 [차이별 분석](DISTANCE.md), proxy 오차까지 교차한 조건표와 후보 교환 손실은 [부록](EVIDENCE.md#조건별-승패와-해석)에 정리.

**P1이 존재할 때**

- 실제 P1 실행을 대신한 **고정 draft 후보 집합** 분석만 있음: 4거절 위치마다 y를 제외한 q top3, 총 DS12개. 이후 P2는 DS와 중복되지 않는 후보를 추가. 실제 P1의 준비 완료/중단 시점은 재현하지 않음.
- 자료: 기존1–32행 입력128개×2seed, T1/exit56, M64. 아래는 전역 coverage이며 앞의33–64행 자료/위치별 top3 그래프와 구분.

| 구성 | 총 고유 root | Residual P2 | Proxy P2 | Residual−proxy |
|---|---:|---:|---:|---:|
| Draft 집합 없이 P2만15개 | 15 | 60.18% | 64.14% | −3.96%p |
| DS12 + P2 추가3개 | 15 | 57.28% | 57.84% | **−0.56%p** |
| DS12 + P2 추가15개 | 27 | 69.10% | 70.45% | **−1.35%p** |

- **보완 효과:** residual이 누락한 실제 correction 질량의39.30%가 DS12에 포함. 그러나 동일 DS 이후 추가 후보는 proxy가 더 좋았음. Proxy−residual CI는 추가3개 [+.31,+.81]%p, 추가15개 [+1.07,+1.65]%p.
- 비교 대상은 $U(D\cup C)=U(D)+U(C\setminus D)$, 즉 **이미 준비된 D를 제외한 P2 추가 coverage**. 격차 감소에는 P2 예산 변화도 포함되므로 DS 보완만의 인과 효과로 단정하지 않음.
- 동일 총15개에서 DS12 고정 배정은 P2만15개보다 coverage가 낮음. 이 고정 배정의 결과이며, 실제 P1의 시간 활용 이득을 부정하는 결과는 아님.
- **미검증:** 실제 P1 on의 source 통제 비교, P1 on에서 TV(p_T,p_D)별 coverage, 개선식 e(1-q)+위치 보정의 P1 on 평가. [기존 분석 §7](../shared_review/REPORT.md#7-ds-보완-효과의-정확한-의미와-측정).

## 2 토큰 예측 수식 개선

- 표기: target $p$, draft $q$, proxy $e$, $a=[p-q]_+$, $b=[e-q]_+$, $Z=\sum a$, $\widehat Z=\sum b$, $\delta=\mathrm{TV}(p,e)$.
- $Z,\widehat Z>0$일 때 residual 근사 오차의 상한:

$$
\mathrm{TV}\!\left(\frac aZ,\frac b{\widehat Z}\right)
\leq \min\!\left(1,\frac{\delta}{\max(Z,\widehat Z)}\right).
$$

- 해석: **proxy 오차가 residual mass보다 충분히 작을 때 근사에 유리**. 작은 mass로 정규화하면 오차에 민감. 이 bound만으로 proxy-only 대비 후보 우위는 보장되지 않음. [짧은 증명과 조건](EVIDENCE.md#residual-오차-상한의-증명)
- 단순 개선안: $\boxed{s(v)=e(v)(1-q(v))}$. $[e-q]_+\leq e(1-q)\leq e$: 강한 차감으로 후보를 0으로 만드는 현상을 완화하고 draft 확률이 큰 token을 할인.
- 근거와 한계: 고정 q에서 e의 상대오차가 점수에 그대로 전달됨. **True residual의 정확한 대체식은 아니며, cache 후보 순위를 위한 완화식**. 실제 correction sampler/verification은 변경하지 않음.

**기존 위치 수식에서 토큰 점수만 비교**

- 세 정책 모두 기존 $\widehat h_i$와 top-M 정규화 사용. 바뀌는 것은 token 점수뿐이며, bonus 점수는 모두 e.
- 자료: 4datasets의65–88행 입력, 총96개 prompt를 각 T에서 실행. Exit56, K4, M17, root15, P1 off. 저장 top15 replay의 전역 coverage.

| T | 기존 residual $[e-q]_+$ | Proxy e | $e(1-q)$ |
|---|---:|---:|---:|
| 1.0 | 61.762% | 65.039% | 65.045% |
| 0.7 | 78.275% | 82.042% | 82.041% |
| 0.5 | 85.100% | 88.931% | 88.785% |

- **해석:** $e(1-q)$는 원래 residual보다 +3.28/+3.77/+3.68%p 높지만, proxy 대비 차이는 +.005/−.001/−.146%p. 세 차이 모두95% CI에0 포함. Proxy보다 낫다는 근거는 부족.
- 그림: [원래 위치식 고정·세 점수 비교](06_token_original_position.png) · [PDF](06_token_original_position.pdf).
- 실제 P1-on/cache-hit 결과가 아님. 세 T에서 모두 β=1이며, 개발 자료가 T별로 선택한 서로 다른 개선식을 혼합하지 않음.

**배분/정규화까지 바꾼 조합의 별도 비교**

아래는 실제 top17→15 wire 방식을 재현한 결과이므로 위 top15 replay 표와 수치를 혼합하지 않음.

| T=.7 실제 wire 방식의 coverage | 결과 |
|---|---:|
| Proxy + 기존 배분/정규화 | 82.020% |
| Proxy + 개선 배분/정규화 | 84.363% |
| $e(1-q)$ + 같은 개선 배분/정규화 | 84.431% |

- 전체 조합은 기존 proxy 대비 +2.411%p. **Token 식만의 추가 효과는 +0.068%p, CI [−0.110,+0.274]로 미확정.**
- 그림: [Token 식과 배분 효과 분리 PNG](03_token_wire.png) · [PDF](03_token_wire.pdf). 재구성 float32 logits의 top17→15 replay이며 실제 cache hit가 아님.

## 3 위치 예측 수식 개선

- 기존 수락률 추정: $\widehat\alpha_i=\min(1,e_i(y_i)/q_i(y_i))$, 첫 거절 위치 추정: $\widehat h_i=(\prod_{j<i}\widehat\alpha_j)(1-\widehat\alpha_i)$.
- 문제: 실제 거절 가능한 token도 proxy가 $e_i(y_i)\geq q_i(y_i)$로 추정하면 위치 가중치 0. 관측 token 하나의 오차가 뒤 위치의 누적곱에도 영향.
- 추가 추정: $\bar\alpha_i=\sum_v\min(e_i(v),q_i(v))$. **고정 prefix에서 draft token을 뽑기 전의 proxy 기반 평균 수락률**이라는 정확한 근거가 있음.
- 두 수락률로 위치 가중치 $\widehat h,\bar h$를 각각 구성. 개발 자료에서 선택한 개선은 $\boxed{\widetilde h=0.75\widehat h+0.25\bar h}$.
- 혼합 이유: 관측 token의 조건부 정보 유지 + proxy 오차로 0이 된 위치 보완. **25%는 이론적 최적값이 아닌 개발 자료로 선택한 계수**.

| T | 기존 위치 확률 | $\bar\alpha$만 사용 | 25% 혼합 |
|---|---:|---:|---:|
| 1.0 | 65.04% | 63.67% | **66.65%** |
| 0.7 | 82.04% | 82.96% | **84.39%** |
| 0.5 | 88.93% | 90.84% | **91.26%** |

- 토큰 점수 e, 정규화 방식 고정. **위치 가중치만 변경**한 저장 replay.
- **$\bar\alpha$ 단독도 가능하나 일괄 대체 근거 부족:** T=1에서는 기존 대비 −1.37%p, CI [−2.20,−0.54]. 실제 나온 token 정보를 평균화하는 손실 존재.
- 혼합은 기존 대비 +1.61/+2.34/+2.33%p. 단독 대비 혼합의 추가 우위는 T=.5에서 CI에 0 포함.
- 그림: [위치 추정 비교 PNG](02_position.png) · [PDF](02_position.pdf).
- 정규화 변경은 별도 요인: top-M 안의 합 대신 전체 허용 token의 점수 합 사용. 상위 후보 밖 질량을 무시해 특정 위치 점수가 부풀려지는 문제를 줄임. 위 위치 단독 표에는 적용하지 않음.

## 4 실제 cache hit와 남은 검증

- 기존 source의 P1-off 실제 생성(단일 seed): residual 59.7%, proxy 63.9%. 고정 분배 arm은 없어 **실제 cache hit 2×2는 미완성**.
- Proxy 점수 유지·배분/정규화만 개선한 별도 무계측 실행(16 prompts×2seed): cache hit **81.835→82.390%**, +0.555%p. Seed별 +2.17/−1.18%p로 방향이 달라 확대 검증 필요.
- $e(1-q)$ 자체의 추가 cache hit 이득: 자원 간섭이 있던 예비 실험 외에 독립적인 무간섭 반복 검증 부족.
- 다음 검증: **2점수×2배분×P1 on/off**, 고정 총 예산·동일 모델/온도·여러 seed의 실제 cache hit. P1/P2 hit, 중복 제거 후 P2 추가 coverage, 준비 완료 여부를 함께 기록.
- 실제 Mirror-SD 전체 시스템, dense full model 및 full dataset에서의 위 root 비교는 별도 수행 필요.

실험 범위·CI·기존 그림 사용 시 주의점·증명은 [근거와 부록](EVIDENCE.md)에 정리. 초기/후속 표본 규모와 믿을 수 있는 결론의 범위는 [데이터 규모와 신뢰 범위](EVIDENCE.md#데이터-규모와-신뢰-범위) 참조.
