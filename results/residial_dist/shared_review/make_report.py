"""Render the final Korean review from completed replay statistics."""
import json
from pathlib import Path


def main():
    out=Path(__file__).parent
    d=json.loads((out/'results.json').read_text())
    if d['totals']['runs']!=8:raise ValueError('final report requires all 8 runs')
    p=d['policies'];s=d['secondary'];sel=d['selections'];n=d['totals']
    f=lambda x:f'{x:.4f}'
    pct=lambda x:f'{100*x:.2f}%'
    ci=lambda a:f"{a['mean']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}]"
    st=lambda k:s['step::'+k]['mean']
    pool=s['pooled'];feasible=s['thermal_feasible_rows']
    notes=[]
    notes.append(f'''# 공유 분석 검토 및 실제 모델 검증

검토 대상: [수식 기반 후보선정 분석](https://chatgpt.com/s/t_6aa29a78c4488191a8443e4d147cc79e).
작성일: 2026-09-10. 공유 페이지에 포함된 답변 본문을 읽고 수식, 현재 구현,
기존 계측, 새 모델 실행과 후보 replay를 대조했다. 본문에 연결된 별도
`sandbox:` HTML/ZIP/그림 원본은 확보하지 못했다. 따라서 원래 3,000-context
합성 heatmap의 정확한 재현을 주장하지 않으며, 명시된 수치 예제와 핵심
가설을 독립적인 합성 검사 및 실제 모델 분포로 검증했다.

**판정:** residual의 oracle 의미와 clipping/support 오류 설명은 타당하다.
그러나 평균 entropy 차이는 실제 오류 방향이나 최적 temperature를 결정하지 않는다.
실제 분포에서는 위치별 순위·확률 오차를 함께 보아야 하며, target KL 최소화와
cache 후보 coverage 최대화도 서로 다른 목적이다.

## 1. 새 실험의 범위와 평가 기준

- Target: AWQ 보정 norm을 적용한 LayerSkip Llama2-70B, TP=4.
  Draft: TinyLlama-1.1B-Chat. RTX 4090에서 실행했다.
- alpaca/c4/gsm/humaneval × seed 42/123, 각 32 prompt × 최대 256 output token.
  B=1, sampling T=1, only-proxy chain, exit=56. Config K1=8/K2=4지만 실제 verify K는 모두 4.
- 총 **{n['generations']} generation, {n['unique_prompts']}개의 서로 다른 prompt,
  {n['verification_steps']:,} verification step**. Prompt마다 step 0,8,16,…에서
  **{n['sampled_steps']:,}개 step / {n['contexts']:,}개 context**를 저장했다.
  이 중 거절 후보 위치는 {n['reject_positions']:,}개이며 나머지는 bonus 위치다.
- pT, pD, pE56, pE79, 실제 draft token을 **전체 32,000 vocabulary / float32**로
  보존했다. 저장 시 top-k/저정밀도 절단을 하지 않았다.
- 주 평가: 실제 엔진과 같은 sampled-token 제외, 위치별 top-M=17 재정규화,
  전 위치에서 총 15개 후보 선택. M=64와 전체 vocabulary 정규화도 대조했다.
- Calibration은 각 dataset의 prompt id 1–16에서만 선택했다. **17–32는 held-out**이며
  두 seed에서 같은 prompt 분할을 사용했다. 전역 tau 7개 × lambda 5개를 비교했다.
- Coverage 표는 generation 안에서 step 평균 → 같은 prompt의 seed 평균 → dataset별
  prompt 평균 → 4개 dataset 평균이다. CI는 dataset 안에서 prompt cluster를
  4,000회 paired bootstrap했다. 토큰 위치를 독립 표본으로 취급하지 않았다.
  튜닝한 정책을 고정한 뒤 평가 prompt의 변동을 나타내며, train 집합을 다시 뽑는
  튜닝 불확실성 전체를 포함하는 CI는 아니다.
- Support/entropy 조건부 표는 별도로 명시한 **실제 reject-event 가중**이다.
  같은 수식이라도 이 가중 평균과 prompt 평균의 분모는 다르다.

모든 후보 정책은 같은 저장 분포를 평가한다. 정책이 바뀐 뒤 실제 생성 경로,
P1 시간 배분, subtree 길이, 최종 TPS가 어떻게 바뀌는지는 이번 replay의 측정 대상이 아니다.

## 2. 원문 핵심 주장과 추가 가설의 판정

원문도 평탄화 모형의 조건, 합성 실험의 한계, entropy만으로 원인을 확정할 수 없다는
점을 명시한다. 원문이 부정하거나 유보한 명제는 아래에 따로 표시했다.

| 주장 | 검토 결과 |
|---|---|
| 정확한 residual의 top-k는 correction hit에 최적 | 고정 reject 위치, 동일 후보 비용에서 맞음. 전역 선택에는 정확한 h와 정상적인 전체 분포 점수가 필요 |
| q를 빼면 절대 오차가 커짐 — 원문도 부정 | 성립하지 않음. 차의 절대 오차는 그대로이며, clipping도 L1 오차를 늘리지 않음 |
| 작은 residual에 비해 proxy 오차가 커 support/순위가 불안정 | 맞음. 실제 누락 질량을 추가 측정 |
| 원문의 residual TV 상계 | 맞음. 아래에 더 강한 상계와 증명 제시 |
| 순위가 완벽한 평탄화도 residual을 망칠 수 있음 | 명시된 4-token 예제를 재현. 충분한 반례이나 실제 주원인이라는 증명은 아님 |
| proxy가 항상 평평해짐 — 원문의 주장이 아닌 추가 확인 | 성립하지 않음. 기존 12런의 위치별 entropy 결과에서 양방향 오류 확인 |
| tau<1 보정 — 원문은 평탄화가 주된 오차일 때로 조건부 제안 | 평균 entropy만으로 도출되지 않음. KL의 미분식과 held-out 보정 실험으로 실제 조건 확인 |
| h 오차를 별도로 분리해야 함 | 맞음. oracle h 및 고정 위치별 후보 수로 비교 |
| DS가 residual의 누락을 보완할 수 있음 | 가능. 동일 DS의 누락 질량 복구와 추가 coverage를 계산. 실제 비동기 P1 성능은 별개 |
| q와 e의 convex mixture로 같은 위치 residual 개선 — 원문도 부정 | 정확한 산술에서는 개선되지 않음. 정규화 residual이 동일함을 재현 |
| lambda 중간값은 정확한 sampling residual — 원문도 구분 | 후보 순위를 위한 정책으로만 사용 가능 |
| 높은 후보 coverage가 높은 TPS를 보장 — 원문도 구분 | 준비 비용과 hit 이후 수락 길이 검증이 필요 |

## 3. 수식 검토와 보완
''')
    notes.append(r'''
같은 prefix에서 $p=P_T$, $q=P_D$, $e=P_E$로 두고

\[
a=(p-q)_+,\quad b=(e-q)_+,\quad Z=\sum_v a_v=\operatorname{TV}(p,q),\quad
\widehat Z=\sum_v b_v=\operatorname{TV}(e,q).
\]

$Z>0$에서 실제 correction 분포는 $R=a/Z$이다. Bonus는 해당 위치의 $p$이다.
이는 [표준 speculative decoding 알고리즘](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)의 확률적 검증에 해당한다.
고정된 reject 위치에서 동일 비용 후보 $k$개는 $R$의 top-k가 최적이다.
다만 실제 코드가 top-M을 **다시 정규화**하면 그 점수는 전체 $R$과 다르므로,
정확한 p를 넣었다고 전역 top-R 최적성이 자동으로 따라오지는 않는다.

**오차 증폭의 정확한 뜻.** $e=p+\varepsilon$이면 $(e-q)-(p-q)=\varepsilon$이다.
양수 부분 연산은 좌표별 1-Lipschitz이므로
$\|(e-q)_+-(p-q)_+\|_1\leq\|e-p\|_1$이다.
증폭은 절대 차의 뺄셈 자체가 아니라, 작은 residual 신호와 비교하거나 정규화한 뒤의 문제다.

$p_v>q_v$인 토큰이 사라지는 조건은 $\varepsilon_v\leq-(p_v-q_v)$이다.
원문의 순위 역전 조건 $\varepsilon_w-\varepsilon_v>d_v-d_w$는 **clipping 전** 순위에 정확하다.
Clipping 후에도 엄격한 역전이라고 하려면 $e_w-q_w>0$이 추가로 필요하다.
둘 다 음수이면 순위 역전 대신 0점 동률이 된다.

**더 강한 residual 안정성 상계.** $Z,\widehat Z>0$이면

\[
\boxed{\operatorname{TV}(R,\widehat R)
\leq \min\left(1,\frac{\operatorname{TV}(p,e)}{\max(Z,\widehat Z)}\right).}
\]

증명: $D_+=\sum(a-b)_+$와 $D_-=\sum(b-a)_+$는 각각
$\sum(p-e)_+$, $\sum(e-p)_+$ 이하이므로 둘 다 $\operatorname{TV}(p,e)$ 이하이다.
$Z\geq\widehat Z$이면

\[
\sum_v\min\left(\frac{a_v}{Z},\frac{b_v}{\widehat Z}\right)
\geq\frac{\sum_v\min(a_v,b_v)}{Z}
=1-\frac{D_+}{Z}.
\]

반대 경우는 $D_-/\widehat Z$를 쓰면 된다. 원문의
$\|p-e\|_1/Z=2\operatorname{TV}(p,e)/Z$ 상계보다 강하다.
하지만 이 역시 **상계**이며, 작은 Z가 큰 실제 오류를 필연적으로 만든다는 뜻은 아니다.
예컨대 $e=(1-c)q+cp$, $c>0$이면 $\widehat R=R$이다.

**평탄화 반례는 정확하지만, 특정 오차 모형에 대한 반례다.**
균등분포 u를 사용해 $e=ap+(1-a)u$라 하면 p/e의 토큰 순위는 같지만

\[
e-q=a(p-q)+(1-a)(u-q).
\]

q가 큰 토큰에는 두 번째 항이 음의 편향을 준다. 실제로 필요한 correction 토큰인데
target과 draft가 모두 높은 확률을 주면, 작은 residual이 이 편향에 묻힐 수 있다.

| 분포 | A | B | C | D |
|---|---:|---:|---:|---:|
| p | .50 | .25 | .15 | .10 |
| q | .45 | .26 | .17 | .12 |
| e=.6p+.4u | .40 | .25 | .19 | .16 |
| (p−q)+ | .05 | 0 | 0 | 0 |
| (e−q)+ | 0 | 0 | .02 | .04 |

거절 뒤 실제 correction은 A인데 e-only top-1은 A, proxy residual top-1은 D다.
이 예제에서 hit 1 대 0이 정확히 재현됐다. 여기서 알 수 있는 것은 “순위 보존도
residual을 보장하지 않는다”는 것이지 “실제 proxy가 이 균등 혼합 모형이다”가 아니다.

이 모형을 정확히 역보정하려면 $e-aq-(1-a)u=a(p-q)$이다.
같은 위치의 raw score 순위만 볼 때는 균등한 baseline을 생략할 수 있지만,
support, 정규화, 위치 간 결합확률에는 생략할 수 없다.
또한 $\widetilde p=(1-\lambda)q+\lambda e$로 단순 혼합하면
$\widetilde p-q=\lambda(e-q)$이므로, $\lambda>0$이고 residual 분모가 양수일 때
정규화 residual의 순위·support·분포가 모두 그대로다. h만 달라질 수 있다.

**경로를 조건으로 한 h와 marginal reject 확률의 구분.**
현재 관측한 draft 경로 $y_{1:K}$를 조건으로

\[
\alpha_i=\min(1,p_i(y_i)/q_i(y_i)),\qquad
h_i=\prod_{j<i}\alpha_j(1-\alpha_i),\quad
h_{K+1}=\prod_{j=1}^K\alpha_j.
\]

따라서 $\pi(i,v)=h_iR_i(v)$, bonus는 $h_{K+1}p_{K+1}(v)$이다.
고정 prefix에서 $y\sim q$까지 평균하면 한 위치의 reject 확률이 Z가 되지만,
**이미 관측한 경로의 h를 Z로 대체해서는 안 된다.**
누락 질량도 실제 step 목적에서는 $h_i$로 가중해야 한다.

**현재 코드와의 일치.** `bench.py`의 DUET 경로는 `jit_speculate=True`를 강제한다.
따라서 cache miss에서도 유효한 q에 대한 ratio acceptance를 사용한다.
이번 실험의 tree 정책은 모두 off이고 sampler_x도 꺼져 있다.
`ssd/utils/verify.py`에서 실제 reject는 $(p-q)_+$, all-accept는 p로 샘플링한다.
Tree sibling residual ladder나 변형 proposal에 대한 원문의 주의는 타당하지만
이번 chain 실험에는 해당하지 않는다.

Mirror-SD의 top-kappa 정책은 [논문 §3.1](https://arxiv.org/html/2510.13161v2#S3.SS1)의 아이디어와 연결되지만,
여기서 `proxy`는 **DUET의 동일한 h 및 후보 예산을 사용하는 pE 점수 ablation**이다.
전체 Mirror-SD 시스템 성능 비교로 부르면 안 된다.
''')
    notes.append('''
## 4. 실제 모델의 oracle 및 후보 정책 비교

아래는 held-out prompt 결과다. 값은 0–1 범위의 recovery-event coverage이며,
실측 cache hit/TPS로 해석하지 않는다. h와 후보 분포를 바꾸는 oracle은 실행 가능한 정책이 아니다.

| 토큰 분포 / 위치 정책 | Held-out coverage | 현재 residual 대비 차이 [95% CI] |
|---|---:|---:|
''')
    names=[('residual','Proxy residual / proxy h'),('proxy','Proxy pE / proxy h'),
           ('residual_true_h','Proxy residual / true h'),('proxy_true_h','Proxy pE / true h'),
           ('oracle_residual_proxy_h','True residual+true bonus / proxy h'),
           ('oracle_residual_true_h','True residual+true bonus / true h, M=17'),
           ('oracle_target_true_h','True pT / true h, M=17'),('ceiling','Exact joint top-15 oracle')]
    for key,label in names:
        a=p[key];notes.append(f"| {label} | {f(a['test']['mean'])} | {ci(a['test_gain_vs_residual'])} |\n")
    notes.append(f'''
동일한 위치에 3개씩 고르는 통제(총 15개)의 전체 prompt 평균은
residual **{f(st('residual__equal3'))}**, pE **{f(st('proxy__equal3'))}**다.
이 비교는 h 추정과 위치 간 예산 배분을 제거한 뒤의 토큰 선택 차이다.

Top-M 재정규화 민감도, 전체 prompt 평균:

| 정책 | M=17 | M=64 | 전체 vocabulary 정규화 |
|---|---:|---:|---:|
| residual, proxy h | {f(st('residual__proxy_h__m17'))} | {f(st('residual__proxy_h__m64'))} | {f(st('residual__proxy_h__full'))} |
| pE, proxy h | {f(st('proxy__proxy_h__m17'))} | {f(st('proxy__proxy_h__m64'))} | {f(st('proxy__proxy_h__full'))} |
| true residual, true h | {f(st('oracle_residual__true_h__m17'))} | {f(st('oracle_residual__true_h__m64'))} | {f(st('oracle_residual__true_h__full'))} |

마지막 열의 oracle은 exact ceiling과 일치하는 것을 매 chunk 검사했다.
독립된 기존 12런(M=64, run 동일 가중)에서는 hhat residual/pE가
{f(d['prior_12run_matrix'][0][0])}/{f(d['prior_12run_matrix'][0][1])}였다.
새 표는 다른 생성 경로에서 subsampling했고 prompt 가중도 달라 절대값을 그대로 합치지 않는다.

최종 TP head 대신 동일한 replica head의 layer 79를 참값으로 쓰는 통제에서도
residual/pE coverage는 {f(st('samehead_residual'))}/{f(st('samehead_proxy'))}였다.
위 표의 M=17 전체 prompt 평균과 비교해 head의 수치 경로 차이를 확인할 수 있다.

위치 오차도 실제로 있다. 전체 prompt 평균에서
TV(proxy h, true h)는 **{f(st('h_tvd'))}**이고, true bonus 확률은
**{pct(st('bonus_true'))}**, proxy 추정은 **{pct(st('bonus_proxy'))}**다.
proxy h가 정확히 0인 위치에 놓이는 실제 event 질량은 평균 **{pct(st('h_zero_true_mass'))}**다.
다만 이 손실을 토큰 오차에 단순히 더해 총 손실로 분해할 수는 없다.
후보 선택과 두 추정 오차의 상호작용이 있기 때문이다.

## 5. 실제 평탄화, support 누락, 순위 오류

앞서 완료한 [전체 80층 entropy 측정](../entropy/REPORT.md)은 4 dataset × 3 seed,
34,702 step / 173,510 context다. Layer 56의 draft 위치 동일 가중 평균은
H(pE)-H(pT)=**+0.1589 nats**, signed median +0.0093, 절대차 중앙값 0.3042다.
H(pE)>H(pT)는 52.6%, 반대는 47.4%였다.
따라서 평균의 양수 부호를 모든 context의 단조 평탄화로 해석하면 안 된다.

새 replay에서 실제 reject-event로 가중한 true residual 누락 질량
`sum R(v) 1[e(v)<=q(v)]`는 **{pct(pool['lost_mass'])}**다.
같은 가중에서 TV(p,e)={f(pool['tvd'])}, TV(R,Rhat)={f(pool['residual_tvd'])},
target/proxy top-1 일치율={pct(pool['top1_match'])}였다.

| Entropy 차이 조건 | 위치 비중 | Reject-event 비중 | 누락 residual 질량 | Residual top-3 coverage | pE top-3 coverage |
|---|---:|---:|---:|---:|---:|
''')
    for key,label in [('sharper','ΔH < −0.05'),('similar','abs(ΔH) ≤ 0.05'),('flatter','ΔH > +0.05')]:
        a=s['conditional'][key]
        notes.append(f"| {label} | {pct(a['row_fraction'])} | {pct(a['reject_event_fraction'])} | {pct(a['lost_mass'])} | {f(a['local_residual'])} | {f(a['local_proxy'])} |\n")
    notes.append(f'''
ΔH와 누락 질량의 reject-event 가중 상관계수는 **{s['delta_lost_mass_correlation']:+.3f}**,
ΔH와 `(residual top3 − pE top3)` coverage 차이의 상관은
**{s['delta_local_gap_correlation']:+.3f}**다. 인과관계나 unseen 모델에서의 예측 성능으로 해석하지 않는다.

**Entropy를 유지한 oracle 통제도 추가했다.**

1. *Rank repair*: 실제 e의 확률값 전체를 그대로 두고, 큰 값을 실제 p 순위의 토큰에
   재배치한다. Entropy, peak, 확률값의 histogram이 모두 같다. 최대 entropy 수치 오차는
   {s['rankfixed_entropy_max_error']:.2g} nats다. 이 통제는 초기 결과 확인 후 추가한 기전 분석이며
   배포 정책의 hyperparameter 선택에 사용하지 않았다.
2. *Temperature surrogate*: p를 온도로 변환하여 실제 e의 entropy에 맞춘다.
   p 순위를 보존하지만 실제 e의 전체 histogram까지 보존하지는 않는다.
   p의 정확한 top-logit 동률 때문에 목표 entropy를 만들 수 없는 경우 등이 있어,
   abs(entropy 오차)<0.001인 위치만 양쪽 비교에 포함했다.
   이 조건은 위치의 {pct(feasible['row_fraction'])}, reject-event 질량의
   {pct(feasible['reject_event_fraction'])}를 남긴다.

동일한 비교 위치에서:

| 분포 | TV(p,proxy) | 누락 residual 질량 | Residual top-3 | pE top-3 |
|---|---:|---:|---:|---:|
| 실제 proxy | {f(feasible['tvd'])} | {pct(feasible['lost_mass'])} | {f(feasible['local_residual'])} | {f(feasible['local_proxy'])} |
| 확률 histogram 유지 + true rank | {f(feasible['rankfixed_tvd'])} | {pct(feasible['rankfixed_lost'])} | {f(feasible['rankfixed_local_residual'])} | {f(feasible['rankfixed_local_proxy'])} |
| 같은 entropy의 temperature surrogate | {f(feasible['thermal_tvd'])} | {pct(feasible['thermal_lost'])} | {f(feasible['thermal_local_residual'])} | {f(feasible['thermal_local_proxy'])} |

이 통제의 의미는 **동일한 entropy에서도 residual 품질이 달라진다**는 것이다.
특히 rank repair는 확률값 모양을 그대로 두고 토큰 배치만 바꾼다.
그러나 true p를 이용한 oracle이며, 관측 가능한 보정법의 개선 폭이나
원인별 손실을 더할 수 있는 분해로 해석하면 안 된다. 동률 토큰의 정렬 순서도 하나의 convention이다.

![기전 통제](02_mechanisms.png)

## 6. Temperature calibration 검증과 중요한 추가 수식

[Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html)은 기존 calibration 방법이다.
여기서의 목적은 정답 여부에 대한 confidence calibration이 아니라 최종 target 분포 p를
근사하는 soft-label KL 최소화이며, 다시 candidate coverage 목적과도 구분해야 한다.
''')
    notes.append(r'''
$\beta=1/\tau$, $e_\beta(v)=e_v^\beta/\sum_w e_w^\beta$라 하면

\[
L(\beta)=\operatorname{KL}(p\Vert e_\beta),\qquad
L'(\beta)=\mathbb E_{e_\beta}[\log e]-\mathbb E_p[\log e].
\]

따라서 원래 proxy인 $\beta=1$에서

\[
\boxed{L'(1)=\operatorname{KL}(p\Vert e)-\big(H(e)-H(p)\big).}
\]

그리고 $L''(\beta)=\operatorname{Var}_{e_\beta}(\log e)\geq0$이므로 convex이다.
같은 가중으로 평균했을 때 KL이 entropy 증가보다 크면 $L'(1)>0$이고,
**beta를 줄이는 방향, 즉 tau>1로 더 평평하게 만드는 방향이 KL을 낮춘다.**
잘못된 토큰에 큰 확률을 주는 오류가 있으면 그 확신을 줄이는 쪽이 유리할 수 있다.
따라서 “proxy entropy가 높다 → tau<1이 KL 보정에 맞다”는 추론은 일반적으로 틀리다.

이 식은 candidate hit에 대한 미분식이 아니다. KL을 줄이는 tau가 후보 coverage에도
좋으리라는 보장은 없으며, 그래서 두 목적을 별도로 tuning했다.
''')
    g=d['temperature_gradient']['train']
    notes.append(f'''
실제 train prompt의 동일 가중 평균에서 KL(p||e)={f(g['kl'])},
ΔH={f(g['delta_entropy'])}, 따라서 beta 미분값은 **{ci(g['beta_gradient'])}**였다.
train KL로 선택한 전역 온도는 **tau={sel['tau_kl']:g}**다.
Coverage로 선택한 residual 온도는 `{sel['residual_tau']}`,
tau와 lambda를 함께 선택한 정책은 `{sel['joint']}`다.

후보 점수는 `s_lambda=(p_calibrated-lambda*q)+`이고 bonus는 p_calibrated를 쓴다.
lambda=0은 proxy-only, lambda=1은 residual이다. 최종 verifier의 p/q/샘플링 온도는 바꾸지 않았다.

| Train에서 선택한 정책 | Held-out coverage | 기존 residual 대비 [95% CI] | 기존 pE 대비 [95% CI] |
|---|---:|---:|---:|
''')
    calibration_rows=[('residual','기존 residual'),('proxy','기존 pE'),('KL_calibrated_residual','Grid KL 기준 tau → residual'),
                      ('KL_calibrated_proxy','KL 기준 tau → pE'),('coverage_calibrated_residual','Coverage 기준 tau → residual'),
                      ('lambda_only','tau=1, lambda 선택'),('joint_tau_lambda','tau/lambda 공동 선택')]
    if 'exact_calibration' in d:
        calibration_rows.extend([('exact_KL_residual','연속 KL 최적 tau → residual'),('exact_KL_proxy','연속 KL 최적 tau → pE')])
    for key,label in calibration_rows:
        a=p[key]
        notes.append(f"| {label} | {f(a['test']['mean'])} | {ci(a['test_gain_vs_residual'])} | {ci(a['test_gain_vs_proxy'])} |\n")
    if 'exact_calibration' in d:
        ex=d['exact_calibration'];meta=ex['metadata']
        notes.append(f'''\nGrid 간격의 영향을 확인하기 위해 train prompt의 KL을 연속 변수 beta에 대해
Newton 방법으로 추가 최적화했다. **tau={meta['tau']:.6f}**,
최종 train gradient={meta['history'][-1]['gradient']:.3g}로 수렴했다.
Held-out을 보지 않고 하나의 tau를 확정한 다음 평가했다.
Held-out KL은 원래 {f(d['temperature_gradient']['test']['kl'])}에서
{f(ex['test']['exact::kl']['mean'])}로 바뀌었다.
따라서 coarse grid 선택과 연속 KL 최적화를 구분해서 해석할 수 있다.
이 추가 분석은 새 파라미터나 데이터 분할을 도입하지 않고 같은 train KL 목적을 정밀하게 최적화한다.\n''')
    notes.append('''
전체 grid는 [calibration_grid.csv](calibration_grid.csv)다. h를 고정한 온도 보정,
h만 보정한 정책, true h를 사용한 정책도 저장했다.
여기서 얻은 최적값은 이 grid, 모델, T=1, prompt 분할에 한정한다.
Held-out curve에서 다른 설정이 더 높아 보이더라도 그것을 새로운 선택 결과로 보고하지 않았다.

![보정 실험](01_calibration.png)

## 7. DS 보완 효과의 정확한 의미와 측정

실제 P1과 동일하다고 주장하지 않는 고정 reference DS를 구성했다.
4개 거절 위치 각각에서 관측 draft token을 제외한 q top-3을 준비해 총 12개 root를 만든다.
Bonus 위치의 q는 수집하지 않았으므로 DS에는 bonus root가 없다.
PS는 동일한 DS와 중복된 후보를 제외하고 순서대로 새 후보를 채운다. M=64를 사용했다.
''')
    notes.append(f'''
실제 누락 residual 질량 {pct(pool['lost_mass'])} 중 이 DS가 복구하는 양은
**{pct(pool['ds_recovered_lost'])}**이며, 누락분의 **{pct(pool['ds_recovered_lost']/pool['lost_mass'])}**다.
이는 다른 토큰을 준비하는 비용을 무시한 “순이득”이 아니라 누락분과 DS의 교집합 질량이다.

| 구성 | 고유 root 총수 | 전체 prompt 평균 coverage |
|---|---:|---:|
| DS만 | 12 | {f(st('ds12'))} |
| DS12 + residual PS3 | 15 | {f(st('ds12_add3_residual'))} |
| DS12 + pE PS3 | 15 | {f(st('ds12_add3_proxy'))} |
| Residual PS만 | 15 | {f(st('ps_only15_residual'))} |
| pE PS만 | 15 | {f(st('ps_only15_proxy'))} |
| DS12 + residual PS15 | 27 | {f(st('ds12_add15_residual'))} |
| DS12 + pE PS15 | 27 | {f(st('ds12_add15_proxy'))} |
| Residual PS만 | 27 | {f(st('ps_only27_residual'))} |
| pE PS만 | 27 | {f(st('ps_only27_proxy'))} |

동일 DS12 이후 PS3에서 `pE−residual` 차이는
**{ci(d['ds_pairs']['ds12_add3_proxy_minus_residual'])}**다.
PS15를 추가하면 차이는 **{ci(d['ds_pairs']['ds12_add15_proxy_minus_residual'])}**다.

목적은 `U(D∪C)=U(D)+sum_(c in C\\D) pi(c)`이다.
DS가 일부 누락을 복구한다는 사실과 residual PS가 더 낫다는 주장은 별개다.
또한 고정 root 수는 준비 시간의 동일성을 보장하지 않는다. 실제 P1은 도착 시점,
중단, subtree 재사용, K와 후보 예산이 함께 바뀌므로 위 표를 DUET end-to-end 효과로 쓰면 안 된다.

## 8. 논문에 반영할 해석

수학적 oracle 의미는 유지할 수 있다. 수정할 부분은 “proxy에 p를 대입해도 동일한
우월성이 보장된다”는 해석이며, 평균 entropy를 단일 원인으로 제시하는 것도 피해야 한다.
실제 근사 오차는 critical token의 부호와 순위, 확률값 크기, 위치 hazard를 함께 왜곡한다.

Temperature는 한 개의 전역 파라미터로 **순위를 바꾸지 못한다**.
확률 보정의 출발점으로 검증할 가치는 있지만, 이번 결과에서 target KL이 좋아진 것을
residual cache 점수 개선이라고 바꾸어 표현해서는 안 된다.
Lambda sweep 역시 cache predictor의 목적함수로 평가해야 하며 verifier 수식을 바꾸는 근거가 아니다.

후속 방법의 학습/선택 목적은 실제 joint event와 미준비 후보의 추가 가치에 가까워야 한다.
다만 `E[hR | e,q,관측]`이라는 Bayes 점수는 기대값을 추정할 데이터·모형이 있어야 구현된다.
단순한 평균 p의 residual이나 Gaussian positive-part expectation을 그대로 대입하면,
위치·토큰 간 의존성과 정규화 때문에 같은 점수가 되지 않는다.

간단한 반례로 q=(0.5,0.5), 관측 y=A이고, p가 (0.9,0.1) 또는 (0.1,0.9)일
가능성이 각각 1/2이라고 하자. 전자는 이 위치에서 reject하지 않고, 후자는 확률 0.8로
reject하여 B로 복구한다. 따라서 E[hR(B)]=0.4지만 평균 p=q를 대입하면 reject 예측은 0이다.
분포 평균을 잘 추정하는 것과 correction event의 기대값을 잘 추정하는 것은 다르다.

이번 실험은 한 모델 쌍, AWQ, T=1, only-proxy chain, 선택된 prompt 집합의 결과다.
원 논문의 T=0.7, tree 및 실제 P1/P2 일정, 다른 모델 쌍, 보정 후 TPS까지 검증한 것은 아니다.
공유 글의 구조적 설명을 지지하는 것과 실제 시스템의 주원인 및 최종 개선 폭을 확정하는 것을
구분해야 한다.

## 9. 검증과 재현 파일

- [checks.json](checks.json): 합성 simplex 36,000건의 강화 상계, exact-sampling 질량 항등식,
  q/e mixture residual 불변성, 원문 4-token·6-position 예제, temperature 미분식 수치 검사.
- 실제 `chain_proxy_candidates_fixed` 함수 본문을 그대로 실행한 40건에서 replay의
  선택 token/position 및 점수가 일치했다. 별도 엔진을 import해 초기화하지 않았다.
- Snapshot 저장 전 확률의 유한성·비음수·합, 입력/RNG 보존, 최종 부분 chunk flush를 검사했다.
  모든 run의 tap은 engine과 exact match이고 모든 prompt의 예정된 sampling step을 확인했다.
- 모든 replay에서 h 합 및 joint probability 합을 검사했고, 15개 후보 정책이 exact oracle을
  넘지 않는지 확인했다. 최종 replica를 reference로 바꾸는 same-head 통제도 저장했다.
- [replay.py](replay.py), [aggregate.py](aggregate.py), [make_report.py](make_report.py),
  [checks.py](checks.py), [results.json](results.json), [policies.csv](policies.csv),
  [conditional.csv](conditional.csv). 그림은 PNG와 PDF를 함께 제공한다.
- [calibrate_exact.py](calibrate_exact.py), [exact_calibration.json](exact_calibration.json):
  train KL 연속 최적화와 수렴 기록. [audit.py](audit.py), [validation.json](validation.json):
  원시 chunk SHA256, 확률합, log floor 질량, 코드·dataset hash 확인.
- 모델 실행·환경·원시 데이터 경로: [실험 README](../../../ssd/experiments/proxy_source_ablation/probe_replay_20260910/README.md).
  같은 디렉터리의 `CAMPAIGN.json`, `out/*.complete.json`에 command와 provenance가 있다.

```bash
OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/checks.py
CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/replay.py
CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/calibrate_exact.py
OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/aggregate.py
ssd/.venv/bin/python results/residial_dist/shared_review/make_report.py
```
''')
    (out/'REPORT.md').write_text(''.join(notes))
    print(out/'REPORT.md')


if __name__=='__main__':main()
