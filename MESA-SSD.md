# MESA-SSD: Multi-Exit Speculative Assist for SSD

## 동기

SAGUARO는 acceptance rate와 cache-hit rate를 **같은 sampling policy**(downweight sampling)로 동시에 최적화하려 한다. 이 때문에 acceptance rate가 하락하는 trade-off가 발생한다.

MESA-SSD는 이 문제를 해결하기 위해:
- **Acceptance rate는 건드리지 않는다**: downweight sampling을 사용하지 않고 standard SD sampling을 유지한다.
- **Cache coverage는 별도로 최적화한다**: target의 early-exit proxy 정보를 활용하여, 독립적으로 cache token을 배치한다.

핵심 아이디어는 **draft 과정을 early-exit 전후 2단계로 나누는 것**이다.

---

## 전체 흐름 개요

```
[Draft Phase 1]  ──early-exit proxy 수신──▶  [Draft Phase 2]
 Standard SSD                                Proxy-guided cache preparation
 (no downweight)                             (reject 예측 + cache token 배치)
      │                                              │
      ▼                                              ▼
 Main draft chain y_{1:L}                    Cache tokens + rollout
      │                                              │
      └──────────────┬───────────────────────────────┘
                     ▼
              Target verification
                     │
               ┌─────┴─────┐
               ▼           ▼
            Full hit    Cache miss
            (cache)    (fallback to
                       standard SD)
```

---

## Phase 1: Early-Exit 이전 — Standard SSD

Early-exit proxy를 받기 전에는 target으로부터 어떤 힌트도 없다. 따라서 이 단계에서는:

- **SAGUARO의 기본 SSD 프레임워크**를 따른다 (outcome 기반 cache lookup 포함).
- 단, **downweight sampling은 사용하지 않는다**. Acceptance rate 하락을 방지하기 위해 standard SD sampling을 유지한다.
- 이 단계의 목적은 **acceptance rate가 높은 main draft chain $y_{1:L}$을 생성**하는 것이다.

---

## Early-Exit Proxy 정보

Target이 verification을 시작하면, early-exit layer에서 proxy 정보를 추출하여 draft device로 전송한다.

Target은 verify 요청 시 draft로부터 $p_i^D(\cdot)$를 이미 받은 상태이므로, early-exit layer의 $p_i^E(\cdot)$와 합쳐 **residual proxy까지 직접 계산**할 수 있다.

### 전송 정보 (2단계)

**1) 모든 position에서 전송하는 cheap scalar**
- 현재 draft token $y_i$에 대한 accept probability proxy $\hat{\alpha}_i$
- Entropy 또는 top-1/top-2 margin

→ **"어디서 reject가 날 가능성이 큰가"**를 추정하는 데 사용한다.

**2) 위험 position에만 전송하는 residual top-$k$**

위 cheap scalar를 기준으로 reject risk가 큰 position 몇 개를 골라서, target이 직접 residual proxy를 계산한다:

$$\hat{r}_i(v) \propto \left[p_i^E(v) - \beta_i \, p_i^D(v)\right]_+$$

그 결과에서 top-$k_i$ token ID + 확률값을 전송한다.

→ Draft device는 받은 residual 분포를 **그대로 사용**하여 cache token을 sampling하고 budget을 배분한다. 별도의 residual 계산이 필요 없다.

---

## Phase 2: Early-Exit 이후 — Proxy-Guided Cache Preparation

Early-exit proxy를 수신한 뒤, draft model은 이를 활용하여 cache token을 배치한다.

### Step 1: Accept Probability 추정

Draft token $y_i$가 accept될 확률을 proxy 기반으로 근사한다.

$$\hat{\alpha}_i = \text{Calibrate}\left(\min\left(1,\; \frac{p_i^E(y_i)}{p_i^D(y_i)}\right)\right)$$

- $p_i^D(\cdot)$: draft model 분포
- $p_i^E(\cdot)$: early-exit proxy 분포
- Calibration은 최근 몇 step의 실제 verify 결과로 online 보정한다.

### Step 2: 첫 Reject 위치 분포

첫 reject가 position $i$에서 발생할 확률:

$$\hat{h}_i = \left(\prod_{j<i} \hat{\alpha}_j\right)(1 - \hat{\alpha}_i), \quad i \leq L$$

모두 accept될 확률:

$$\hat{h}_{L+1} = \prod_{j=1}^{L} \hat{\alpha}_j$$

### Step 3: Correction Token 분포 (Residual Proxy)

Target이 early-exit 시점에 계산하여 전송한 residual top-$k$를 그대로 사용한다.

$$\hat{r}_i(v) \propto \left[p_i^E(v) - \beta_i \, p_i^D(v)\right]_+$$

- $v = y_i$는 제외 (reject된 토큰은 correction 후보가 될 수 없음)
- All-accept branch ($i = L+1$)에서는 bonus token이므로 $p_{L+1}^E$를 사용
- 이 계산은 **target 쪽에서 수행**된다 (target은 $p_i^D$와 $p_i^E$를 모두 보유)

### Step 4: 최종 Outcome Posterior

$$\hat{P}(i, v) = \hat{h}_i \cdot \hat{r}_i(v)$$

이 posterior는 **"position $i$에서 reject되고, correction token이 $v$일 확률"**을 나타낸다. SAGUARO의 fixed geometric prior와 달리, 현재 context의 실제 위험 위치와 correction token을 반영한다.

---

## Cache Token 배치 및 Budget 배분

### Cache 구조

각 cache entry는 다음으로 구성된다:
- **Root**: reject position $i$에서의 correction token $v$ (residual proxy 분포에서 sampling)
- **Rollout**: root $(i, v)$에서 이어지는 $d_{i,v}$ 토큰의 continuation

### Budget 배분 Policy

세 가지 배분 정책을 고려한다.

#### Policy S: SAGUARO Baseline (Fixed Geometric Prior)

SAGUARO의 기존 방법을 그대로 사용한다. Early-exit proxy 없이 draft model의 정보만으로 budget을 배분한다.

**1) Outcome 확률 추정**

각 position에서의 reject 확률을 fixed geometric prior로 모델링한다. Draft model의 top-$k$ token 확률을 기반으로 각 outcome의 가중치를 결정한다.

**2) Token 선택**

각 outcome에 대해 draft model의 top-$k$에서 correction token 후보를 선택한다.

**3) Depth 배분**

Outcome 확률에 비례하여 각 branch에 depth를 배정한다.

이 방식은 **early-exit proxy를 사용하지 않으므로** Phase 1에서도 동작 가능하며, baseline 비교 대상으로 사용한다.

#### Policy A: Reject 위치 기반 ($\hat{h}_i$만 사용)

첫 reject 위치 분포 $\hat{h}_i$만을 기준으로 budget을 배분한다.

**1) Position 선택**

$\hat{h}_i$가 큰 position $i$부터 우선적으로 선택한다. 누적 위험 질량이 목표치(예: 80~90%)에 도달하거나 budget이 찰 때까지 선택한다.

**2) Token 선택**

선택된 position $i$에서 residual top-$k$의 확률이 큰 correction token $v$부터 할당한다.

**3) Depth 배분**

선택된 $(i, v)$ branch에 continuation depth $d_{i,v}$를 배정한다.

이 방식은 **"어디서 reject가 나는가"에 집중**하며, correction token 확률은 position 내부에서만 반영한다. 구현이 단순하고, reject 위치를 넓게 커버하는 데 유리하다.

#### Policy B: Outcome Posterior 기반 ($\hat{P}(i, v)$ 사용)

최종 outcome posterior $\hat{P}(i, v) = \hat{h}_i \cdot \hat{r}_i(v)$를 기준으로 budget을 배분한다.

**1) Position × Token 선택**

$\hat{P}(i, v)$가 큰 $(i, v)$ 쌍부터 우선적으로 cache에 할당한다. 이 방식은 reject 위치뿐 아니라 해당 위치에서 나올 correction token의 확률까지 함께 고려한다.

**2) Depth 배분**

선택된 $(i, v)$ branch에 continuation depth $d_{i,v}$를 배정한다.

이 방식은 **reject 위치와 correction token을 joint하게 고려**하므로, 특정 position의 특정 token에 확률이 집중되어 있을 때 budget을 더 효율적으로 쓸 수 있다.

#### 공통: Depth Score

Policy A, B 모두 depth 배분 시 다음 score를 사용한다 (Policy S는 SAGUARO 자체 방식을 따른다):

$$\text{Score}(i, v, d) = w(i, v) \cdot \frac{\Delta T(d)}{\text{Cost}(i, v, d)}$$

- $w(i, v)$: Policy A에서는 $\hat{h}_i \cdot \text{rank\_weight}(v)$, Policy B에서는 $\hat{P}(i, v)$
- $\Delta T(d)$: depth $d$까지 precompute 시 절약되는 draft latency
- $\text{Cost}(i, v, d)$: branch rollout cost

점수가 큰 branch부터 한 토큰씩 depth를 늘리는 **greedy knapsack**으로 배분한다.

---

## Cache Hit / Miss 처리

### Full Hit
실제 outcome이 $(i^*, v^*)$이고, 해당 branch의 continuation까지 cache에 준비되어 있으면 **즉시 사용**한다.

### Cache Miss
실제 outcome $(i^*, v^*)$가 cache에 없으면 (reject 위치가 없거나, 위치는 맞지만 correction token이 없는 경우 모두 포함), **standard SD 방식으로 fallback**한다.

---

## SAGUARO 대비 차별점 요약

| | SAGUARO | MESA-SSD |
|---|---|---|
| **Acceptance 최적화** | Downweight sampling (acceptance 하락 위험) | Standard SD sampling (acceptance 유지) |
| **Cache 최적화** | 같은 sampling policy로 동시 최적화 | Early-exit proxy로 별도 최적화 |
| **Outcome 예측** | Fixed geometric prior | Online posterior (context-aware) |
| **Cache miss 처리** | Binary (hit or miss) | Full hit 또는 standard SD fallback |
| **핵심 정보원** | Draft model만 사용 | Draft + target early-exit proxy |
