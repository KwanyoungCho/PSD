1. Draft를 두 역할로 분리한다: main path와 cache path

SAGUARO의 아쉬운 점은 acceptance와 cache-hit를 같은 sampling policy로 동시에 최적화하려 한다는 것입니다. 그래서 MESA-SSD에서는 이 둘을 아예 분리합니다.

main path

현재 speculative window $y_{1:L}$는 acceptance가 높은 체인으로 만듭니다.
즉, 여기서는 standard SD sampling이나 아주 약한 Saguaro-style downweighting만 씁니다. main path는 target이 최대한 많이 accept하게 만드는 것이 목적입니다. SAGUARO가 지적한 acceptance–cache-hit trade-off를 main path에서 크게 건드리지 않겠다는 뜻입니다.

cache path

동시에 draft device는 **“verification outcome이 이럴 것 같다”**는 가정 하에 auxiliary branches를 만듭니다. 이쪽은 acceptance가 아니라 cache coverage가 목적입니다.
즉, main path는 “잘 맞는 예측”, cache path는 “miss 안 나는 보험”입니다.

이 분리는 꽤 중요합니다. 이렇게 하면 Saguaro sampling이 안고 있던 acceptance 저하 리스크를 줄이면서도, cache coverage는 더 공격적으로 올릴 수 있습니다.

2. Early-exit를 “첫 토큰”이 아니라 “현재 verify window 전체”에 쓴다

네가 말한 포인트가 바로 여기서 살아납니다.
한 번의 verify pass는 첫 결과만 주는 게 아니라, 현재 speculative window 안의 각 position에 대한 intermediate state를 이미 계산하고 있습니다. Mirror-SD는 이 중에서 next decision point 쪽 proxy top-$k$를 token channel로 보내는데, 내 제안은 이걸 window 전체의 sparse proxy matrix로 확장하는 것입니다. 이건 Mirror-SD의 직접 결과가 아니라 그 구조에서 나오는 제 확장입니다.

다만 full vocab logits를 모든 position에 대해 전부 뽑으면 비싸니, 모바일/병렬 환경에서는 다음처럼 합니다.

target이 early-exit layer에서 뽑아 보내는 정보

position $i=1,…,L+1$마다 전부 보내는 건 아니고, 두 단계로 나눕니다.

모든 position에서 보내는 cheap scalar
현재 draft token $y_i$에 대한 proxy score
entropy 또는 top-1/top-2 margin
간단한 disagreement score

이건 **“어디서 reject가 날 가능성이 큰가”**를 추정하는 데 씁니다.

위험한 position에만 보내는 top-$k$

위 cheap scalar를 보고 reject risk가 큰 position 몇 개만 골라서,

correction candidate top-$k_i$
각 token의 log-prob
를 보냅니다.

즉, 모든 position에 대해 힌트를 얻되, 비싼 top-$k$는 risky positions에만 sparse하게 요청하는 구조입니다.

3. Verification outcome posterior를 online으로 추정한다

이제 draft logits와 early-exit proxy를 합쳐서, 각 outcome의 posterior를 직접 만듭니다.

현재 main draft chain이 $y_{1:L}$라고 하겠습니다.
draft model의 분포를 $p_i^D(⋅)$, early-exit proxy 분포를 $p_i^E(⋅)$라고 두겠습니다.

(a) 각 위치의 accept probability proxy

draft token $y_i$가 accept될 확률을 아래처럼 근사합니다.

$\alphâ_i = \text{Calibrate}(\min(1, p_i^E(y_i) / p_i^D(y_i)))$

직관은 간단합니다.
proxy가 draft token을 좋게 보느냐 나쁘게 보느냐로 accept 가능성을 본다는 뜻입니다. calibration은 최근 몇 step의 실제 verify 결과로 online 보정합니다.

(b) 첫 reject가 i에서 날 확률

$\hat{h}_i = (∏_{j<i} \alphâ_j)(1 − \alphâ_i), i ≤ L$

모두 accept될 확률은

$\hat{h}_{L+1} = ∏_{j=1}^L \alphâ_j$

즉, 먼저 reject 위치 분포를 만듭니다.

(c) correction token 분포

SAGUARO의 핵심은 bonus token이 residual distribution에서 나오므로 그걸 잘 맞춰야 한다는 점입니다. 그래서 correction token 후보는 draft top-logits만 쓰지 않고, early-exit proxy 기반 residual proxy로 만듭니다.

$\hat{r}_i(v) ∝ [p_i^E(v) − β_i p_i^D(v)]_+$

여기서 $v = y_i$는 제외합니다. SAGUARO도 sampled token은 bonus token이 될 수 없으므로 cache candidate에서 제외합니다. all-accept branch ($i=L+1$)에서는 correction이 아니라 다음 bonus token이므로 그냥 $p_{L+1}^E$를 씁니다.

(d) 최종 outcome posterior

$\hat{P}(i,v) = \hat{h}_i ⋅ \hat{r}_i(v)$

이제 우리는 “몇 개 accept + correction token”의 joint posterior를 step마다 가지게 됩니다.
이게 Saguaro의 fixed geometric prior보다 더 좋은 이유는, 이번 문맥의 실제 위험 위치와 correction token을 반영하기 때문입니다.

4. Cache를 flat list로 만들지 말고, 계층형으로 만든다

이 부분이 가장 중요합니다.

SAGUARO에서는 outcome마다 full speculation을 준비하는 쪽에 가깝습니다.
MESA-SSD에서는 cache를 세 층으로 나눕니다.

ABC cache
A. Anchor cache

각 reject position $i$마다
**“$y_{<i}$까지 accept된 상태의 draft prefix state”**를 anchor로 둡니다.

이건 사실 current draft chain을 만들면서 이미 대부분 계산된 state라, compute 측면에서는 싸고 재사용 가치가 큽니다.

B. Branch cache

선택된 anchor $i$에 대해, correction token 후보 $v$ 몇 개를 붙인 node를 만듭니다.

즉 cache key는 $(i,v)$입니다.

C. Continuation cache

각 $(i,v)$에서 앞으로 $d_{i,v}$ 토큰만큼 rollout한 continuation을 둡니다.

5. Budget은 세 단계로 나눠 배분한다

제일 중요한 budgeting rule은 이겁니다.

(1) 위치를 먼저 고르고 → (2) 토큰을 고르고 → (3) 깊이를 고른다.

이게 fixed fan-out보다 좋은 이유는, budget이 적을수록 “exact branch 몇 개”보다 **“많은 위치를 anchor로라도 커버”**하는 게 miss 완화에 더 유리하기 때문입니다.

Step 1: Anchor budget

reject hazard $\hat{h}_i$가 큰 position부터 anchor를 고릅니다.
누적 위험 질량이 예를 들어 80~90%가 될 때까지, 혹은 anchor budget이 찰 때까지 선택합니다.

즉, 먼저 어디서 틀릴 가능성이 큰가를 넓게 커버합니다.

Step 2: Token budget

선택된 position $i$ 안에서 correction token 후보 top-$k_i$를 고릅니다.
여기서는 $\hat{r}_i(v)$가 큰 token부터 채웁니다.

Step 3: Depth budget

각 $(i,v)$ branch에 continuation depth $d_{i,v}$를 배정합니다.
여기서는 expected latency saved per cost를 씁니다.

$Score(i,v,d) = \hat{P}(i,v) ⋅ \Delta T(d) / Cost(i,v,d)$

$\Delta T(d)$: depth $d$까지 precompute했을 때 절약되는 draft latency
$Cost$: branch rollout cost

점수가 큰 branch부터 한 토큰씩 depth를 늘리는 greedy knapsack으로 가면 됩니다.

6. Exact miss를 soft miss로 바꾼다

이 구조의 진짜 장점은 miss가 세 단계로 나뉜다는 점입니다.

1) Full hit

실제 outcome이 $(i*,v*)$이고, 그 branch continuation까지 준비되어 있으면 즉시 사용합니다.

2) Anchor hit

실제 reject 위치 $i*$는 맞췄는데 correction token $v*$는 cache에 없으면,
anchor $A_{i*}$에서 실제 correction token을 넣고 그 시점부터만 draft를 다시 시작합니다.

이건 full miss보다 훨씬 싸고, Saguaro의 binary miss를 graded miss로 바꿉니다.

3) Hard miss

reject 위치 자체가 anchor에 없으면 일반 fallback으로 갑니다.

즉, limited budget일수록 “full branch를 적게 저장하더라도 anchor를 많이 저장”하는 쪽이 낫다는 겁니다.
이게 “cache를 다른 방식으로 만들어야 하지 않나?”에 대한 제 답입니다.
네, 맞습니다. flat outcome cache보다 hierarchical cache가 훨씬 낫습니다.