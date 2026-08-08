# DUET P1/P2 동적 트리: 설계, 구현, 검증 기준 문서

- 최종 갱신: 2026-08-08
- 대상 브랜치: `feat/duet-p2tree-g0`
- 공개 정책: `duet_p1_tree_policy=off|on`, `duet_p2_tree_policy=off|on`

이 문서는 DUET의 P1/P2 chain을 확률 기반 동적 트리로 일반화한 전체 과정을 논문과
코드 감사에 사용할 수 있도록 한 곳에 정리한다. 성공한 구현뿐 아니라 반증된
가설, 실제 버그, 측정 오류, 현재 남은 제한도 포함한다. P2 tree 관련 과거 문서
15번과 17--29번은 [`internal/`](internal/)에 보관하며, 기존 DUET 일반 문서는
원래 위치에 유지한다. 현재 tree 계약은 이 문서와 코드가 결정한다.

---

## 1. 한눈에 보는 현재 상태

### 1.1 완료된 것

1. P2의 네 번의 draft forward 사이에서 CPU가 트리를 다시 만들던 공백을 없앴다.
   네 번의 forward, sampling, 부모 선택, 자식 삽입과 mask 갱신이 하나의 CUDA
   Graph replay 안에서 실행된다. timeline에서 P2가 하나의 막대로 보이는 이유도
   이것이며, 실제 forward가 한 번으로 합쳐진 것은 아니다.
2. 첫 P2 forward는 proxy가 고른 모든 root를 평가한다. 기본 설정은
   `R=W=10`이며, `R>W`는 첫 round에서 모든 root를 평가할 수 없으므로 거부한다.
3. production `on` 정책은 round 0에서 모든 root를 평가한 뒤, 이후 round마다
   누적 경로 점수가 높은 자식을 전역 선택한다. 기본 `R=W`에서도 token과 parent
   topology, root별 깊이와 valid node 수가 replay마다 달라진다.
4. temperature가 0보다 클 때, 한 부모의 여러 자식은 ordered sampling without
   replacement로 생성되고 target은 이에 맞는 residual ladder로 lossless하게
   검증한다.
5. CPU 참조, eager GPU arena, production CUDA Graph 실행기의 token, topology,
   mask와 동일-noise parity를 테스트로 고정했다.
6. 잘못된 page/slot/position, 비활성 lane, 중복 scatter, graph bucket 공유,
   runtime attention plan 등의 주요 안정성 문제를 수정했다.
7. target tree verify 준비는 실모델 P2 hit 기준 1.72ms에서 0.99ms로 줄였다.
8. 같은 고정-shape 실행기를 P1에도 확장했다. P1은 position별 root 수를
   균등하게 정하고 context reach×root 확률을 초기 점수로 사용한 뒤 P2와 같은
   전역 선택을 한다. 아홉 번의 forward 사이 host 개입은 없고, root별 응답
   상한은 18이다.
9. P1의 `F*W>63` 형상을 위해 조상 관계를 63-bit word 여러 개로 확장했고,
   P1/P2가 서로 다른 최대 node 수를 사용하도록 공통 응답 wire와 target verifier를
   일반화했다.

### 1.2 기본값 변경의 정확한 의미

공개 CLI에는 외부 기법 이름을 사용하지 않는다. 현재 기본값은 P1 `off`, P2
`on`이지만 재현 실험은 두 값을 모두 명시한다. 전체 chain은
`--duet_p1_tree_policy off --duet_p2_tree_policy off`, 전체 동적 tree는 두 값을
모두 `on`으로 둔다. 과거 `duet_tree_policy` 값들은 저장된 실험의 재현용 내부
호환 경로로만 남아 있다.

이 변경은 **tree가 이미 모든 workload의 성능 champion이라는 선언이 아니다.**
과거 전역 점수 formal은 이후 수정된 실행기 버그가 섞인 코드에서 수행돼 정책
판정 자료로 사용할 수 없다. 현재 production `on`은 P2의 P1 도입 전 전역 동적
알고리즘을 복원하고 P1도 같은 확장을 사용한다. 정확성 gate 뒤 동일 workload의
다중 seed 실험으로 AL, hit와 추가 target 검증비를 다시 확정해야 한다.

### 1.3 현재 기본 설정

| 기호 | 코드/CLI | 기본 실험값 | 의미 |
|---|---|---:|---|
| K1 | `duet_phase1_k` | 9 | proxy를 기다리는 동안 수행하는 P1 draft round 수 |
| K2=F | `duet_phase2_k` | 4 | proxy 수신 후 P2 draft round 수 |
| W | `duet_p2_budget`에서 유도 | 10 | 한 P2 forward가 동시에 평가하는 부모 수 |
| R | `duet_tree_root_count` | `None` → W | 첫 round에 평가하고 cache에 보존할 root 수 |
| C | `duet_tree_c_tensor` | 3 | 한 부모에서 한 번에 뽑는 ordered 자식 수 상한 |
| N1 | `duet_p1_tree_max_nodes` | 18 | P1 root 하나의 응답 node 상한 |
| N2 | `duet_p2_tree_max_nodes` | 8 | P2 root 하나의 응답 node 상한 |
| U1 | `duet_p1_roots_per_position` | 2 | P1 context마다 만드는 시작 root 수 |
| W1/R1 | `duet_p1_tree_forward_scale` | 1.0 | P1 forward 폭/root 수 비율 |
| τproxy | `duet_tree_proxy_threshold` | 0.01 | P2의 round 1 이후 확장 threshold |
| τconf | `duet_tree_conf_threshold` | 0.03 | P1/P2의 round 1 이후 확장 threshold |

W와 R을 혼동하면 안 된다. W는 모델 forward의 물리 폭이고 R은 의미 있는 root
수다. 기본은 R=W이며, P2의 첫 forward에서 열 개 root가 모두 실제 평가된다.

---

## 2. DUET의 전체 비동기 흐름

DUET는 target과 draft를 서로 다른 GPU에서 동시에 움직인다.

1. target은 현재 token을 처리하면서 중간 layer의 early-exit logits를 만든다.
2. draft는 그 정보가 도착하기 전 P1 chain 또는 동적 tree를 먼저 만든다.
3. target은 P1 결과와 early-exit 분포를 사용해 다음 context 후보와 그 확률
   `P_iv`를 계산하고 draft로 보낸다.
4. draft는 상위 R개의 P2 root를 만들고 P2를 네 round 수행한다.
5. target은 준비된 cache key가 현재 context와 정확히 일치하면 P1 또는 P2
   후보를 검증한다. key는 시간 제한이 아니라 값의 exact match다.
6. target은 수락한 경로의 KV만 canonical 위치로 옮기고 recovery token을 낸다.

P1/P2의 chain 경로와 기존 KV pool은 삭제되지 않았다. phase별 `off|on` 설정으로
chain과 동적 tree를 독립적으로 비교하며, tree가 hit했을 때만 topology sidecar와
tree verifier를 사용한다.

cache hit은 단순한 실행 타이밍 지표가 아니다. `(sequence, terminal context,
recovery token)`의 정확한 key를 미리 만들었는지를 나타내므로 root coverage와
이전 step의 수락 궤적을 함께 반영한다. “draft가 너무 빨라져 hit이 떨어졌다”는
과거 가설은 0/5/10/13ms 인위적 지연 실험에서 방향성 없이 기각됐다.

---

## 3. Target이 P2 root와 proxy 점수를 만드는 방법

### 3.1 chain context별 종단 가능성

P1 또는 기존 후보의 각 위치 `i`에서 draft가 제안한 token을 `y_i`, target
early-exit 분포를 `pE_i`, draft 분포를 `pD_i`라 두면 조건부 수락확률은

```text
alpha_i = min(1, pE_i(y_i) / pD_i(y_i)).
```

앞 token이 모두 수락되고 i번째가 기각될 확률과 전부 수락될 확률은

```text
h_0 = 1 - alpha_0
h_i = (product_{j<i} alpha_j) * (1 - alpha_i)
h_K = product_{j<K} alpha_j.
```

기각 위치에서는 `(pE_i - pD_i)_+`를 정규화한 residual 분포를, 모두 수락한
위치에서는 마지막 target 분포를 사용한다. context `i`에서 token `v`가 다음
root가 될 질량은

```text
P_iv(i,v) = h_i * correction_i(v)
```

이다. 모든 `(i,v)` 후보를 펼쳐 상위 wire 후보를 고른 뒤 position, token,
`P_iv`를 draft로 보낸다. token의 상위 비트에 log-scale `P_iv`를 pack하며,
vocabulary가 32768보다 큰 설정은 이 wire 형식 때문에 현재 거부한다.

### 3.2 tree context의 정확한 sibling ladder

이미 tree를 검증한 뒤 다음 proxy 후보를 만들 때는 형제의 기각을 서로 독립이라고
가정할 수 없다. 같은 부모의 자식 `x_0, x_1, ...`을 순서대로 시도하면서

```text
a_j = min(1, R_j[x_j] / D_j[x_j])
reject 후 R_{j+1} = normalize((R_j - D_j)_+)
reject 후 D_{j+1} = normalize(D_j with x_j removed)
```

를 적용한다. 어떤 node까지 도달할 질량에는 앞 형제들이 모두 기각될 확률이
반드시 포함된다. 이 인자를 빼먹은 과거 구현은 두 번째 형제의 prior를 약 2.08배
과대평가했으며 수정됐다. 현재 target proxy CUDA Graph는 이 residual ladder,
terminal mass, 후보 top-k와 wire pack까지 고정 shape로 수행한다.

### 3.3 전송 개수와 중복 제거

target은 P1과 겹칠 가능성을 고려한 wire buffer를 보내고 draft는 P1 seed와
중복되는 `(position, token)`을 제거한 뒤 상위 R개를 고른다. wire 수를 R과
정확히 같게 줄이면 중복 때문에 실제 root 수가 R보다 작아질 수 있으므로 여유
후보가 필요하다. 이 여유는 root 수나 P2 forward 폭을 늘리는 것이 아니라,
dedup 이전 통신 후보만 늘린다.

---

## 4. P1/P2 공통 동적 트리 구성 알고리즘

### 4.1 production 정책: 모든 root를 먼저 평가하고 이후 부모를 전역 선택

공개 CLI의 `--duet_p1_tree_policy on`과 `--duet_p2_tree_policy on`은 모두 내부
`dynamic` 정책에 연결된다. 두 phase는 root 점수의 출처만 다르고, root가 준비된
뒤의 부모 선택, fanout, sampling, attention mask와 출력 구성은 같은 실행기를
사용한다.

- P2 root prior: target early-exit에서 계산한 `P_iv`
- P1 root prior: 해당 context까지 도달할 draft 확률 × 대체 root token의 draft 확률

round 0은 모든 실제 root를 한 번씩 평가한다. 따라서 각 root/cache key는 최소
하나 이상의 검증 가능한 자식을 갖는다. round 1부터는 root별 의무 chain을 두지
않고, 직전 round에서 생성된 모든 자식 중 누적 점수가 높은 `W`개를 다음 draft
forward의 부모로 고른다. 낮은 점수 root는 얕게 끝날 수 있고, 높은 점수 root는
여러 가지가 동시에 깊어질 수 있다. 이것이 현재 DUET의 기본 동적 topology다.

내부 `eagle` 문자열은 P1 도입 전 P2 전역 선택 실험을 재현하기 위한 별칭이며,
`dynamic`과 동일한 선택/fanout 코드를 탄다. 외부 실행 옵션에는 방법 이름을
노출하지 않고 phase별 `off|on`만 사용한다. 과거 `backbone`과 `hybrid` 정책은
비교·이력 재현용으로만 남는다.

형제 순서는 proposal 분포의 일부다. 부모별 fanout은 token identity를 보기 전에
결정하며, ordered without-replacement 순서를 target residual verifier까지 그대로
보존한다.

### 4.2 node 상태와 점수

각 node는 다음 정보를 가진다.

- token id
- arena 안의 부모 인덱스와 root-local 부모 인덱스
- root id, depth, sibling 순서
- 해당 부모를 평가한 forward cell
- 조상 attention bitset
- 원래 draft 확률 `raw_q`
- 누적 로그 우선순위 `logpri`
- 유효 여부와 이미 확장했는지 여부

P2 root `r` 아래에서 `x_1,...,x_d`를 거친 경로의 점수는

```text
score(r, x_1...x_d)
  = P_proxy(r) * q(x_1|r) * ... * q(x_d|r,x_1...x_{d-1})

logpri = log P_proxy(r) + sum log q(x_j | parent_j).
```

P1은 위 식의 `P_proxy(r)` 자리에
`P_context_reach(r) * q(root_token|context)`를 넣는다. 그 이후의 경로 점수는
P2와 완전히 같은 방식으로 갱신된다. 현재 기본 정책에는 beta, 제곱근, depth
bonus를 넣지 않는다. `beta=0.5`는 과거 root-budget 정책 재현용이며 production
dynamic 점수에는 쓰이지 않는다.

### 4.3 고정 실행 틀 안의 동적 내용

CUDA Graph가 요구하는 것은 tensor shape, 주소와 연산 순서가 고정되는 것이다.
선택되는 token과 부모 인덱스까지 고정할 필요는 없다. 따라서 실행 틀은 항상
`F × W = 4 × 10`개의 부모 lane을 갖지만, 각 lane에 들어가는 부모, fanout,
token, attention mask는 replay마다 달라진다. 무효 lane은 안전한 token, rope,
page를 가리키고 mask가 0인 padding으로 남는다.

### 4.4 round별 절차

초기화 시 R개의 root를 arena에 넣는다. root의 `logpri`는 phase별 root prior의
로그이며, root별 저장 가능 node 수는 N1 또는 N2다. 이 값은 최대 응답 용량이지
항상 채워야 하는 topology 크기가 아니다.

각 round `f = 0,...,F-1`은 다음 순서로 실행한다.

1. **전역 부모 선택**
   round 0에는 R개의 root를 모두 선택한다(`R<=W`가 필수). 이후 round에는 바로
   앞 depth에서 생성됐고 아직 확장하지 않은 모든 node를 `logpri` 내림차순으로
   정렬해 상위 W개를 선택한다. 동률은 arena 삽입 순서를 유지하는 stable sort다.
   선택되지 않은 낮은 점수 node는 검증 가능한 leaf로 남지만 추가 forward를
   받지는 않는다.
2. **fanout 결정**
   token을 뽑기 전에 선택 부모의 우선순위와 root별 남은 N 용량을 사용해 보관할
   자식 수를 정한다. 부모당 상한은 C이며, 같은 root의 여러 부모가 선택됐으면
   누적 우선순위 순서로 그 root의 남은 공간을 나눠 쓴다. 남은 future round에서
   최소한의 확장 기회를 보존하되 root별 전체 깊이를 강제하지 않는다.
3. **forward 입력 준비**
   lane의 input id는 선택된 부모 node의 token이고, rope position은
   `root_rope_base + parent_depth`다. attention은 공통 prefix, 해당 root의
   glue context, 부모의 조상 forward cell과 자기 자신만 볼 수 있도록 packed
   mask에 직접 기록한다.
4. **draft forward와 분포 계산**
   실제 draft transformer를 W 폭으로 한 번 실행한다. logits에 draft
   temperature와 `sampler_x`를 적용한 `q`를 만든다.
5. **ordered without-replacement sampling**
   각 token의 `E_v ~ Exp(1)`을 만들고 `q_v / E_v` 상위 C개를 고른다. 이
   exponential race는 같은 분포에서 순차 비복원 추출한 순서와 같다. 선택된
   token의 확률은 형제 제거 후 재정규화한 값이 아니라 원래 `q_v`인 `raw_q`로
   저장한다.
6. **자식 삽입**
   앞서 정한 fanout만큼 lane-major, sibling-major 순서로 자식을 arena와
   phase-local `[R,N1]` 또는 `[R,N2]` 출력 view에 기록한다. 자식 점수는
   `parent.logpri + log(child.raw_q)`다.

현재 구현은 round-synchronous다. round f에서는 정확히 depth f인 node만 고른다.
이번 round에서 선택되지 않은 node가 더 늦은 round에 다시 등장하는 일반 priority
queue는 아니다. 이 제약은 P1/P2에 동일하게 적용된다.

### 4.5 threshold의 현재 상태

기본 threshold는 `proxy=0.01`, `confidence=0.03`이다. round 0에는 적용하지 않아
모든 root/cache key를 유지한다. round 1 이후 P2는 root proxy가 proxy threshold
미만이거나 현재 node의 `raw_q`가 confidence threshold 미만이면 그 node를 더
확장하지 않는다. P1에는 같은-step target proxy가 없으므로 proxy threshold는
적용하지 않고 confidence threshold만 공유한다. threshold 아래 node도 이미
sampled된 leaf에서 삭제되지 않는다.

### 4.6 기본 P2 형상의 정확한 topology

기본 `W=R=10,F=4,C=3,N2=8`에서 round 0은 열 root를 모두 평가하고 각 부모가
ordered child 후보를 만든다. round 1부터는 최대 30개의 자식 중 누적 점수가 높은
10개가 선택된다. 한 root의 형제 두세 개가 동시에 선택될 수도 있고, 낮은 점수
root는 더 이상 선택되지 않을 수도 있다. 따라서 token뿐 아니라 parent layout,
root별 valid node 수와 최대 깊이도 replay마다 바뀐다.

CUDA Graph에는 여전히 4×10개의 forward lane과 고정 주소 buffer가 들어가지만,
각 lane의 실제 parent id, input token, rope position과 attention mask는 device
tensor 값으로 바뀐다. 동적 topology 때문에 forward 사이에 CPU가 개입하지 않는다.

### 4.7 P1 동적 tree specialization

P1에는 같은 step의 target proxy가 없으므로 시작 후보 생성과 root prior 계산만
P2와 다르다. root가 생성된 뒤에는 위 P2 동적 선택과 같은 코드를 사용한다.

1. **현재 context 집합**
   일반 응답에서는 recovery와 chain token들, 이전 tree hit에서는 recovery와
   유효 tree node들이 context다. 각 context는 자신에게 허용된 prefix/조상만 보는
   정확한 glue mask와 실제 token depth를 가진다.
2. **균등 root 수**
   각 context에서 `U1=duet_p1_roots_per_position`개 token을 logits top-k로 고른다.
   이미 반환된 다음 token은 같은 cache key를 중복 생성하지 않도록 제외한다.
   단, 후보 점수는 제외 후 softmax가 아니라 원래 draft proposal 확률
   `q(root|context)`다. 제외된 token이 확률 대부분을 차지할 때 작은 대안이
   confidence 1처럼 보이는 오류를 막는다.
3. **첫 forward**
   모든 실제 root를 반드시 평가한다. 캡처 폭은 `context_bucket*U1`이고 실제
   context가 bucket보다 적으면 나머지 root는 score 0인 안전 padding이다.
4. **이후 K1-1 rounds**
   첫 forward에서 생성된 최대 `R*C`개 자식 중 누적 점수가 높은 W개를 선택하고,
   이후에도 같은 과정을 반복한다. 기본 `W=R`이어도 후보 수가 W보다 많으므로
   topology는 동적으로 바뀐다. P1에는 proxy threshold를 적용하지 않는다.
5. **응답 view와 cache key**
   각 시작 root는 `(sequence, context id, root token)` key와 최대 N1개의 node
   view를 갖는다. 다음 request가 그 key를 hit하면 해당 root의 tree 하나만 공통
   wire로 보내고 target이 lossless 검증한다.

기본 형상에서 context bucket은 10과 19 두 개다. U1=2이므로 P1 forward 폭은
각각 20과 38, round 수는 K1=9다. 첫 번째는 K1/K2 chain과 P2 tree hit를,
두 번째는 최대 18-node P1 tree hit의 19 context를 처리한다. 가능한 context 수를
모두 별도 full-model graph로 만들지 않고 이 두 coarse canvas에 zero-score padding을
사용한다.

P1의 `F*W`는 최대 `9*38=342`라 하나의 64-bit 조상 bitmap에 들어가지 않는다.
현재 구현은 부호 비트를 피한 63-cell word를 여러 개 사용한다. 이 예에서는 여섯
word가 필요하며, mask pack과 child insertion kernel이 필요한 word를 자동으로
선택한다. 따라서 과거 `F*W<=63` 제한은 더 이상 존재하지 않는다.

P1/P2의 node 상한은 서로 독립적이다. 순차 chain 깊이와 일반 logits 통신은
`speculate_k=K1+K2=13`을 유지하고, 정수 token 응답 폭은
`max(speculate_k, active N1, active N2)`로 별도 계산한다.
`duet_tree_wire_nodes=max(active N1,N2)`는 topology와 tree 전용 parent-q sidecar를
맞춘다. 따라서 P1 18-node 설정은 chain logits를 18행으로 키우지는 않지만,
P1 tree hit의 target 검증은 recovery를 포함해 최대 19행이고 tree 전용 parent-q
버퍼도 18행이다. 이것은 추가 AL과 교환하는 실제 비용이므로 결과에서 함께 잰다.

기본 P1 `K1=9,C=3,N1=18`에서 N1은 root별 최대 응답 node 수다. 각 root가 깊이
9를 가져야 한다는 제약은 없으며, N1이 K1보다 작거나 커도 구성 자체는 유효하다.
낮은 점수 root는 round 0의 leaf만 남을 수 있고, 높은 점수 root는 여러 sibling
branch가 N1까지 채워질 수 있다. Config는 N1을 순차 깊이와 결합하지 않고 양수인
고정 응답 용량으로만 검증한다.

---

## 5. P2 CUDA Graph 실행기

### 5.1 캡처 범위

`P2TreeExecutor`는 다음 전체를 한 graph로 캡처한다.

```text
arena/output reset
  -> [부모 선택 -> fanout -> id/rope/mask 준비
      -> raw draft forward -> logits -> ordered sampling -> 자식 삽입] x 4
  -> root별 고정 출력 metadata와 parent-q 연결
```

replay당 Python이 네 forward 사이에 개입하지 않고, 중간 GPU→CPU 복사,
FlashInfer `plan()` 호출, tensor allocation이 없다. timeline의 단일
`p2_graph_replay` 막대 내부에는 네 forward가 연속 실행된다.

### 5.2 attention과 page bucket

FlashInfer plan은 capture 중 또는 replay마다 바꿀 수 없다. 초기화 때 가능한
page 수별 wrapper와 workspace를 독립적으로 만들고 plan한 뒤 graph를 캡처한다.
runtime에는 page id, slot, context length, last-page length와 packed mask의
고정 주소 buffer 내용만 갱신한다.

현재 page canvas는 실제 context 마지막에 예비 page를 포함한다. 예비 page가
block table에 아직 할당되지 않았다고 `-1`을 넘기면, mask가 0이어도 kernel이
그 주소의 Inf/NaN을 읽어 `0*Inf`로 전체 출력이 NaN이 될 수 있다. 따라서 모든
page id는 유효한 물리 page로 대체하고 mask로 완전히 차단한다. position과 slot
역시 음수 sentinel이 model/rotary/KV store에 도달하기 전에 degenerate step으로
감지해 chain/arena fallback 또는 skip한다.

P2 모든 page bucket의 과거 실모델 비용은 약 7--9초와 1014MiB였다. P1을 켠
2026-08-07 스모크에서는 P1 context 10/19 × page 1--7 준비가 약 20--28초와
약 2.8GiB를 추가했다. P1 graph의 transient capture pool은 공유하지만 page별
FlashInfer workspace와 persistent 입력/출력은 분리한다. 이 비용은 decode TPS와
정상 request step에는 포함되지 않지만 cold start와 상주 메모리에는 포함된다.
P1/P2 graph를 모두 준비한 직후에는 live graph가 참조하지 않는 compiler/warmup
cache가 수백 MiB 남을 수 있다. 서비스 준비 신호를 보내기 전에 synchronize 후
`empty_cache()`로 이 미사용분만 반환한다. 실제 graph pool은 유지되며, 24GiB
GPU에서 실제 반환량이 0일 수도 있으므로 이것만 메모리 안전장치로 보지 않는다.
P1과 P2 tree를 동시에 켠 draft는 KV 자동 할당 비율을 80%에서 75%로 낮춰 약
1GiB의 명시적 graph/prefill 여유를 먼저 확보한다. B=1, max length 2048에서는
남은 KV block이 요구량보다 훨씬 많으며, chain/P1-only/P2-only 용량은 바꾸지
않는다.

### 5.3 kernel 융합

초기 구현은 작은 `scatter/gather/one_hot/cumsum` kernel을 수십 개 실행해 graph
안에서도 GPU 시간이 컸다. 현재는 다음을 Triton으로 합쳤다.

- arena와 고정 출력 buffer reset
- 한 round의 child 삽입, parent/local index 기록, root view count 갱신
- 최종 root-local parent-q와 backbone metadata 구성

mask는 `[rows,kv_len]` boolean 행렬을 만들고 pack하는 대신 prefix와 조상/self
bit를 packed buffer에 직접 쓴다. P2 준비에서는 NumPy glue 임시 tensor,
block-table zero-fill, dtype 변환, canvas clone도 제거했다.

저장된 같은 구성의 짧은 profile에서 P2 graph 평균은 12.37ms에서 12.08ms로
약 0.29ms 감소했다. 이는 graph 내부 host gap이 다시 생겼다는 뜻이 아니라,
모델 forward 외에 동적 선택과 비복원 sampling이 쓰는 GPU 계산이 chain보다
여전히 약 2.5--3ms 많다는 뜻이다.

### 5.4 RNG 계약

Graph는 P2 전용 CUDA generator를 등록해 사용한다. replay마다 generator state가
전진하며, 기본 generator를 오염시키지 않는다. same-noise 진단에서는 각 round의
exponential noise buffer를 eager와 graph에 동일하게 주입해 token/topology/mask를
정확 비교한다. 일반 실행의 graph와 eager가 서로 다른 난수열을 썼다는 이유로
bitwise output이 같아야 한다고 요구하지는 않는다.

---

## 6. Target의 tree 검증

### 6.1 검증 row와 attention

root 하나가 hit하면 target 입력은 recovery context 한 행과 최대 Nv개의 tree
node 행이다. 각 node 행은 공통 prefix와 자신의 조상 node만 볼 수 있다. target
KV는 먼저 scratch slot에 쓰고, 수락된 한 경로의 KV만 순서대로 canonical slot에
복사한다. 기각된 형제나 다른 branch의 KV는 commit하지 않는다.

검증 row 수에 맞는 CUDA Graph bucket과 packed mask를 사용한다. runtime
FlashInfer plan을 제거하고 graph가 읽는 buffer를 직접 갱신한 결과, 실모델
P2 hit의 verify 준비는 다음처럼 줄었다.

| 항목 | 이전 | 현재 |
|---|---:|---:|
| 전체 verify 준비 | 1.72ms | 0.99ms |
| attention 준비 | 0.75ms | 0.17ms |
| mask 준비 | 0.57ms | 0.41ms |

page 수 1/2/3과 last-page length 1/15/16에서 기존 plan 경로와 attention 출력이
bit-identical했고, 70B target + 1B draft 실모델에서 283 replay, fallback 0,
오류 0으로 완주했다.

### 6.2 lossless residual walk

현재 context의 target 분포를 `R=p`, 첫 형제가 공유하는 draft 분포를 `D=q`로
시작한다. sibling 순서대로 token `t_j`에 대해

```text
a_j = min(1, R[t_j] / D[t_j])
```

확률로 수락한다. 수락하면 그 자식 context로 내려간다. 기각하면

```text
R <- normalize((R-D)_+)
D <- normalize(D with t_j removed)
```

를 적용하고 다음 형제를 시도한다. 모든 형제가 기각되면 최종 R에서 recovery를
뽑는다. 수락된 node가 leaf이면 그 leaf의 target 분포에서 bonus token을 뽑는다.
형제를 여러 개 뽑는 비용이 작은 이유는 draft transformer forward는 한 번이고
logits 하나에서 C개를 고르기 때문이다. 하지만 형제 순서와 residual 갱신은
정확성에 필수다.

### 6.3 temperature=0

현재 tree 경로는 temperature>0만 지원한다. temperature=0의 one-hot proposal에는
두 번째 비복원 자식이 정의되지 않는다. 단순히 gate를 지우지 말고 별도의 greedy
정책이 필요하다: 부모별 logits top-C를 서로 다른 자식으로 두고 target argmax와
일치하는 자식만 따라가며, residual coin flip은 사용하지 않는다. 구현 전까지는
chain fallback이 맞다.

---

## 7. P1과 cache 품질 보존 계약

P1 policy, root layout, cache key와 chain verifier는 유지한다. P2 tree는 P1의
node를 별도의 branch로 재배치하지 않아야 하며, target proxy 후보를 만들 때
P1과 중복된 root만 제거한다. P1 hit/AL이 tree arm에서 체계적으로 떨어지면
“P1은 안 바뀌었으니 분산”으로 넘기지 않고 proxy wire, RNG stream, 이전 P2
수락 경로와 다음 key 생성을 함께 비교한다.

P2 품질은 다음 세 지표를 분리해서 본다.

1. `P2 hit`: 필요한 root/cache key를 만들었는가
2. `P2 AL`: hit했을 때 평균 몇 node를 수락했는가
3. `P2 contribution = P2_hit * (P2AL + 1)`: hit 폭과 조건부 깊이를 결합한 기여

P2AL만 높이고 root를 버려 hit를 낮추는 정책은 성공이 아니다. 과거 R=6
confidence 정책이 바로 이 문제를 보였다.

---

## 8. 정책별 실험 결과

모든 표는 workload, seed와 profiler 조건이 같을 때만 직접 비교해야 한다.
짧은 smoke와 장기 gate 수치를 섞어 champion을 선언하지 않는다.

### 8.1 3-seed 짧은 final gate: root coverage의 중요성

40 prompts × output 128, seed 42/123/2024, arm 순서 회전, profiler off:

| arm | TPS | tok/step | P1 hit | P1AL | P2 hit | P2AL | P2 기여 | target ms | draft ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 62.37 | 3.487 | 0.505 | 3.403 | 0.274 | 1.553 | 0.700 | 50.46 | 50.38 |
| confidence-R6 | 58.97 | 3.567 | 0.513 | 3.350 | 0.232 | 1.967 | 0.689 | 52.08 | 60.03 |
| coverage-R10 | 60.57 | 3.703 | 0.515 | 3.580 | 0.276 | 1.897 | 0.800 | 52.79 | 60.58 |

confidence-R6는 P2AL을 높였지만 하위 네 root를 버려 hit가 4.17%p 내려갔고,
최종 P2 기여가 chain보다 낮았다. coverage는 root를 모두 보존해 이 gate에서는
P2 기여 +14.4%, tok/step +6.2%를 얻었지만 target row와 draft GPU 계산 때문에
TPS는 2.9% 낮았다.

### 8.2 80-prompt 장기 gate: workload 의존성과 동적 정책 반례

output 384의 장기 seed 결과:

| seed | arm | P2 hit | P2AL | P2 기여 | TPS |
|---|---|---:|---:|---:|---:|
| 42 | chain | 0.250 | 1.76 | 0.690 | 73.65 |
| 42 | 동적 정책, threshold 전 | 0.237 | 1.38 | 0.564 | 65.08 |
| 123 | chain | 0.254 | 1.83 | 0.719 | 69.48 |
| 123 | 동적 정책, threshold 전 | 0.242 | 1.46 | 0.595 | 65.48 |

낮은 proxy root를 깊이 1에서 멈춘 결과, 실제 그 root가 hit했을 때 chain보다
짧았다. 이것은 구현 crash가 아니라 현재 round-synchronous 전역 정책의 품질
반례다.

같은 장기 seed 42에서 coverage는 다음과 같았다.

| arm | TPS | tok/step | P1 hit | P1AL | P2 hit | P2AL | P2 기여 | target verify | draft step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 73.65 | 4.03 | 0.560 | 4.04 | 0.250 | 1.76 | 0.690 | 49.63ms | 49.78ms |
| coverage | 66.98 | 4.01 | 0.574 | 3.86 | 0.240 | 1.93 | 0.703 | 52.18ms | 59.31ms |

P2AL과 P2 기여는 소폭 증가했지만 전체 tok/step은 사실상 같고 TPS는 9.1%
낮았다. 짧은 gate보다 tree 품질 이득이 작아 prompt/output 분포 영향을 보여준다.

### 8.3 threshold calibration 결과

두 서버에서 10×4 sequence, output 384로 동적 정책의 0/0과 0.01/0.03을 비교한
단순 평균:

| 지표 | threshold 0/0 | 0.01/0.03 | 변화 |
|---|---:|---:|---:|
| P2 AL | 1.415 | 1.860 | +0.445 |
| tok/step | 3.960 | 4.135 | +4.4% |
| TPS | 68.685 | 71.480 | +4.1% |
| P1 hit | 0.587 | 0.586 | 동등 |
| P2 hit | 0.237 | 0.246 | +0.9%p |
| target full step | 59.855ms | 59.870ms | 동등 |
| draft step | 57.105ms | 57.090ms | 동등 |

이 결과는 약한 leaf를 삭제해서가 아니라 낮은 확률 leaf 아래의 추가 확장을
막아 같은 고정 Graph 시간 안에서 lane 배치를 바꾼 효과다. 다만 두 seed 결과라
논문 최종값으로는 더 큰 paired dataset이 필요하다.

### 8.4 최신 최적화 smoke

융합 kernel과 target plan 제거 후 실모델 동적 tree smoke는 275 replay,
fallback/error 0, TPS 63.99, tok/step 3.79, P1 hit 0.513, P2 hit 0.255,
P1AL 3.71, P2AL 1.91이었다. 같은 날 coverage smoke P2AL은 2.03이지만 실행
조건과 표본이 같지 않아 우열 비교에 사용하지 않는다.

### 8.5 chain 회귀 확인

pre-tree ancestor `e29c4b6`과 현재 `off`를 같은 짧은 workload에서 비교했다.

| branch | TPS | target verify | draft step | P1AL | P2AL |
|---|---:|---:|---:|---:|---:|
| pre-tree | 69.52 | 46.34ms | 46.13ms | 3.72 | 1.79 |
| current chain | 70.86 | 46.38ms | 46.17ms | 3.74 | 1.99 |

공통 latency 차이는 0.04ms였다. 과거 80+ TPS headline과 최근 수치는 서버 부하,
prompt 수와 output 길이가 다르므로 tree 공통 코드 회귀 증거가 아니다.

### 8.6 과거 P1/P2 전역 선택 formal gate — 성능 근거로 사용 금지

P1을 추가한 직후 public `on`을 전역 누적점수 선택에 연결한 첫 formal gate의
관측값은 다음과 같았다. 그러나 이 실행은 이후 수정된 graph 입력 갱신, root lane,
page/last-page 처리, target plan state와 wide workspace 버그보다 앞선 코드에서
수행됐다. 따라서 아래 값은 당시 구현이 실패했다는 기록일 뿐, 전역 동적 정책의
품질을 판정하는 근거로 사용하면 안 된다.

| server/seed | arm | TPS | tok/step | cache hit | P1 hit | P1AL | P2 hit | P2AL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eslab18/42 | chain | 75.24 | 4.10 | 0.82 | 0.568 | 4.10 | 0.256 | 1.86 |
| eslab18/42 | P1 global | 35.04 | 2.60 | 0.71 | 0.376 | 1.92 | 0.335 | 1.40 |
| eslab18/42 | P2 global | 57.80 | 3.48 | 0.76 | 0.449 | 4.10 | 0.312 | 1.00 |
| eslab17/123 | chain | 77.73 | 3.97 | 0.82 | 0.566 | 3.98 | 0.250 | 1.68 |
| eslab17/123 | P1 global | 37.15 | 2.55 | 0.71 | 0.373 | 1.83 | 0.333 | 1.40 |
| eslab17/123 | P2 global | 62.96 | 3.44 | 0.76 | 0.447 | 4.05 | 0.312 | 1.02 |
| eslab17/123 | P1+P2 global | 33.70 | 2.43 | 0.62 | 0.315 | 2.11 | 0.304 | 0.91 |

당시에는 이 하락을 “root별 전체 깊이를 보존하지 않은 정책 문제”로 단정하고
public `on`을 backbone 정책으로 되돌렸다. 이 판정은 구현 버그와 정책 효과를
분리하지 못한 과잉 수정이었다. commit `b8e8bfd`에서 P2는 P1 도입 전의 전역
동적 알고리즘으로 복원했고, P1도 같은 알고리즘을 사용하도록 통일했다. 위 표와
이를 바탕으로 한 고정-backbone sweep은 새 production 정책의 성능 자료로
재사용하지 않는다.

### 8.7 과거 backbone 정책 실모델 기능 gate

commit `9551466`, eslab17, 4 datasets × 2 prompts, output 64의 짧은
P1+P2 smoke 결과:

| TPS | tok/step | cache hit | P1 hit | P1AL | P2 hit | P2AL | draft step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 54.42 | 4.04 | 0.77 | 0.529 | 4.01 | 0.243 | 2.03 | 70.63ms |

P1 executor와 P2 executor는 각각 136회 replay했고 runtime capture, fallback,
error는 0이었다. 직전 smoke에서는 warmup 허용 정책 목록에 `backbone`이 빠져
P2 graph를 첫 요청 중 capture했고, 그 한 번의 비용이 8-prompt 평균에 섞여
draft 232.24ms/TPS 17.53으로 보였다. warmup 목록을 runtime dispatch와 같은
단일 상수로 통일한 뒤 위 값으로 회복했다.

이 표는 당시 backbone 실행기 배선과 warmup 누락 수정을 확인한 역사적 기능
gate다. 현재 dynamic 정책의 TPS 또는 품질 근거로 사용하지 않는다.

### 8.8 현재 P1 dynamic 실경로 진단과 3-seed 짧은 gate (2026-08-08)

P1만 `on`, P2는 `off`로 두고 chain/P1-tree를 같은 서버에서 seed별로 순서
회전했다. 각 arm은 네 dataset에서 4 prompt씩, output 128이며 profiler는
껐다. 원 로그는
`experiments/proxy_async_overlap/tree_sweep/p1_quality_debug_20260808/`에 있다.

| seed | arm | P1 hit | P1AL | tok/step | TPS | target verify | draft step |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | chain | 0.531 | 3.55 | 3.67 | 71.16 | 46.92 ms | 45.17 ms |
| 42 | P1 dynamic | 0.529 | 3.52 | 3.71 | 52.09 | 58.92 ms | 67.25 ms |
| 123 | chain | 0.502 | 3.58 | 3.62 | 70.63 | 46.56 ms | 44.15 ms |
| 123 | P1 dynamic | 0.546 | 4.40 | 4.19 | 56.90 | 59.31 ms | 71.36 ms |
| 2024 | chain | 0.498 | 3.77 | 3.70 | 72.70 | 46.43 ms | 43.89 ms |
| 2024 | P1 dynamic | 0.499 | 3.92 | 3.83 | 54.41 | 58.16 ms | 66.65 ms |
| 평균 | chain | 0.510 | 3.633 | 3.663 | 71.50 | 46.64 ms | 44.40 ms |
| 평균 | P1 dynamic | 0.525 | 3.947 | 3.910 | 54.47 | 58.80 ms | 68.42 ms |

평균 P1AL은 `+0.313`(`+8.6%`), P1 hit는 `+1.43%p`, tok/step은
`+6.7%`였다. 따라서 과거의 P1AL 1.8--2.1 붕괴는 현재 코드에서 재현되지
않았다. seed 42의 `-0.03`처럼 짧은 stochastic run 하나는 반대 방향일 수
있으므로, “모든 seed의 표본 평균이 반드시 증가한다”는 판정 규칙은 사용하지
않는다.

별도의 real-model audit는 P1 생성 71 step의 모든 1,618 root view를 검사했다.
각 node token과 raw draft 확률이 실제 forward lane의 ordered sample과 exact,
parent cell과 target parent-q reference도 exact였고 누락/중복/OOB는 0건이었다.
34번의 실제 P1 tree hit에서 수락된 116 edge 중 sibling 1/2가 11번 사용돼
대체 가지가 실제 수락 경로에 기여했다. 진단 결과는
`p1_output_exact_20260808/smoke_nodes.p1.jsonl`과 topology trace에 있다.

다만 현재 전역 동적 정책은 모든 root의 first-child chain을 깊이 K1까지 강제
포함하지 않는다. 따라서 같은 root/noise에서 chain proposal의 strict superset인
정책은 아니며, per-hit/per-seed 수학적 non-regression을 주장할 수 없다. 그
보장이 필요하면 mandatory first-child backbone을 유지하고 남는 forward lane만
전역 점수로 배분하는 별도 정책이 필요하다. 이는 현재 “P1/P2 동일 global
dynamic” 정책과 다른 알고리즘 변경이다.

토큰 품질과 별개로 이 P1 형상은 아직 성능 채택에 실패했다. 평균 target verify는
`+12.16 ms`, draft step은 `+24.02 ms`, TPS는 `-23.8%`다. chain P1의 16-lane
forward 대신 일반 hit에서 20 lane, 최대 P1-tree hit 뒤에는 38 lane을 9 round
실행하고, target도 최대 9개가 아닌 18개 node를 검증하기 때문이다. P1 tree는
기본값으로 승격하지 않고 폭/재귀 context 비용을 줄이는 설계가 선행되어야 한다.

---

## 9. Timeline 해석

현재 tree P2의 `p2_graph_replay` p50은 저장 profile에서 약 12.34ms다. 그 안에
네 번의 forward가 있으므로 막대가 하나라고 모델이 한 번만 돈 것은 아니다.
forward 사이 host readback과 runtime plan은 0이다. chain의 네 replay 합은 약
9.2--9.9ms이며, tree는 ordered sampling과 동적 topology GPU kernel 때문에
약 2.5--3ms 더 든다.

warmup 후 target K2 p50의 한 비교는 다음과 같다.

| target 구간 | chain K2 | tree K2 |
|---|---:|---:|
| verify 준비 | 0.36ms | 1.63ms |
| 앞부분 graph | 26.72ms | 33.76ms |
| exit proxy 실제 side work | 0.70ms | 1.24ms |
| 뒷부분 graph | 9.70ms | 13.05ms |
| 수락/복구 | 4.40ms | 4.30ms |

tree가 recovery+4 chain row 대신 recovery+최대 8 node를 검증하므로 model graph
시간이 실제로 증가한다. 이후 verify 준비 최적화로 1.63ms 계열을 약 0.99ms까지
줄였지만 더 많은 row의 transformer 계산 자체는 남는다.

profiler는 exit side stream을 `exit_proxy_launch`와 `exit_proxy_side`로 나누고,
target과 draft에 공통 response marker가 있는 step만 선택한다. 약 23,000번째
CUDA event에서 보였던 700--800ms 막대는 profiler event 축적 stall이었고
event cap을 12,000으로 제한했다. 첫 verifier softmax/multinomial cold start도
warmup에서 분리한다.

---

## 10. 발견하고 수정한 문제와 교훈

### 10.1 알고리즘 및 품질 문제

- **R=6 root 절단**: 깊이를 늘리기 위해 하위 root를 버렸고 P2 hit가 줄었다.
  P2AL만 보아서는 이 실패를 놓친다.
- **P1 sibling prior 오류**: 앞 형제 기각확률이 빠져 후속 형제 prior를
  과대평가했다. 정확한 residual ladder로 교체했다.
- **sampling 분포 불일치**: proxy는 plain softmax, draft walk는 temperature와
  `sampler_x`를 썼다. 두 경로가 같은 `q_probs_from_logits`를 쓰도록 통일했다.
- **정적 topology 오해**: topology는 budget만의 함수가 아니다. sampled q가
  전역 rank와 fanout을 바꾸므로 정적 template은 다른 정책이다.
- **너무 빠른 draft가 hit를 낮춘다는 오해**: cache key에는 timeout이 없으며
  인위적 지연으로 반증됐다.

### 10.2 CUDA 및 메모리 안전 문제

- boolean indexing이 내부 `nonzero`와 device-to-host 동기화를 일으켜 forward
  간 gap을 만들었다. 고정 shape dense write와 scratch slot으로 바꿨다.
- 비활성 lane이 같은 slot 0에 중복 scatter되어 서로 다른 필드를 섞은
  키메라 record를 만들었다. lane별 dummy/scratch routing과 융합 kernel로 수정했다.
- 비활성 lane의 mask self-bit가 1이던 불일치를 arena 규약의 0으로 고쳤다.
- 완전 degenerate step의 `position=-1`, `slot=-1`이 rotary indexSelect assert와
  KV OOB write를 만들 수 있었다. 진입 guard와 store mask를 추가했다.
- canvas `page_id=-1`은 mask=0이어도 Inf/NaN을 읽을 수 있었다. 유효 page id로
  대체하고 mask로 격리하는 규약을 테스트로 고정했다.
- page bucket들이 root-local node index tensor를 공유해 긴 replay에서 illegal
  access가 났다. bucket별 독립 fixed-address state로 분리했다.
- graph capture용 RNG와 기본 generator가 섞이던 문제를 전용 등록 generator로
  분리했다.
- 정규화된 내부 정책 Config를 draft용 `dataclasses.replace()`가 다시 검증하지
  못하던 비멱등 초기화 오류를 수정했다. 현재 `dynamic`도 이 계약을 테스트한다.
- runtime P2 허용 목록과 all-page warmup 목록의 정책 집합이 달라 fallback 또는
  첫 요청 중 capture가 발생하던 배선 오류를 공통 정책 상수로 통일했다.
- `assert` 기반 runtime guard는 Python `-O`에서 사라지므로 외부 입력 계약은
  `ValueError`/`RuntimeError`로 바꿨다.

### 10.3 진단 및 측정 문제

- 초기 보고의 P2 28.3→1.8ms(-94%)는 label 경계 오류였다. 실행기 forward와
  후처리가 무레이블이라 선택기만 집계했다. 공식 경계
  `phase2_build.start -> merge_cache.end`로 재계산한 당시 값은
  약 30.9→20.1ms(-35%)였다.
- graph 뒤의 세 번 `.cpu()`, Python loop와 full-vocab allocation이 약 11ms
  남아 있었다. 고정 GPU view와 parent-q 출력으로 제거했다.
- scratch eager probe가 serving KV를 복원하지 않던 진단 버그를 수정했다.
- draft crash 뒤 target rank가 NCCL에서 무한 대기하던 문제를 watchdog/runner
  cleanup과 bounded tree walk로 방어했다.

---

## 11. 자동 calibration 도구

### 11.1 threshold

`ssd/tools/duet_calibration/`의 수집기와 분석기는 각 root/node의 proxy,
confidence, 실제 hit, 실제 accepted child를 JSONL로 남긴다. 분석기는 후보를
삭제했을 때가 아니라 **그 node 아래 확장을 멈췄을 때** 잃는 실제 수락 기여를
계산해 threshold를 추천한다.

```bash
cd ssd
python tools/duet_calibration/analyze_thresholds.py \
  --input /path/to/calibration.jsonl
```

모델, dataset, exit layer, draft/target temperature, sampler, R/W/C/Nv가 바뀌면
다시 calibration한다. 수집은 threshold 0/0으로 해야 tail 표본을 검열하지 않는다.

### 11.2 K1/K2 latency 균형

K1은 draft P1 완료와 proxy 도착의 signed gap, K2는 target verify 완료와 P2
draft 완료의 signed gap을 직접 사용한다. 양수/음수 부호와 절댓값을 모두 보고
서로 기다리는 시간이 최소인 정수 K를 고른다. 최신 동적 tree profile에서는
K1=9의 gap +5.04ms, K1=10은 -6.26ms라 K1=9가 가까웠고, K2=4는 -0.04ms로
K2=3의 +3.98ms, K2=5의 -4.13ms보다 정확히 균형이었다.

```bash
cd ssd
bash experiments/proxy_async_overlap/tree_sweep/run_k1_k2_calibration.sh
```

도구는 chain과 tree 양쪽 profile을 읽을 수 있지만, 계산한 K가 실제 TPS champion임을
보장하지 않는다. 후보를 줄인 뒤 마지막 paired TPS gate 한 번으로 확정한다.

---

## 12. 테스트와 재현 규율

### 12.1 필수 테스트 계층

1. pure policy: budget, stable ordering, R/W, fanout, parent-before-child
2. CPU vs eager arena: same noise에서 round별 selection/token/topology/mask
3. eager executor vs CUDA Graph replay: page boundary, page/slot 교체, RNG 전진
4. target verify: packed mask, planless attention, residual walk, KV commit
5. 실모델 smoke: 모든 page bucket capture, replay/fallback/error count, NaN/OOB
6. paired performance: profiler off, arm 순서 회전, 다중 seed
7. timeline: 최종 채택 후 한 번만, common response step과 cold-start 제외

주요 테스트 파일은 다음과 같다.

- `tests/test_p2_tree_alloc.py`
- `tests/test_p2_executor_parity.py`
- `tests/test_tree_verify_mask_direct.py`
- `tests/test_tree_verify_planless.py`
- `tests/diag/test_p2_dispatch_contract.py`
- `tests/diag/test_store_kvcache_neg_slot.py`
- `tests/diag/test_cg_input_check.py`

### 12.2 비교 실행

새 실험은 phase별 `--duet_p1_tree_policy off|on`과
`--duet_p2_tree_policy off|on`만 사용한다. 과거 단일 정책 옵션과 외부 기법
이름은 저장된 결과 재현용으로만 남아 있다.

다음은 현재 70B target + 1B draft, GPU 5장 설정을 재현하는 전체 예시다. 경로는
서버별 실제 artifact 위치로 바꿔야 한다. `--k`는 생략하면 K1+K2=13으로 자동
유도되고, `--f 3`은 DUET의 전체 async fanout이다. 이것은 tree의 자식 수 C와
다르다.

```bash
cd /path/to/PSD/ssd
source env.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DIST_PORT=16201

# 최적화된 production tree 실행 경로
export SSD_TREE_EXEC=1
export SSD_TREE_ARENA=1
export SSD_TREE_EXEC_WARMUP=all
export SSD_TREE_PROXY_GRAPH=1
export SSD_CHAIN_PROXY_GRAPH=1
export SSD_DUET_EXIT_REPLICA=1
export SSD_ASYNC_PROXY_SEND=1
export SSD_PROXY_STREAM=0

# 성능 측정에는 진단 기능을 섞지 않는다.
unset SSD_TREE_STAGE1 SSD_TREE_STAGE2 SSD_TREE_TOPO_TRACE
unset SSD_TREE_NODE_AUDIT SSD_TREE_ALLOC_CHECK SSD_TREE_EXEC_DELAY_MS
unset SSD_TREE_EXEC_EAGER_DIAG SSD_TREE_EXEC_SYNC_DIAG SSD_TREE_GAP_PROF
unset SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER

/path/to/conda/envs/ssd/bin/python -O bench/bench.py \
  --llama --size 8 \
  --model_path /path/to/target-70b \
  --quant_awq --quant_awq_artifact /path/to/target-awq-tp4 \
  --quant_group_size 128 \
  --draft_path /path/to/draft-1b \
  --quant_awq_draft --quant_awq_draft_artifact /path/to/draft-awq-tp1 \
  --gpus 5 --b 1 --async --spec --duet \
  --temp 0.7 --input_len 512 --output_len 384 \
  --numseqs 20 --all --max_model_len 2048 --seed 42 \
  --duet_exit_layer 56 --f 3 \
  --duet_k1 9 --duet_k2 4 \
  --duet_p1_fanout 2 \
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 \
  --duet_p2_budget 10 \
  --duet_p1_tree_policy on --duet_p2_tree_policy on \
  --duet_p1_roots_per_position 2 \
  --duet_p1_tree_max_nodes 18 --duet_p2_tree_max_nodes 8 \
  --duet_tree_root_count 10 --duet_tree_c_tensor 3 \
  --duet_tree_proxy_threshold 0.01 \
  --duet_tree_conf_threshold 0.03
```

논문 script에는 P1/P2 정책을 둘 다 명시해 실행 시점의 기본값 변화와 무관하게
재현되게 한다.
`SSD_TREE_EXEC`의 코드 기본값은 진단 호환 때문에 아직 0이므로, 최적화된 전체-P2
Graph를 측정할 때는 반드시 1로 명시한다. `SSD_TREE_ARENA=1`은 지원되지 않는
사전 분류 shape의 안전 fallback과 참조 경로를 보존한다.

chain과 coverage는 위 공통 인자에서 정책 및 tree runtime 환경만 바꾼다.

```bash
# chain 비교군: 공통 exit/proxy 최적화는 유지하고 tree 전용 실행만 끈다.
SSD_TREE_EXEC=0 SSD_TREE_ARENA=0 SSD_TREE_EXEC_WARMUP=0 \
SSD_CHAIN_PROXY_GRAPH=1 SSD_TREE_PROXY_GRAPH=0 \
SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0 \
  python -O bench/bench.py <공통 인자> \
  --duet_p1_tree_policy off --duet_p2_tree_policy off

# P1/P2 동적 tree
SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 SSD_TREE_EXEC_WARMUP=all \
SSD_CHAIN_PROXY_GRAPH=1 SSD_TREE_PROXY_GRAPH=1 \
SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0 \
  python -O bench/bench.py <공통 인자> \
  --duet_p1_tree_policy on --duet_p2_tree_policy on \
  --duet_p1_roots_per_position 2 \
  --duet_p1_tree_max_nodes 18 --duet_p2_tree_max_nodes 8 \
  --duet_tree_c_tensor 3
```

### 12.3 config 전체 의미와 제약

#### 공통 DUET 인자

| 인자 | 권장/예시 | 의미와 주의점 |
|---|---:|---|
| `--duet` | 켬 | early-exit DUET 경로 활성화 |
| `--spec --async` | 켬 | speculative + async draft 활성화 |
| `--gpus` | 5 | target TP 4 + draft 1 구성의 예시 |
| `--b` | 1 | 현재 production tree는 B=1만 지원 |
| `--temp` | 0.7 | target temperature; tree는 0보다 커야 함 |
| `--dtemp` | 생략 또는 >0 | 별도 draft temperature; 생략 시 target 값 사용 |
| `--x` | 설정별 값 | sampler-x; 사용하면 proxy와 verifier가 같은 값을 사용 |
| `--duet_exit_layer` | 56 | target early-exit layer; 바꾸면 threshold/K calibration 재실행 |
| `--duet_k1` | 9 | P1 draft round 수 |
| `--duet_k2` | 4 | P2 round 수 F; K2≤K1 |
| `--k` | 생략 또는 13 | 반드시 K1+K2; 생략 시 자동 유도 |
| `--f` | 3 | 전체 async fanout; tree C와 다른 값 |
| `--duet_p1_fanout` | 2 | P1 기본 fanout |
| `--duet_p1_fanout_list` | 10개 | 길이는 K1+1이어야 함 |
| `--duet_p2_budget` | 10 | P2 물리 forward 폭 W의 직접 설정 |
| `--duet_proxy_top_k` | 보통 생략 | wire 요구량보다 작으면 Config가 안전 하한으로 올림 |
| `--duet_no_jit_short` | 보통 끔 | 주면 miss의 K2-depth JIT-short를 비활성화 |

#### tree 인자

| 인자 | 기본값 | 역할 |
|---|---:|---|
| `--duet_p1_tree_policy` | `off` | P1 chain/dynamic 선택 |
| `--duet_p2_tree_policy` | `on` | P2 chain/dynamic 선택 |
| `--duet_p1_roots_per_position` | 2 | P1 context별 균등 시작 후보 수 U1 |
| `--duet_p1_tree_forward_scale` | 1.0 | P1 W/R; 1에서도 round 1부터 전역 동적 선택 |
| `--duet_p1_tree_max_nodes` | 18 | P1 root별 최대 응답 node N1 |
| `--duet_p2_tree_max_nodes` | 8 | P2 root별 최대 응답 node N2 |
| `--duet_tree_root_count` | `None` | P2 R; 동적 P2는 기본 R=W |
| `--duet_tree_c_tensor` | 3 | 부모별 ordered 비복원 자식 상한 C, 허용 1--8 |
| `--duet_tree_proxy_threshold` | 0.01 | P2 round 1 이후 root 확장 threshold |
| `--duet_tree_conf_threshold` | 0.03 | P1/P2 round 1 이후 child 확장 threshold |
| `--duet_tree_fanout_policy` | `backbone` | 과거 정책 재현용; production dynamic은 전역 fanout 사용 |
| `--duet_tree_beta` | 0.5 | 과거 root-budget 재현용; 현재 동적 점수에는 사용 안 함 |

Config는 다음을 시작 전에 거부한다.

- `K1+K2 != k`, `K2>K1`
- `R>W`
- P1/P2 max nodes가 1 미만이거나 model length 이상인 경우
- 명시적으로 과거 backbone P2를 재현할 때 `N2<K2` 또는 `N2>K2*C`
- vocabulary>32768인 packed P_iv wire
- tree와 `SSD_DUET_PROXY_ON_DRAFT=1` 또는
  `SSD_DUET_EXIT_TOPM_GATHER=1`의 조합

#### runtime 환경 변수

| 환경 변수 | 성능 실행값 | 의미 |
|---|---:|---|
| `SSD_TREE_EXEC` | 1 | P2 전체 CUDA Graph 실행기; P1 `on`은 같은 실행기를 항상 사용 |
| `SSD_TREE_ARENA` | 1 | executor 사전-분류 fallback/참조 경로 허용 |
| `SSD_TREE_EXEC_WARMUP` | `all` | 가능한 P1/P2 context/page graph를 요청 전 전부 capture |
| `SSD_TREE_EXEC_WORKSPACE_MB` | 64(기본) | executor FlashInfer workspace 기준 크기 |
| `SSD_TREE_PROXY_GRAPH` | 1 | target tree proxy 계산 CUDA Graph |
| `SSD_CHAIN_PROXY_GRAPH` | 1 | chain 비교군의 proxy 계산 CUDA Graph |
| `SSD_DUET_EXIT_REPLICA` | 1 | rank0의 local exit 계산 경로 사용 |
| `SSD_ASYNC_PROXY_SEND` | 1 | persistent buffer 기반 non-blocking proxy send |
| `SSD_PROXY_STREAM` | 0 | 별도 proxy stream을 사용하지 않는 현재 기준 |
| `SSD_DIST_PORT` | run별 고유값 | 동시 실행끼리 process-group port 충돌 방지 |

`SSD_TREE_EXEC_WARMUP=all`은 steady-state 측정에서 첫 hit compile/capture를 없애지만
P2만 켠 과거 실모델에서는 약 7--9초/1014MiB였고, P1까지 켠 현재 형상에서는
P1이 약 20--28초/2.8GiB를 더 예약했다. 짧은 end-to-end wall time을 보고할 때는
이 비용을 별도로 적는다. 특정 bucket만 진단할 때는 `2,3`처럼
쉼표 목록을 줄 수 있으나 최종 성능 gate에는 `all`을 사용한다.

#### profiler와 진단 설정

성능 수치에는 다음을 사용한다.

```bash
export SSD_PROFILE=0
export SSD_PROFILE_DUET=0        # 최종 TPS/AL gate
export SSD_PROFILE_DUET_DETAIL=0
```

timeline 한 번을 만들 때만 다음으로 바꾼다.

```bash
export SSD_PROFILE_DUET=1
export SSD_PROFILE_DUET_DETAIL=0
export SSD_PROFILE_DUET_MAX_EVENTS=12000
export SSD_PROFILE_DIR=/path/to/profile_dir
```

`SSD_TREE_STAGE1`, `SSD_TREE_STAGE2`, `SSD_TREE_TOPO_TRACE`,
`SSD_TREE_NODE_AUDIT`, `SSD_TREE_LAB*`, `SSD_TREE_EXEC_ALT`,
`SSD_TREE_EXEC_EAGER_DIAG`, `SSD_TREE_EXEC_SYNC_DIAG`, `SSD_TREE_GAP_PROF`,
`SSD_CG_INPUT_CHECK`, `SSD_TREE_EXEC_DELAY_MS`는 정확성/원인 진단 전용이다. D2H,
파일 I/O, 교차 실행 또는 의도적 지연을 추가하므로 TPS, hit, AL 최종값을 잴 때는
모두 unset한다.

### 12.4 결과 기록 규약

실험 결과에는 commit, 서버, GPU, dataset, prompt/output 수, seed, 정책, R/W/F/C/Nv,
temperature, profiler 유무와 raw 분모를 함께 남긴다. 같은 seed의 반복 cycle을
독립 표본으로 세지 않는다.

---

## 13. 현재 지원 범위와 남은 연구 문제

### 13.1 지원되는 주 경로

- B=1
- target/draft temperature > 0
- Llama 계열 draft와 production P1/P2 executor
- P2 R≤W, 현재 기본 R=W=10; P1은 실제 roots≤captured width
- multiword 63-bit ancestry (`F*W>63` 지원)
- vocabulary≤32768인 P_iv token pack
- phase별 `on`이 동일한 최적화 CUDA Graph 실행기 사용

### 13.2 B>1

B=1 gate만 지워서는 안 된다. sequence별 root/topology, page/slot/context,
verify mask/row bucket, key와 commit path를 `[B,...]` 고정 buffer로 분리하고,
부모관계를 block-diagonal mask로 만들어야 한다. B=2 exact parity 후 B=4/8로
확장한다. 그 전에는 chain fallback이 안전하다.

### 13.3 동적 정책의 다음 품질 검증

production은 round 0 root coverage만 보장하고 이후 부모를 전역 선택한다. 다음
과제는 고정-backbone으로 돌아가는 것이 아니라 이 동적 정책의 점수와 threshold를
올바르게 검증하는 것이다.

1. 같은 입력/noise에서 P2 `dynamic`이 P1 도입 전 전역 selector와 exact인지 유지
2. P1 root prior의 context-reach 계산과 P2 `P_iv`가 실제 사후 hit/acceptance에
   얼마나 calibration되는지 측정
3. proxy/confidence threshold는 sampled leaf나 root key를 삭제하지 않고 이후
   forward 배분만 줄이는지 확인
4. 실제 topology를 root별 valid 수, 최대 깊이, 선택 parent와 accepted path로 기록

판정은 conditional AL 하나가 아니라 P1/P2 hit, phase contribution,
tok/step과 wall TPS를 함께 사용한다.

### 13.4 남은 시간 비용

- tree graph의 동적 선택과 ordered sampling이 chain보다 약 2.5--3ms 더 든다.
- target은 최대 8-node tree를 실제 transformer로 검증하므로 chain 4-node보다
  model row 비용이 크다.
- P1/P2 all-page warmup의 cold-start와 약 3.8GiB 추가 예약 비용이 있다.
- target mask 준비 0.41ms와 일부 output gather는 더 줄일 수 있지만 예상 폭은
  sub-ms다. 품질 정책을 바꾸지 않는 kernel-level 최적화만 허용한다.

### 13.5 P1 tree

P1 tree의 코드·CUDA Graph·multiword ancestry·공통 wire 배선은 완료됐고 P2와
같은 global dynamic selector를 사용한다. 과거 전역 선택 formal과 이후 고정
backbone sweep은 현재 정책의 성능 근거로 재사용하지 않는다. 다음 성능 실험은
P1만 on, P2만 on, 둘 다 on의 세 분해 arm을 충분한 prompt/길이와 순서 회전으로
다시 비교해야 한다.

---

## 14. 논문 작성 시 주장할 수 있는 것과 없는 것

현재 증거로 주장할 수 있는 내용:

- 고정 CUDA Graph shape 안에서도 token과 선택 부모가 replay마다 달라지는 tree를
  구현할 수 있다. 기본 `R=W`에서도 round 1부터 parent topology가 동적이다.
- P1/P2의 phase 내부 forward 사이 host 개입을 제거하면서 temperature>0 ordered
  residual sampling과 lossless target verification을 유지했다.
- sibling branch가 실제 accepted path에 쓰이며 특정 gate에서 P2AL/P2 기여를
  높였다.
- root coverage를 버리고 조건부 AL만 높이는 설계가 총 기여를 악화시킬 수 있다.
- post-hoc utility label로 정한 expansion threshold가 두 seed에서 tail 확장을
  줄이고 P2AL/tok-per-step을 개선했다.

아직 주장하면 안 되는 내용:

- 현재 global dynamic 정책이 모든 dataset에서 chain보다 빠르다.
- 두 seed threshold 결과가 일반적인 optimal threshold다.
- 짧은 smoke의 TPS/AL이 논문 최종 성능이다.
- all-page warmup과 상주 메모리 비용이 0이다.
- B>1 또는 greedy temperature=0 tree가 지원된다.

최종 논문 표에는 최소 세 독립 seed, 충분한 prompt/output 길이, chain/동적
순서 회전, P1/P2 hit와 conditional AL, phase contribution, tok/step, TPS,
target/draft p50 및 startup memory를 함께 보고한다.

---

## 15. 코드와 산출물 위치

| 내용 | 위치 |
|---|---|
| config와 validation | `ssd/config.py` |
| CLI | `bench/bench.py` |
| 점수, sampling, arena, mask, verifier helper | `ssd/engine/helpers/p2_tree.py` |
| P1 roots/context/executor specialization | `ssd/engine/helpers/p1_tree.py` |
| production full-P1/P2 graph | `ssd/engine/helpers/p2_tree_executor.py` |
| P1/P2 dispatch/cache/view | `ssd/engine/draft_runner.py` |
| target residual walk | `ssd/engine/verifier.py` |
| target tree rows/KV | `ssd/engine/model_runner.py` |
| attention/KV 안전 guard | `ssd/layers/attention.py` |
| timeline plot | `bench/plot_duet_aligned_timeline.py` |
| threshold calibration | `tools/duet_calibration/` |
| K1/K2 calibration | `tools/duet_calibration/analyze_k_balance.py` 및 tree_sweep 실행 script |
| 실험 실행 script | `experiments/proxy_async_overlap/tree_sweep/` |
| 과거 설계/정정 이력 | `docs/duet/internal/` |

대표 raw 결과 디렉터리는 `coverage_final_gate_20260807`,
`eagle_global_20260807_v2`, `threshold_gate_*_20260807`,
`kbalance_eagle_20260807`, `runtime_opt_20260807`이다. raw profiler JSON은 크기가
크므로 Git 문서의 결과 표와 실행 script를 재현 기준으로 삼고, 서버 보관본은
논문 artifact를 만들 때 별도 압축/manifest/checksum으로 고정한다.

---

## 16. 변경 이력 요약

1. offline feasibility에서 sibling branch의 AL 가능성을 확인했다.
2. 초기 confidence tree가 상위 R=6 root만 보존해 hit를 잃는 문제를 발견했다.
3. Python으로 매 forward topology를 갱신해 생긴 GPU idle을 계측했다.
4. 고정-shape GPU arena로 의미를 옮겼으나 작은 kernel과 숨은 sync 때문에 처음엔
   더 느렸다.
5. 전체 P2를 raw forward까지 CUDA Graph로 캡처하고 CPU gap을 제거했다.
6. page/slot/mask/RNG/recording 버그와 측정 경계 오류를 단계별로 수정했다.
7. coverage 정책으로 root 보존 시 P2 기여 증가를 확인했다.
8. P2 전역 경로 점수 정책을 구현했다.
9. 사후 calibration으로 낮은 확률 leaf의 확장만 멈추는 threshold를 도입했다.
10. tree update kernel과 target verify 준비를 최적화했다.
11. 전역 confidence 정책의 초기 formal에서 큰 하락을 관측했으나, 이후 수정된
    구현 버그가 섞여 정책 반례로 사용할 수 없음을 확인했다.
12. P1 동적 tree, multiword ancestry, phase별 공통 wire와 full-P1 CUDA Graph를
    추가하고 공개 CLI를 P1/P2 `off|on`으로 정리했다.
13. public `on`을 한때 backbone 정책으로 후퇴시켰으나 이 판정을 철회했다.
14. 실모델 초기화에서 Config clone, P2 graph dispatch, all-page warmup의 새 정책
    누락을 찾아 수정하고 P1/P2 모두 runtime replay만 수행함을 확인했다.
15. commit `b8e8bfd`에서 public `on`을 `dynamic`에 연결하고 P2를 P1 도입 전
    전역 알고리즘으로 복원했으며, P1도 root prior만 다르게 같은 알고리즘을 쓴다.
