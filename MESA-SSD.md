# DUET-SSD: Early-Exit Guided Asynchronous Speculative Decoding

> 이 파일의 이름은 초기 연구명인 `MESA-SSD.md`로 남아 있지만, 현재 기법의
> 이름은 **DUET**이다. 이 문서는 새 실험에서 사용할 chain 및 동적 tree 기법과
> 최적화 실행 설정을 설명한다. tree 구현의 세부 불변조건과 과거 실험 이력은
> [`ssd/docs/duet/TREE_IMPLEMENTATION.md`](ssd/docs/duet/TREE_IMPLEMENTATION.md)를
> 참고한다.

## 1. 핵심 목표

DUET는 target과 draft를 서로 다른 GPU에서 비동기로 실행하면서, target의
early-exit 정보를 다음 draft cache를 준비하는 데 사용한다.

일반 speculative decoding에서 draft token의 수락확률을 높이기 위해 sampling
분포 자체를 바꾸면 target 분포를 보존하기 어렵거나 acceptance와 cache coverage가
서로 충돌할 수 있다. DUET는 두 역할을 분리한다.

- draft token은 기존 speculative decoding의 proposal 분포로 생성한다.
- target early-exit는 다음에 필요할 cache root의 위치와 token을 예측한다.
- P1은 proxy가 오기 전의 시간을 사용하고, P2는 proxy가 도착한 뒤 target과
  남은 시간을 겹쳐 사용한다.
- cache miss에서는 기존 lossless speculative decoding 경로로 복구한다.
- P1과 P2는 서로 독립적으로 chain 또는 동적 tree continuation을 만들 수 있다.

공개 설정은 두 단계 모두 `off|on`만 사용한다. 현재 코드 기본값은 P1 `off`,
P2 `on`이지만, 새 실험은 비교 대상을 명확히 하기 위해 두 값을 항상 명시한다.
기존 chain은 `--duet_p1_tree_policy off --duet_p2_tree_policy off`로 완전히
유지된다.

---

## 2. 전체 비동기 흐름

```text
target GPU 0--3
  target 앞부분 -> early-exit logits -> proxy 후보 계산/전송 -> target 뒷부분
        |                                      |
        |                                      v
        |                               다음 cache root
        v
draft GPU 4
  P1 chain/tree 생성 ---- proxy 대기 -------- P2 chain/tree 생성
        |                                      |
        +---------------- cache ---------------+
                                               |
다음 step의 실제 recovery context -------------+--> exact key hit/miss
                                                      |
                                                      v
                                                target verification
```

한 step에서의 순서는 다음과 같다.

1. draft는 target proxy를 기다리지 않고 P1을 먼저 수행한다.
2. target은 현재 verify 도중 early-exit logits를 계산한다.
3. target은 draft proposal과 early-exit 분포를 결합해 다음 cache root 후보를
   계산하고 draft GPU로 보낸다.
4. draft는 P2에서 root마다 chain 또는 tree continuation을 만든다.
5. 다음 step의 실제 `(sequence, terminal context, recovery token)`과 cache key가
   정확히 같으면 P1/P2 hit가 된다.
6. target은 hit한 continuation을 검증하고 수락된 경로의 KV만 commit한다.

cache hit에는 timeout이나 “너무 일찍 만들면 무효” 같은 조건이 없다. hit은
정확한 key를 미리 만들었는지의 문제다.

---

## 3. P1: proxy 도착 전 draft

P1은 target proxy 없이 draft 분포만 사용한다. 정책은 두 가지다.

- `--duet_p1_tree_policy off`: 기존 위치별 fanout chain을 그대로 사용한다.
- `--duet_p1_tree_policy on`: 각 현재 context에서 같은 수의 시작 후보를 만들고,
  첫 forward에서 모두 평가한 뒤 누적 점수가 높은 자식을 전역적으로 확장한다.

P1만 `on`으로 두는 분해 실험도 같은 동적 selector를 사용한다. 내부 정책은 P2
스위치가 아니라 P1/P2 중 하나라도 `on`이면 동적으로 정규화되고, 둘 다 `off`일
때만 chain이다.

동적 P1의 시작 단계는 다음과 같다.

1. 일반 chain 응답이면 `K1+1` 또는 `K2+1`개의 현재 context를 사용한다. 이전
   tree가 hit했다면 recovery와 그 tree node가 context가 된다.
2. 각 context에서 `duet_p1_roots_per_position`개 후보를 만든다. 이미 반환된 다음
   token은 coverage 중복을 피하기 위해 후보에서 제외하지만, 선택한 후보의 점수는
   제외 후 재정규화 확률이 아니라 원래 draft 분포 `q(x|context)`다.
3. P1 root의 초기 점수는 `해당 context까지 도달할 확률 × q(root|context)`다.
   첫 P1 forward에서는 모든 실제 root를 평가한다. 이후 `K1-1`번은 앞 round에서
   생성한 모든 자식 중 이 초기 점수와 경로 confidence의 누적 곱이 높은 W개를
   다음 부모로 선택한다. P2와 같은 전역 동적 선택이며 proxy score만 없다.
4. root 하나가 보낼 수 있는 node 수는 `duet_p1_tree_max_nodes`로 제한한다.
   이 값은 기본 18이며 항상 채우는 수가 아니라 root별 고정 응답 상한이다.
   순차 draft round `K1=9`와 응답 node 수는 독립적이고, 모든 root에 깊이 9를
   강제하지 않는다.

기본 동적 P1 예시는 `K1=9`, position당 root 2개, root당 최대 node 18개다.
일반 10-context step의 실제 root는 20개이며, P1 forward 9번 전체와 그 사이의
선택·sampling·mask 갱신은 하나의 CUDA Graph replay로 실행된다.

기본 P1 forward 폭은 root 수와 같은 W=R=20이다. 첫 forward 뒤에는 최대
`R×C=60`개 자식이 생기므로, 추가 폭 없이도 그중 상위 20개를 다음 부모로 골라
동적 topology를 만들 수 있다. `root 20개`, `root당 최대 node 18개`,
`target verify 최대 19행`은 각각 시작 후보 수, 응답 용량, recovery 포함 검증
입력으로 서로 다른 수치다.

P1의 주 목적은 proxy를 기다리는 시간을 실제 draft 계산으로 채우는 것이다.
P1이 너무 짧으면 draft가 proxy를 기다리고, 너무 길면 proxy가 도착한 뒤에도
P1이 끝나지 않아 P2 시작이 늦어진다. 따라서 K1은 acceptance만 보고 정하지
않고 `P1 완료 시각 - proxy 도착 시각`이 0에 가까운 값으로 정한다.

현재 P1 dynamic은 기능 구현과 CUDA Graph 실행은 완료됐지만 production
champion은 아니다. commit `0dc9e9d`의 3-seed P1-only gate에서 P1AL은 평균
`+1.3%`였지만 P1 hit와 tok/step이 각각 `-1.53%p`, `-0.8%`였고 TPS는
68.4→43.2로 내려갔다. 일반 context의 20-vs-16 draft lane, tree-hit 뒤 최대
38 lane, target의 최대 18-node 검증이 실제 모델 계산이기 때문이다. 따라서 새
실험의 안전한 기준선은 P1 `off`이며, P1 `on`은 정책 연구 arm으로 취급한다.

---

## 4. Target early-exit proxy

### 4.1 조건부 수락확률

draft가 위치 `i`에서 제안한 token을 `y_i`, draft 분포를 `p_i^D`, target의
early-exit 분포를 `p_i^E`라 하면 다음 proxy를 사용한다.

```text
alpha_i = min(1, p_i^E(y_i) / p_i^D(y_i))
```

첫 기각 위치의 확률은 다음과 같다.

```text
h_0 = 1 - alpha_0
h_i = product(alpha_j, j<i) * (1-alpha_i)
h_K = product(alpha_j, j<K)       # 모두 수락되는 context
```

### 4.2 correction token 분포

위치 `i`에서 draft token이 기각됐다고 가정한 correction 분포는

```text
r_i(v) = normalize([p_i^E(v) - p_i^D(v)]_+)
```

를 사용한다. 모두 수락된 마지막 context에서는 해당 context의 early-exit target
분포를 사용한다.

최종 root 후보의 질량은

```text
P_iv(i,v) = h_i * r_i(v)
```

이다. target은 모든 `(position, token)`을 펼쳐 상위 후보를 전송한다. draft는
P1과 중복되는 후보를 제거한 뒤 상위 R개 root를 사용한다. 중복 제거 전에 여유
후보를 보내는 이유는 wire 후보를 R개로 딱 자르면 P1 중복 후 실제 root 수가
R보다 작아질 수 있기 때문이다.

### 4.3 tree를 검증한 뒤의 proxy

형제 tree에서는 각 형제의 수락 사건이 독립이 아니다. 앞 형제가 기각되면 target
residual과 draft proposal을 갱신한 뒤 다음 형제를 시도한다.

```text
a_j = min(1, R_j[x_j] / D_j[x_j])
reject:
  R_{j+1} = normalize((R_j-D_j)_+)
  D_{j+1} = normalize(D_j with x_j removed)
```

현재 구현은 이 sibling residual ladder를 target proxy 계산과 실제 verifier에서
같이 사용한다. 앞 형제의 기각확률을 빼면 뒤 형제의 도달 질량을 과대평가하므로
단순 독립 확률로 바꾸면 안 된다.

---

## 5. P2 chain 정책

`--duet_p2_tree_policy off`가 기존 P2 chain 경로다.

1. proxy `P_iv` 상위 W개 root를 선택한다.
2. 첫 P2 forward에서 W개 root를 동시에 평가한다.
3. 각 root에서 하나의 sampled child를 따라 K2 round까지 continuation을 만든다.
4. root와 continuation을 cache에 저장한다.
5. hit하면 target은 recovery와 K2 chain을 순차적으로 검증한다.

기본 실험 설정은 `W=10`, `K2=4`다. 이때 model workload는 네 번의 W-wide
draft forward다.

전체 chain 기준선은 P1/P2를 모두 `off`로 둔다. P1만 또는 P2만 tree로 켜는
분해 실험도 가능하므로, 한 개의 공용 tree switch로 두 단계를 묶지 않는다.

---

## 6. P2 tree 정책

P2 tree는 `--duet_p2_tree_policy on`으로 켠다. DUET의 draft model과
temperature>0 residual verifier를 유지하면서, 첫 forward에서 모든 proxy root를
평가하고 이후 forward의 부모를 누적 proxy×confidence 점수로 전역 선택한다.
외부 기법 이름은 공개 정책 이름으로 사용하지 않는다.

### 6.1 기호

| 기호 | 기본값 | 의미 |
|---|---:|---|
| R | 10 | proxy가 선택한 실제 root 수 |
| W | 10 | P2 forward 한 번의 부모 lane 수 |
| F=K2 | 4 | P2 forward round 수 |
| C | 3 | 한 부모에서 동시에 뽑는 형제 수 상한 |
| N2 | 8 | root 하나가 target에 보내는 P2 node 수 상한 |

기본은 `R=W`다. 첫 P2 forward에서 모든 root를 실제로 평가해야 하므로 현재
구현은 `R>W`를 거부한다.

### 6.2 경로 점수

root `r` 아래 경로가 token `x_1,...,x_d`를 생성했다면 다음 점수를 사용한다.

```text
score(r,x_1...x_d)
  = P_proxy(r)
    * q(x_1|r)
    * q(x_2|r,x_1)
    * ...
    * q(x_d|r,x_1,...,x_{d-1})
```

실제 구현은 underflow를 피하기 위해 로그 공간에서

```text
log_score = log P_proxy(root) + sum(log q(child|parent))
```

를 계산한다. 첫 round 뒤 생성된 모든 자식이 이 점수로 경쟁하며 기본 `R=W=10`
에서도 상위 W개만 다음 forward 부모가 된다. beta, proxy 제곱근, depth bonus는
넣지 않는다.

### 6.3 round별 동작

각 P2 round는 다음 순서다.

1. 첫 round에서는 R개 root를 모두 평가한다.
2. 이후 round에서는 직전 depth에서 생성된 모든 미확장 자식을 누적 점수로
   정렬해 상위 W개를 선택한다. root별 의무 깊이는 없다.
3. token을 뽑기 전에 선택 부모의 점수와 root별 남은 N2 용량으로 fanout을
   결정한다. 부모당 상한은 C다.
4. W개 부모를 한 번의 draft forward로 평가한다.
5. ordered 비복원 자식을 arena와 root-local `[R,N2]` view에 기록한다.

CUDA Graph의 shape는 항상 `F×W`로 고정이다. token과 확률은 매 replay 달라지고,
부모 index, rope와 attention mask도 device 값으로 달라진다. 기본
`R=W,F=4,C=3,N2=8`에서도 round 1부터 최대 30개 자식 중 10개를 고르므로 parent
layout과 root별 깊이/node 수가 동적이다.

### 6.4 ordered sampling without replacement

한 부모에서 C개 자식을 뽑을 때 같은 token을 중복 추출하지 않는다. 각 vocabulary
token에 `E_v ~ Exp(1)`을 만들고

```text
q_v / E_v
```

상위 C개를 형제 순서로 사용한다. 이는 q에서 순차적으로 비복원 추출한 순서와
같다. 선택된 token의 원래 확률 `raw_q`를 누적 경로 점수에 사용한다.

형제의 수와 순서는 sampling 전에 확정되며 target verifier까지 그대로 유지된다.
sampling 결과를 본 다음 마음에 드는 자식만 남기면 proposal 분포가 바뀌므로
허용하지 않는다.

P1 도입 직후의 전역 선택 formal에서 큰 AL 하락이 관측됐지만, 그 실행은 이후
수정된 graph 입력·page·plan/workspace 버그가 남아 있던 코드였다. 따라서 해당
수치를 정책 반례로 사용하지 않는다. 현재 P2는 P1 도입 전 전역 알고리즘을
복원했고 P1도 같은 코드를 사용한다.

### 6.5 expansion threshold

기본 threshold는 다음과 같다.

```text
root/start threshold = 0.01
node confidence threshold = 0.03
```

threshold는 root나 이미 sampled된 node를 삭제하지 않는다.

- 모든 root는 첫 forward에서 평가한다.
- 낮은 confidence token도 target이 검증할 leaf로 남긴다.
- 단지 그 leaf 아래에 다음 draft forward를 더 사용하지 않는다.

이 값은 현재 모델/exit layer/dataset의 calibration 결과이므로 설정이 바뀌면 다시
calibration해야 한다.

### 6.6 target의 lossless tree verification

같은 부모의 형제들을 순서대로 보며 target 분포 `R`과 draft 분포 `D`로

```text
accept probability = min(1, R[token] / D[token])
```

를 계산한다. 수락하면 그 자식으로 내려가고, 기각하면 sibling residual ladder로
R과 D를 갱신한다. 모든 형제가 기각되면 최종 residual에서 recovery token을
뽑는다. leaf까지 수락하면 leaf target 분포에서 bonus token을 뽑는다.

target은 tree node를 scratch KV에 계산한 뒤 수락된 한 경로의 KV만 canonical
위치에 commit한다. 다른 branch의 KV는 다음 autoregressive context에 들어가지
않는다.

---

## 7. 현재 반영된 실행 최적화

### 7.1 chain과 tree에 공통으로 적용할 최적화

새 chain/tree 비교에서는 다음을 양쪽에 똑같이 사용한다.

- K1/K2 길이별 target/draft CUDA Graph
- target의 K1/K2 chain proxy CUDA Graph
- rank 0 local early-exit replica: `SSD_DUET_EXIT_REPLICA=1`
- persistent buffer 기반 비동기 proxy send: `SSD_ASYNC_PROXY_SEND=1`
- 별도 proxy stream은 끔: `SSD_PROXY_STREAM=0`
- miss에서 K2 깊이만 만드는 JIT-short
- verifier softmax/multinomial startup warmup
- AWQ target TP4 + AWQ draft 1GPU 구성

shared optimization을 tree arm에만 켜면 tree topology의 효과와 통신/target 개선이
섞인다. 현재 canonical 비교 스크립트는 위 설정을 두 arm 모두에 적용한다.

### 7.2 chain 전용 실행 설정

chain은 tree graph, tree arena와 tree page warmup을 사용하지 않는다.

```bash
SSD_TREE_EXEC=0
SSD_TREE_ARENA=0
SSD_TREE_PROXY_GRAPH=0
SSD_TREE_EXEC_WARMUP=0
--duet_p1_tree_policy off
--duet_p2_tree_policy off
```

chain proxy graph는 공통 최적화이므로 `SSD_CHAIN_PROXY_GRAPH=1`을 유지한다.

### 7.3 동적 tree 전용 최적화

#### 전체 P1/P2 CUDA Graph

각 phase의 모든 forward와 그 사이의 작업을 phase별 graph 하나로 캡처한다.

```text
[부모 선택 -> fanout -> input/rope/mask -> draft forward
 -> logits -> ordered sampling -> node 삽입] × K1 또는 K2
```

따라서 forward 사이 Python, GPU→CPU readback, runtime attention plan과 tensor
allocation이 없다. timeline에서 P1/P2가 각각 막대 하나로 보이지만 내부에는
각각 9번/4번의 실제 draft forward가 들어 있다.

#### 고정 GPU arena와 융합 kernel

root/node/token/parent/depth/score/조상 mask를 고정 주소 tensor에 저장한다.
arena reset, child 삽입, root-local view 기록과 metadata 구성을 작은 Triton
kernel로 융합했다. boolean indexing의 숨은 synchronization과 중복 scatter를
사용하지 않는다.

Tree hit에서는 chain용 전체-vocabulary backbone logits를 만들거나 cache에서
복사하지 않는다. 실제 tree node의 부모 분포만 executor의 고정 buffer에서
모아 보내며, phase 상한까지 padding하지 않고 실제 valid node 행만 전송한다.
staging tensor도 매 hit마다 새로 할당하지 않고 시작할 때 한 번 만든다. 현재
tree payload는 B=1 계약이므로, B>1 요청에서 이전 B=1 tree row가 우연히 key와
일치하면 chain payload로 오독하지 않고 miss로 처리한다.
Topology readback과 valid-node 수 readback도 한 번으로 합쳤고, parent 분포는
임시 tensor 없이 고정 staging buffer로 직접 gather한다.

#### page bucket 사전 준비

FlashInfer plan과 CUDA Graph를 가능한 page bucket별로 요청 전에 만든다.

```bash
SSD_TREE_EXEC_WARMUP=all
```

이 warmup은 steady-state decode 시간에는 들어가지 않는다. P1을 함께 켜면
context 폭 5/10/19와 모든 page bucket을 추가로 준비한다. 최신 실모델 기능
smoke에서는 P2 약 1.0GiB, P1 약 2.2GiB의 추가 예약과 P1 약 32초의 시작 비용이
관측됐다. P1 page/context graph는 서로 동시에 실행되지 않으므로 transient
capture pool을 공유하고, 같은 page/shape의 round들은 동일한 FlashInfer plan
workspace도 공유한다. page shape가 다른 graph의 plan 상태는 계속 분리한다.

#### target tree verify 준비 최적화

- runtime FlashInfer `plan()`을 제거하고 graph가 읽는 page/length buffer만 갱신
- dense bool mask를 만든 뒤 pack하지 않고 packed attention mask에 직접 기록
- page id는 항상 유효한 물리 page를 사용하고 mask=0으로 canvas를 격리
- 음수 position/slot이 rotary와 KV store에 들어가지 않도록 guard

#### tree proxy CUDA Graph

target의 sibling residual ladder, terminal mass, candidate top-k와 P_iv pack을
valid-node width별 고정 graph로 실행한다.

동적 tree의 production 설정은 다음이다.

```bash
SSD_TREE_EXEC=1
SSD_TREE_ARENA=1
SSD_TREE_PROXY_GRAPH=1
SSD_TREE_EXEC_WARMUP=all
--duet_p1_tree_policy on
--duet_p2_tree_policy on
```

`SSD_TREE_ARENA=1`은 executor가 사전에 지원하지 않는 shape를 분류했을 때 참조
fallback을 허용한다. 정상 기본 shape의 최종 결과에서는 executor replay가
사용되고 fallback은 0이어야 한다.

---

## 8. 새 chain/tree 실험 실행

### 8.1 서버 준비

```bash
git fetch origin
git switch feat/duet-p2tree-g0
git pull --ff-only

cd ssd
```

다음 경로가 실험 서버에 존재해야 한다.

```text
target model: /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
target AWQ:   /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
draft model:  /data2/chokwans99/awq_calibrated/tinyllama_1b
draft AWQ:    /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
dataset:      /data2/chokwans99/datasets
Python:       /home/chokwans99/anaconda3/envs/ssd/bin/python
```

경로가 다르면 `MODEL_PATH`, `TARGET_AWQ`, `DRAFT_PATH`, `DRAFT_AWQ`, `PY`,
`SSD_DATASET_DIR` 환경 변수로 덮어쓴다.

### 8.2 먼저 짧은 smoke

성능 trace를 만들지 않는 production-shape smoke부터 실행한다.

```bash
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/new_duet_gate \
RUN_SCOPE=smoke \
SMOKE_AUDIT=0 \
bash experiments/proxy_async_overlap/tree_sweep/run_eagle_global_gate_20260807.sh
```

다음을 확인한다.

- `EXIT:0`
- `p2exec replay > 0`
- executor `fallback=0`, `error=0`
- P1/P2 hit와 accepted length가 finite
- 종료 후 draft/target GPU 프로세스가 남지 않음

### 8.3 전체 paired gate

새 output 폴더를 사용해 세 seed chain/tree 순서 회전 실험을 수행한다.

```bash
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/new_duet_gate \
SMOKE_AUDIT=0 \
bash experiments/proxy_async_overlap/tree_sweep/run_eagle_global_gate_20260807.sh
```

스크립트는 tree arm에서 P1/P2를 모두 `on`으로 명시한다. 파일명에는 과거
실험명이 남아 있지만 공개 정책 값에는 사용하지 않는다. 실행 순서는 다음과 같다.

1. 짧은 tree smoke
2. seed 42: chain -> tree
3. seed 123: tree -> chain
4. seed 2024: chain -> tree
5. 모든 품질/성능 run 이후 chain/tree timeline 한 번씩

최종 비교 run에는 profiler를 켜지 않는다. timeline run만
`SSD_PROFILE_DUET=1`, event cap 12000을 사용한다.

### 8.4 arm별 수동 실행 예시

공통 환경:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_CHAIN_PROXY_GRAPH=1
export SSD_DUET_EXIT_REPLICA=1
export SSD_ASYNC_PROXY_SEND=1
export SSD_PROXY_STREAM=0
export SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0
unset SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER
unset SSD_TREE_STAGE1 SSD_TREE_STAGE2 SSD_TREE_TOPO_TRACE
unset SSD_TREE_NODE_AUDIT SSD_TREE_EXEC_DELAY_MS SSD_TREE_GAP_PROF
```

공통 CLI:

```bash
COMMON=(
  --llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --b 1 --async --spec --duet
  --temp 0.7 --input_len 512 --output_len 384 --numseqs 20
  --all --max_model_len 2048 --seed 42
  --duet_exit_layer 56 --f 3
  --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10
)
```

전체 chain:

```bash
SSD_DIST_PORT=16211 \
SSD_TREE_EXEC=0 SSD_TREE_ARENA=0 \
SSD_TREE_PROXY_GRAPH=0 SSD_TREE_EXEC_WARMUP=0 \
/home/chokwans99/anaconda3/envs/ssd/bin/python -O bench/bench.py \
  "${COMMON[@]}" \
  --duet_p1_tree_policy off --duet_p2_tree_policy off
```

P1+P2 동적 tree:

```bash
SSD_DIST_PORT=16212 \
SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 \
SSD_TREE_PROXY_GRAPH=1 SSD_TREE_EXEC_WARMUP=all \
/home/chokwans99/anaconda3/envs/ssd/bin/python -O bench/bench.py \
  "${COMMON[@]}" \
  --duet_p1_tree_policy on --duet_p2_tree_policy on \
  --duet_p1_roots_per_position 2 \
  --duet_p1_tree_forward_scale 1.0 \
  --duet_p1_tree_max_nodes 18 --duet_p2_tree_max_nodes 8 \
  --duet_tree_root_count 10 --duet_tree_c_tensor 3 \
  --duet_tree_proxy_threshold 0.01 \
  --duet_tree_conf_threshold 0.03
```

phase 분해에는 위 명령에서 P1 또는 P2 하나만 `off`로 바꾼다. 과거
`--duet_tree_policy`/`--duet_tree_nv`는 재현용 deprecated 입력일 뿐 새 실험에는
사용하지 않는다.

---

## 9. 새 설정에서 찾아야 하는 파라미터

모든 값을 큰 grid로 sweep하지 않는다. 먼저 시간 균형과 사후 utility를 측정하고
소수의 후보만 최종 TPS gate로 비교한다.

### 9.1 공통 파라미터

#### Early-exit layer

너무 이르면 proxy quality가 낮고, 너무 늦으면 P2 시작이 늦어진다. 후보 layer별로
다음을 함께 본다.

- proxy root의 실제 cache hit calibration
- target early-exit/proxy 계산 시간
- P1 완료와 proxy 도착의 gap
- 전체 target/draft step p50과 TPS

#### K1

P1 완료 시각과 proxy 수신 시각이 가장 가까운 정수를 선택한다. 현재 기준값은
9지만 모델, exit layer, P1 fanout이 바뀌면 다시 계산한다.

#### K2

P2 완료와 target verify 완료의 gap이 가장 가까운 정수를 선택한다. 현재 기준값은
4다. K2를 늘리면 draft depth와 시간이 함께 늘어나므로 AL만 보고 고르지 않는다.

#### P1 fanout schedule

현재 후보는 `2,2,2,2,2,2,1,1,1,1`이다. P1 hit와 AL을 유지하면서 K1 시간 균형을
맞추는 범위에서만 조정한다.

#### W와 R

W는 P2 forward 폭이고 R은 실제 root 수다. 현재는 cache coverage를 잃지 않도록
`R=W=10`을 기준으로 한다. R을 줄여 P2AL만 높이는 방식은 P2 hit를 떨어뜨릴 수
있으므로 기본 sweep 항목으로 두지 않는다.

### 9.2 동적 tree 파라미터

#### Proxy threshold와 confidence threshold

production dynamic 정책은 round 0에서 모든 root를 평가하고 각 root에서 최대 C개
자식을 sampling하므로 threshold가 root나 cache key를 삭제하지 않는다. round 1
이후 P2는 early-exit proxy score를, P1은 `glue 도달확률 × 시작 token q`를 같은
시작점수로 취급해 root/start threshold를 적용한다. 두 phase 모두 local confidence
threshold도 함께 사용하며, 낮은 점수 leaf 아래의 추가 확장만 막는다.
새 모델/temperature/exit layer에서는 threshold 0/0으로 trace를 수집하고 실제
사후 hit와 accepted child를 라벨로 threshold를 다시 계산한다.

```bash
bash tools/duet_calibration/collect_tree_thresholds.sh
python tools/duet_calibration/analyze_thresholds.py --input /path/to/trace.jsonl
```

threshold는 node를 삭제하는 값이 아니라 그 아래 추가 확장을 멈추는 값으로
해석해야 한다.

Tree 응답의 full-vocabulary parent-q payload도 N1/N2 상한 전체가 아니라 실제 valid
node 수만 송신한다. fused metadata가 먼저 도착하므로 추가 handshake 없이 양쪽이
같은 길이를 알 수 있고, 얕은 P1 tree의 통신량이 상한 크기에 묶이지 않는다.

#### C와 phase별 최대 node 수

- C가 크면 한 부모의 대체 형제를 더 만들 수 있지만 sampling/update 비용과
  저장 후보 수가 늘어난다.
- 최대 node 수가 크면 hit한 root의 coverage/depth가 늘지만 target verify row 비용이
  증가한다.

현재 기준은 C=3, P1 최대 18, P2 최대 8이다. threshold calibration 이후 소수
후보만 paired gate로 비교한다.

### 9.3 K1/K2 자동 분석

```bash
bash tools/duet_calibration/calibrate_k_balance.sh
python tools/duet_calibration/analyze_k_balance.py /path/to/profile_dirs
```

도구가 추천한 K는 시간 균형 후보이지 최종 TPS champion이 아니다. 마지막에는
profiler를 끈 paired run으로 확정한다.

---

## 10. 반드시 기록할 지표

chain/tree 비교에서 전체 AL 하나만 보면 원인을 알 수 없다. 최소한 다음을 함께
기록한다.

| 범주 | 지표 |
|---|---|
| 전체 품질 | tokens/step, 전체 accepted length |
| P1 | hit rate, conditional accepted length, `hit*(AL+1)` |
| P2 | hit rate, conditional accepted length, `hit*(AL+1)` |
| 속도 | decode TPS, target step p50, draft step p50 |
| P1/P2 executor | phase별 replay/capture/fallback/error 수 |
| target | verify 준비, graph pre/post, proxy 계산/전송, accept/recovery |
| startup | tree all-page warmup 시간과 메모리 |

특히 tree 성공 조건은 다음을 모두 만족해야 한다.

1. P1 hit/AL이 chain보다 체계적으로 떨어지지 않는다.
2. P2AL뿐 아니라 `P2 hit*(P2AL+1)`이 증가한다.
3. 전체 tokens/step이 증가한다.
4. 추가 target/tree GPU 비용을 포함한 wall TPS가 개선된다.
5. executor fallback/error가 0이고 timeline의 P2 forward 사이 host gap이 없다.

---

## 11. 현재 지원 범위

현재 최적화된 동적 tree production 경로는 다음 범위다.

- batch size B=1
- target/draft temperature>0
- Llama 계열 draft
- P2는 R≤W; P1은 실제 root 수에 맞춘 고정 canvas를 자동 선택
- ancestry는 63-cell word 여러 개를 사용하므로 `F*W>63`도 지원
- vocabulary≤32768인 packed P_iv wire
- phase별 최대 node 수는 `speculate_k`, `max(K1,K2)`나 P2 W와 독립적인 응답 상한

B>1과 temperature=0은 현재 chain fallback을 사용한다. temperature=0 tree는
ordered residual sampling이 아니라 별도의 greedy top-C proposal과 argmax verifier가
필요하다. 단순히 현재 gate를 제거하면 안 된다.

---

## 12. 코드 위치

| 역할 | 파일 |
|---|---|
| CLI와 기본 정책 | `ssd/bench/bench.py` |
| Config 및 제약 | `ssd/ssd/config.py` |
| proxy 점수, dynamic selection, WOR, verify helper | `ssd/ssd/engine/helpers/p2_tree.py` |
| P1 root/context와 실행기 specialization | `ssd/ssd/engine/helpers/p1_tree.py` |
| 공통 P1/P2 CUDA Graph 실행기 | `ssd/ssd/engine/helpers/p2_tree_executor.py` |
| P1/P2 dispatch, cache/view | `ssd/ssd/engine/draft_runner.py` |
| target tree row와 KV | `ssd/ssd/engine/model_runner.py` |
| target residual verification | `ssd/ssd/engine/verifier.py` |
| target/draft graph helper | `ssd/ssd/engine/helpers/cudagraph_helpers.py` |
| attention/KV guard | `ssd/ssd/layers/attention.py` |
| 공정한 chain/tree gate | `ssd/experiments/proxy_async_overlap/tree_sweep/run_eagle_global_gate_20260807.sh` |
| threshold/K balance 도구 | `ssd/tools/duet_calibration/` |
| tree 상세 기준 문서 | `ssd/docs/duet/TREE_IMPLEMENTATION.md` |
