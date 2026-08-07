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
- P2는 동일한 proxy root 위에 chain 또는 동적 tree continuation을 만들 수 있다.

현재 연구의 기본 P2 정책은 동적 `eagle` tree이고, 기존 chain은
`duet_tree_policy=off` 비교군으로 완전히 유지한다.

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
  P1 chain 생성 -------- proxy 대기 -------- P2 chain 또는 동적 tree 생성
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

P1은 target 정보 없이 실행하는 기존 chain 경로다.

- 깊이는 `K1`이다.
- 위치별 fanout은 `duet_p1_fanout_list`로 지정한다.
- 기본 실험 설정은 `K1=9`, fanout
  `2,2,2,2,2,2,1,1,1,1`이다.
- P1 sampling과 cache key 생성은 tree 도입 전 경로를 유지한다.

P1의 주 목적은 proxy를 기다리는 시간을 실제 draft 계산으로 채우는 것이다.
P1이 너무 짧으면 draft가 proxy를 기다리고, 너무 길면 proxy가 도착한 뒤에도
P1이 끝나지 않아 P2 시작이 늦어진다. 따라서 K1은 acceptance만 보고 정하지
않고 `P1 완료 시각 - proxy 도착 시각`이 0에 가까운 값으로 정한다.

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

`--duet_tree_policy off`가 기존 chain 경로다.

1. proxy `P_iv` 상위 W개 root를 선택한다.
2. 첫 P2 forward에서 W개 root를 동시에 평가한다.
3. 각 root에서 하나의 sampled child를 따라 K2 round까지 continuation을 만든다.
4. root와 continuation을 cache에 저장한다.
5. hit하면 target은 recovery와 K2 chain을 순차적으로 검증한다.

기본 실험 설정은 `W=10`, `K2=4`다. 이때 model workload는 네 번의 W-wide
draft forward다.

chain은 tree 도입 전 P1/P2 semantics와 KV pool을 보존하며, 동적 tree의 품질과
속도를 판단할 기준선이다. 새 실험에서는 기본 정책이 `eagle`로 바뀌었으므로
chain 실행 시 `--duet_tree_policy off`를 반드시 명시한다.

---

## 6. P2 동적 tree 정책

현재 기본 정책은 `--duet_tree_policy eagle`이다. 이것은 별도 EAGLE draft
model을 켜는 `--eagle` 옵션과 다르다. DUET의 draft model과 temperature>0
residual verifier를 유지하고, 다음에 확장할 부모를 누적확률로 선택한다.

### 6.1 기호

| 기호 | 기본값 | 의미 |
|---|---:|---|
| R | 10 | proxy가 선택한 실제 root 수 |
| W | 10 | P2 forward 한 번의 부모 lane 수 |
| F=K2 | 4 | P2 forward round 수 |
| C | 3 | 한 부모에서 동시에 뽑는 형제 수 상한 |
| Nv | 8 | root 하나가 target에 보내는 node 수 상한 |

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

를 계산한다. 현재 동적 정책에는 beta, proxy 제곱근, depth bonus를 넣지 않는다.

### 6.3 round별 동작

각 P2 round는 다음 순서다.

1. 아직 확장하지 않은 현재 depth의 node를 후보로 만든다.
2. 첫 round에서는 R개 root를 모두 평가한다.
3. 이후 round에서는 누적 경로 점수가 높은 부모를 전체 root에서 W개까지 고른다.
4. root별 Nv 공간과 남은 round를 고려해 한 root가 모든 lane을 독점하지 않도록
   quota를 적용한다.
5. token을 뽑기 전에 각 부모의 fanout을 결정한다. 먼저 부모마다 첫 자식을
   배정하고, 여유가 있는 높은 점수 부모에 두 번째와 세 번째 형제를 배정한다.
6. W개 부모를 한 번의 draft forward로 평가한다.
7. 선택된 자식을 arena와 root-local `[R,Nv]` view에 기록한다.

CUDA Graph의 shape는 항상 `F×W`로 고정이지만, 각 lane의 부모, token, fanout,
mask와 topology는 입력 확률에 따라 매 replay 달라진다.

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

### 6.5 expansion threshold

기본 threshold는 다음과 같다.

```text
root proxy threshold = 0.01
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
--duet_tree_policy off
```

chain proxy graph는 공통 최적화이므로 `SSD_CHAIN_PROXY_GRAPH=1`을 유지한다.

### 7.3 동적 tree 전용 최적화

#### 전체 P2 CUDA Graph

tree의 네 P2 forward와 그 사이의 작업을 하나의 graph로 캡처한다.

```text
[부모 선택 -> fanout -> input/rope/mask -> draft forward
 -> logits -> ordered sampling -> node 삽입] × 4
```

따라서 forward 사이 Python, GPU→CPU readback, runtime attention plan과 tensor
allocation이 없다. timeline에서 P2가 막대 하나로 보이지만 내부에 draft forward
네 번이 들어 있다.

#### 고정 GPU arena와 융합 kernel

root/node/token/parent/depth/score/조상 mask를 고정 주소 tensor에 저장한다.
arena reset, child 삽입, root-local view 기록과 metadata 구성을 작은 Triton
kernel로 융합했다. boolean indexing의 숨은 synchronization과 중복 scatter를
사용하지 않는다.

#### page bucket 사전 준비

FlashInfer plan과 CUDA Graph를 가능한 page bucket별로 요청 전에 만든다.

```bash
SSD_TREE_EXEC_WARMUP=all
```

이 warmup은 steady-state decode 시간에는 들어가지 않지만 현재 실모델에서 약
7.18초와 약 1GiB의 시작 비용이 있었다. 서비스 cold-start 및 총 메모리에는
별도로 기록해야 한다.

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
--duet_tree_policy eagle
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

스크립트의 기본 tree 정책은 `eagle`이다. 실행 순서는 다음과 같다.

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

chain:

```bash
SSD_DIST_PORT=16211 \
SSD_TREE_EXEC=0 SSD_TREE_ARENA=0 \
SSD_TREE_PROXY_GRAPH=0 SSD_TREE_EXEC_WARMUP=0 \
/home/chokwans99/anaconda3/envs/ssd/bin/python -O bench/bench.py \
  "${COMMON[@]}" --duet_tree_policy off
```

동적 tree:

```bash
SSD_DIST_PORT=16212 \
SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 \
SSD_TREE_PROXY_GRAPH=1 SSD_TREE_EXEC_WARMUP=all \
/home/chokwans99/anaconda3/envs/ssd/bin/python -O bench/bench.py \
  "${COMMON[@]}" \
  --duet_tree_policy eagle --duet_tree_root_count 10 \
  --duet_tree_c_tensor 3 --duet_tree_nv 8 \
  --duet_tree_proxy_threshold 0.01 \
  --duet_tree_conf_threshold 0.03
```

`--eagle`은 별도의 greedy EAGLE draft model 옵션이므로 위 DUET tree 실행에
추가하지 않는다.

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

새 모델/temperature/exit layer에서는 threshold 0/0으로 trace를 수집하고 실제
사후 hit와 accepted child를 라벨로 threshold를 다시 계산한다.

```bash
bash tools/duet_calibration/collect_tree_thresholds.sh
python tools/duet_calibration/analyze_thresholds.py --input /path/to/trace.jsonl
```

threshold는 node를 삭제하는 값이 아니라 그 아래 추가 확장을 멈추는 값으로
해석해야 한다.

#### C와 Nv

- C가 크면 한 부모의 대체 형제를 더 만들 수 있지만 sampling/update 비용과
  저장 후보 수가 늘어난다.
- Nv가 크면 hit한 root의 coverage/depth가 늘지만 target verify row 비용이
  증가한다.

현재 기준은 C=3, Nv=8이다. threshold calibration 이후 `(C,Nv)` 소수 후보만
paired gate로 비교한다.

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
| P2 executor | replay/capture/fallback/error 수 |
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
- R≤W
- `F*W≤63`인 단일 64-bit ancestry
- vocabulary≤32768인 packed P_iv wire
- `Nv≤max(K1,K2)`, `Nv+1≤W`

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
| 전체 P2 CUDA Graph 실행기 | `ssd/ssd/engine/helpers/p2_tree_executor.py` |
| P1/P2 dispatch, cache/view | `ssd/ssd/engine/draft_runner.py` |
| target tree row와 KV | `ssd/ssd/engine/model_runner.py` |
| target residual verification | `ssd/ssd/engine/verifier.py` |
| target/draft graph helper | `ssd/ssd/engine/helpers/cudagraph_helpers.py` |
| attention/KV guard | `ssd/ssd/layers/attention.py` |
| 공정한 chain/tree gate | `ssd/experiments/proxy_async_overlap/tree_sweep/run_eagle_global_gate_20260807.sh` |
| threshold/K balance 도구 | `ssd/tools/duet_calibration/` |
| tree 상세 기준 문서 | `ssd/docs/duet/TREE_IMPLEMENTATION.md` |
