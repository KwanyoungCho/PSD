# 25 — DUET tree 목표 재정의와 coverage-preserving 설계

## 1. 결론

기존 `confidence` 구현은 출력 샘플링의 무손실 규약을 깨지는 않았지만,
DUET의 시스템 목표를 직접 최적화하지 않았다. 가장 큰 설계 오류는 다음
두 예산을 같은 것으로 본 것이다.

- **부모 평가 예산**: draft forward 4회 × 폭 10 = 40개 셀
- **보관 가능한 자식 노드**: 한 부모 평가에서 WOR 자식 최대 3개가 이미
  생성되며, 캐시 응답 버퍼는 root 10 × `Nv=8` = 80개 노드를 수용

기존 방식은 보관 노드도 40개로 제한한 뒤 root당 깊이 4와 분기를 함께
사기 위해 root를 10개에서 6개로 줄였다. 그 결과 per-hit P2 AL은
올랐지만 P2 cache coverage가 내려가 총 기여가 상쇄됐다. 이는 파라미터
문제가 아니라 목적함수와 자원 회계의 문제다.

새 1차 정책 이름은 `coverage`다. 10개 root와 각 root의 4단계
first-child backbone을 모두 유지하고, 같은 40개 부모 forward에서 이미
생성한 형제들을 root당 `Nv=8`까지 추가한다. production 형상은 root마다
`[3,3,1,1]` fanout이며 총 8노드다. forward 횟수, 폭, CUDA graph의
연산 순서는 바뀌지 않는다.

## 2. 기존 연구에서 가져올 것과 그대로 복사하지 않을 것

- **EAGLE-2**: draft confidence와 acceptance의 상관, 경로 확률 곱을
  노드 값으로 쓰는 원칙, 전체 생성 노드의 global rerank.
- **OPT-Tree**: `E[accepted length]`를 선택된 노드의 prefix probability
  합으로 근사하고 제한된 노드 예산에서 이를 최대화하는 목적함수.
- **Sequoia**: node budget 아래 기대 생성 토큰을 최대화하는 tree
  topology와 다후보 검증의 엄밀한 분리.
- **SGLang EAGLE**: 고정 shape 텐서 안에서 동적 score와 global top-k를
  처리하는 구현 구조.
- **DFVG**: draft/verify 중첩과 동적 sparse tree를 함께 다루는 시스템
  참고점. 이식할 알고리즘의 주 근거는 EAGLE-2/OPT-Tree로 둔다.

참고:

- EAGLE-2: <https://aclanthology.org/2024.emnlp-main.422/>
- OPT-Tree: <https://aclanthology.org/2025.tacl-1.8/>
- Sequoia: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html>
- SGLang 구현: <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_utils.py>
- DFVG: <https://doi.org/10.1145/3779212.3790153>

EAGLE-2의 token top-k를 그대로 복사하지는 않는다. DUET는 temperature
sampling과 ordered WOR residual verifier를 사용한다. 샘플된 token 정체를
본 뒤 유리한 token만 남기면 현재 verifier가 가정하는 proposal 분포를
바꿀 수 있다. 따라서 topology/fanout은 token draw 전에 정하고, 뽑힌
형제는 원래 WOR 순서를 유지한다.

## 3. 최적화 목표와 불변조건

root `r`의 다음 요청 확률 추정치를 `pi_r`, tree node `n`까지의 추정
도달 확률을 `reach(n)`이라 하면 forest의 토큰축 목적은 다음과 같다.

```text
maximize  J = sum_r pi_r * sum_(n in tree_r) reach(n)
```

`reach`의 draft-only 1차 근사는 경로상의 confidence 곱이다. 후속 정책은
실측 depth/sibling acceptance calibration을 사용한다. 단, 아래 제약이
목적함수보다 우선한다.

1. **root coverage**: chain이 보관한 10개 `(seq, terminal, recovery)`
   cache key를 모두 보관한다.
2. **chain containment**: 각 root의 `sib_order=0` 경로는 깊이 K2까지
   존재하며 기존 chain proposal과 같은 순서를 사용한다.
3. **ordered siblings**: 추가 자식은 WOR 순서대로만 검증한다.
4. **prefix closure**: 보관한 모든 노드의 부모도 반드시 보관한다.
5. **fixed execution**: P2는 여전히 4×10 fixed-shape CUDA graph 한 번으로
   실행하며 forward 사이 CPU 작업은 0이다.
6. **P1 floor**: tree hit 뒤 P1 배분도 backbone context마다 기존 chain의
   lane 수를 먼저 보장한 뒤 남는 lane만 sibling terminal에 쓴다.

1~3 때문에 같은 root, 같은 난수를 결합한 검증에서 새 tree는 chain의
엄격한 superset이다. 추가 형제가 기존 first child보다 먼저 시도되지
않으므로, 동일 hit에서 accepted length가 chain보다 짧아지면 구현 오류다.
root key도 같으므로 cache hit가 구조적으로 줄면 역시 구현 오류다.

## 4. 단계별 구현

### 단계 A — coverage baseline

- `R=W=10`, `F=K2=4`, `C=3`, `Nv=8`
- live root budget은 모두 8, padding root만 0
- parent forward는 매 round 각 root의 backbone tip 10개만 평가
- root별 fanout `[3,3,1,1]`
- 생성/보관 노드 80개, 부모 model evaluation은 기존과 같은 40개
- 기존 `confidence`와 `off`는 재현용으로 그대로 유지

이 단계의 목적은 parameter sweep이 아니라 세 가지 구조 판정이다.

- root coverage 10/10
- backbone depth 4 및 first-child exact
- CUDA executor와 eager arena의 topology/view exact parity

### 단계 B — 확률 기반 optional-node 최적화

단계 A가 실제 엔진에서 hit를 보존하고 AL을 올린 뒤에만 한다. mandatory
40개 backbone은 고정하고 optional sibling 40개 사이의 우선순위만
바꾼다. 후보의 marginal value는 다음 형태로 평가한다.

```text
value(child_j) = pi_root
               * estimated_reach(parent)
               * estimated_reject_probability(previous_siblings)
               * estimated_accept_probability(child_j)
```

초기 calibration은 기존 trace의 depth/sibling별 실제 acceptance로
구한다. token identity를 사용한 사후 가지치기는 하지 않는다. 향후
verifier를 EAGLE식 deterministic top-k에 맞게 별도로 바꾸지 않는 한,
이 제약을 유지한다.

## 5. 빠른 검증 순서

1. CPU/unit: 10개 root 모두 valid=8, topology `[3,3,1,1]`, prefix
   closure, chain containment, tree prefix-mass > chain prefix-mass.
2. CUDA parity: 같은 입력/noise에서 arena와 executor의 token, parent,
   sibling, mask, q-ref가 exact.
3. 짧은 1-seed smoke: crash/OOB/NaN, P2 graph replay/fallback, 실제
   topology 10×8 확인. 여기서는 성능 결론을 내리지 않는다.
4. 최종 3-seed order-rotated A/B/C 한 번: chain / 기존 confidence /
   coverage. P1/P2 hit, conditional AL, phase contribution, tok/step, TPS,
   target/draft span을 함께 판정한다.

최종 채택 조건은 다음과 같다.

- P2 root coverage가 chain과 동일하며 P2 hit의 유의한 하락이 없음
- P1 hit/AL의 유의한 하락이 없음
- P2 conditional AL과 `P2_hit * (P2AL+1)` 기여가 모두 증가
- P2 forward 사이 host gap 0 유지
- tok/step과 TPS가 chain보다 증가

조건을 못 넘으면 sweep하지 않는다. 같은 root/backbone의 paired trace를
먼저 비교해 구현 오류와 acceptance-model 오류를 분리한다.

## 6. 구현 및 1차 최종 게이트 결과 (2026-08-07)

`coverage`를 config/CLI, CPU reference, eager arena, full-P2 CUDA graph
executor에 배선했다. 기존 `off`와 `confidence` 동작은 그대로 남겼다.

구조 검증:

- CPU/arena 91 tests 통과
- executor CUDA parity 9 tests 통과
- 실엔진 smoke: 153 replay, fallback/capture-at-runtime 0
- 모든 replay가 root 10개 × valid 8, topology
  `par=[-1,-1,-1,0,0,0,3,6]`, `sib=[0,1,2,0,1,2,0,0]`
- 41 P2 hit 중 sibling path가 실제 수락된 경우 11개
- 같은 hit에서 sibling을 제거한 paired chain projection AL 1.512,
  실제 tree AL 1.780: **+0.268 (+17.7%)**
- hit 41개 중 10개가 기존 confidence가 버린 rank 6--9 root. 해당
  lower-root hit의 AL은 2.7이었다.

최종 조건은 40 prompt × output 128, seed 42/123/2024, 세 arm의 실행
순서를 seed마다 회전하고 PROFILE을 끈 상태다. 원 로그는
`experiments/proxy_async_overlap/tree_sweep/coverage_final_gate_20260807/`.

| arm | TPS | tok/step | P1 hit | P1AL | P2 hit | P2AL | P2 기여 | target verify ms | draft step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 62.37 | 3.487 | 0.505 | 3.403 | 0.274 | 1.553 | 0.700 | 50.46 | 50.38 |
| confidence-R6 | 58.97 | 3.567 | 0.513 | 3.350 | 0.232 | 1.967 | 0.689 | 52.08 | 60.03 |
| **coverage-R10** | **60.57** | **3.703** | **0.515** | **3.580** | **0.276** | **1.897** | **0.800** | **52.79** | **60.58** |

여기서 `P2 기여 = P2_hit × (P2AL+1)`이다.

판정:

1. 기존 confidence는 P2AL을 +0.413 올렸지만 hit를 -4.17%p 잃어 P2
   기여가 chain보다 -0.011이었다. root를 버린 설계가 잘못이었다.
2. coverage는 hit를 +0.23%p로 보존하면서 P2AL +0.343, P2 기여
   +0.101(+14.4%), tok/step +0.217(+6.2%)를 얻었다. **AL/tree 목표와
   cache coverage 목표는 통과했다.**
3. P1 hit/AL도 평균 하락하지 않았다. seed별 변동은 있으나 P1 기여
   평균은 chain 2.224 → coverage 2.356이다.
4. TPS는 chain 대비 -1.81 tok/s(-2.9%)다. target이 5행 chain 대신
   recovery+8 node를 실제 계산하므로 target verify +2.33ms가 남고,
   늦은 proxy 도착이 draft step에도 반영된다. 이는 forward 사이 CPU
   gap이 돌아온 것이 아니다.

따라서 `coverage`는 **목적함수가 바로잡힌 새 기준선**으로 채택하되 아직
production default로 바꾸지 않는다. 다음 최적화는 root/chain containment를
절대 건드리지 않고 optional sibling 수만 confidence에 따라 줄이는
`coverage-adaptive`다. 목표는 현재 +0.101 P2 기여를 최대한 보존하면서
평균 target row를 낮춰 남은 -2.9% TPS를 회수하는 것이다. root 수, P2
forward 횟수, P1 배분을 다시 sweep하지 않는다.

## 7. 전역 누적확률 정책으로의 수정 (2026-08-07)

위 결론에서 **모든 P2 root를 첫 forward에서 평가한다**는 원칙은 유지한다.
그러나 모든 root에 깊이 4의 첫 경로를 강제로 보장하는 것은 철회한다.
낮은 proxy 확률의 root에도 이후 forward 예산을 계속 쓰기 때문에 사용자가
원한 confidence 기반 트리가 아니며, 높은 확률 경로를 더 깊게 확인할 기회를
잃는다.

새 `duet_tree_policy=eagle`은 DUET의 draft model과 residual sampling/verifier를
그대로 사용하고, **어떤 노드를 다음 round에 확장할지**만 EAGLE-2식 전역
누적확률 기준으로 바꾼다. 이 이름은 `--eagle`로 켜는 별도의 EAGLE draft
model 기능과 무관하다.

### 7.1 R과 W의 의미 및 제약

- `R`: target proxy가 보낸 P2 root 수
- `W`: 한 번의 P2 draft forward가 동시에 계산할 수 있는 부모 수
- 기본값은 `R=W`이며 현재 champion은 `R=W=10`이다.
- 현재 round-synchronous 실행기에서는 `R>W`를 지원하지 않는다. 첫
  forward에서 W개만 평가한 뒤 다음 round로 넘어가므로 나머지 root를 다시
  평가할 차례가 없다. config가 이를 명시적으로 거부한다.
- `R<W`는 가능하지만 root coverage를 스스로 줄이므로 이번 gate에서는 쓰지
  않는다.

즉 첫 forward는 proxy 상위 10개 root를 모두 계산한다. 이후 forward에서는
그 10개 root를 매번 하나씩 계속 계산하는 것이 아니라, 지금까지 관측한
확률이 높은 경로의 부모를 전체 root를 통틀어 W개까지 선택한다.

### 7.2 점수와 round별 동작

root `r` 아래 경로가 지금까지 토큰 `x1,...,xd`를 만들었다면 확장 점수는
다음과 같다.

```text
score(path) = P_proxy(root=r)
            * q(x1 | root)
            * q(x2 | root,x1)
            * ...
            * q(xd | root,x1,...,x{d-1})
```

로그 공간에서는 덧셈으로 계산한다. `P_proxy`는 target early-exit가 준 root
확률이고, 각 `q`는 draft가 해당 부모에서 계산한 자식 확률이다. 별도의 beta,
제곱근, 깊이 보너스는 넣지 않는다.

각 round는 다음 순서를 따른다.

1. 첫 round는 R개 root를 모두 forward한다.
2. 각 부모에서 최대 C개의 ordered residual sample을 만든다.
3. 지금까지 생긴 모든 leaf 중 누적확률이 높은 부모를 다음 round의 W개
   forward lane에 전역 배치한다.
4. 한 root의 최종 view 한도 `Nv`를 넘지 않도록 하고, 그 root가 남은
   round에도 더 깊어질 수 있도록 필요한 슬롯만 예약한다.
5. 선택된 부모에는 먼저 자식 하나씩을 배정하고, 여유 슬롯이 있는 부모만
   confidence 순서대로 형제를 최대 C개까지 추가한다.

따라서 `W=10, F=4, C=3, Nv=8`에서 높은 확률 root는 결과적으로
`[3,3,1,1]` 형태가 될 수 있지만 이것은 고정 topology가 아니다. 낮은 확률
root는 첫 round의 자식만 갖고 멈출 수 있으며, 둘 이상의 root가 중간 깊이로
확장될 수도 있다.

### 7.3 EAGLE과 같고 다른 부분

같은 부분은 누적 경로확률로 전체 leaf를 비교해 다음 확장 부모를 고르는
것이다. 다른 부분은 DUET가 temperature>0의 ordered residual sampling과
lossless verifier를 유지한다는 점이다.

일반 EAGLE처럼 모든 draft가 끝난 뒤 token 점수만 보고 이미 뽑은 형제를
임의로 삭제하거나 순서를 바꾸지 않는다. 그렇게 하면 DUET verifier가
가정하는 제안 분포가 달라질 수 있다. 현재 구현은 다음 round의 **확장 여부**만
과거 round에서 이미 관측한 확률로 정하며, 현재 부모의 fanout 수는 그 부모의
새 token을 뽑기 전에 확정한다. 따라서 생성된 sibling group의 residual
sampling 순서와 검증 규약은 그대로다.

### 7.4 구현 및 검증 상태

- config/CLI, CPU reference, eager GPU arena, full-P2 CUDA graph executor에
  `eagle` 정책을 배선했다.
- `R>W`는 config와 runtime에서 모두 실패하도록 고정했다.
- `run_fi_tree_decode_cudagraph`에 빠졌던 `torch.inference_mode()`를 복구했다.
- CPU/unit 95개와 CUDA executor 포함 106개 테스트가 통과했다.
- CUDA 테스트는 첫 round에서 모든 root가 유효한지, root별 동적 깊이가 실제로
  달라지는지, 같은 입력에서 eager와 graph replay의 token/topology/mask가
  일치하는지 확인한다.

실엔진 판정은 sweep 없이 다음 한 번으로 제한한다.

1. seed 42의 짧은 smoke: OOB/NaN/fallback, 실제 topology, 모든 root 평가 확인
2. seed 42/123/2024의 순서 회전 chain 대 tree: hit, conditional AL,
   `hit*(AL+1)`, tok/step, TPS
3. 통과 후 각 arm의 짧은 profile 한 번: P2 forward 사이 공백과 target
   pre/exit/post 확인

P1 tree는 이 gate에 섞지 않는다. P1은 hit 비중이 더 높고 현재 P1 root에는
P2와 같은 최신 target proxy 확률이 없으며, `W1=16, K1=9`는 기존 단일
64-bit ancestry 표현을 넘는다. P2 정책이 채택된 뒤 별도 flag와 별도
non-regression gate로 검토한다.

## 8. 전역 깊이 배분의 반증과 adaptive-backbone (2026-08-07)

§7의 `eagle` 전역 확장을 80 prompt × output 384에서 직접 비교한 결과,
낮은 누적점수 root를 깊이 1에서 멈추는 정책은 목적에 맞지 않았다.

| seed | arm | P2 hit | P2AL | P2 기여 | TPS |
|---|---|---:|---:|---:|---:|
| 42 | chain | 0.250 | 1.76 | 0.690 | 73.65 |
| 42 | global-depth | 0.237 | 1.38 | 0.564 | 65.08 |
| 123 | chain | 0.254 | 1.83 | 0.719 | 69.48 |
| 123 | global-depth | 0.242 | 1.46 | 0.595 | 65.48 |

두 seed가 같은 방향이므로 세 번째 seed는 중단했다. proxy 점수가 낮은 root도
실제 cache hit이 되며, 그때 view가 깊이 1이면 tree가 chain보다 짧아진다.
따라서 `eagle`은 실험용 반례 정책으로 남기고 채택하지 않는다.

대신 `adaptive`는 다음 불변을 둔다.

1. 모든 R root의 첫 자식 경로는 F=4까지 반드시 연장한다.
2. 부모 점수는 `log P_proxy(root) + Σ log q(previous child)`이다.
3. 현재 살아 있는 backbone tip들의 정규화 질량 평균 이상인 부모만
   2·3번째 형제를 보관한다.
4. 판단은 현재 token을 sample하기 전에 끝내며, sampled identity를 보고
   형제를 버리지 않는다.

따라서 약한 root도 4노드 chain을 유지하고, 강한 root만 6 또는 8노드가
된다. 별도 beta·sqrt·rank threshold는 없다. 짧은 실엔진 감사에서는
195/195 parent-logit 연결이 통과했고 모든 root의 깊이가 4였으며, step당
10개 root의 총 view 노드는 평균 54.8개였다(고정 coverage 80개 대비
31.5% 감소). 최종 품질/TPS 및 timeline gate는 진행 중이다.

### 8.1 함께 발견한 CUDA graph 버킷 버그

page 버킷별 graph가 내부 root-local node-index 텐서를 서로 공유하면 긴
실행에서 illegal memory access가 발생했다. 감사만 고치기 위해 하나의 공용
버퍼로 바꾼 것이 잘못이었다. 버킷별 텐서를 별도로 보관하고 replay 전에
해당 버킷의 텐서를 선택하도록 수정했다. 실모델 7버킷 사전 캡처에서
동기화 ON 357 replay, 동기화 OFF 304 replay를 모두 통과했다.

## 9. adaptive 반증, coverage 재선택, 최종 시간축 감사 (2026-08-07)

### 9.1 adaptive도 주력으로 채택하지 않는다

`adaptive`는 모든 root의 깊이 4를 보존하고 optional sibling만 줄였으므로
`eagle`의 coverage 붕괴는 막았다. 그러나 seed 42 장기 런에서 다음과 같이
tree 이득도 거의 없앴다.

| arm | P2 hit | P2AL | P2 기여 | TPS |
|---|---:|---:|---:|---:|
| chain | 0.250 | 1.76 | 0.690 | 73.65 |
| adaptive | 0.245 | 1.79 | 0.684 | 64.68 |

평균 view 수는 80→54.8로 줄었지만 P2 기여가 chain보다 작았으므로 추가
seed를 돌리지 않았다. `eagle`과 `adaptive`는 반례/연구용 flag로만 남기며,
실험 script의 기본 tree policy는 다시 `coverage`로 고정한다.

### 9.2 현재 선택 정책과 장기 확인 결과

현재 선택 정책은 `coverage`다. 설정은 `R=W=10`, `K2=4`, `C=3`,
`Nv=8`이며 모든 root가 첫 forward를 받고, root마다 깊이 4의 first-child
경로를 유지한다. optional sibling을 포함한 view는 `[3,3,1,1]`이다. 이
형상은 CUDA graph의 강제 형상이 아니라 `Nv=8`, `C=3`, backbone 보존에서
나온 현재 정책의 결과다.

80 prompt × output 384, seed 42의 재확인은 다음과 같다.

| arm | TPS | tok/step | P1 hit | P1AL | P2 hit | P2AL | P2 기여 | target verify | draft step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 73.65 | 4.03 | 0.560 | 4.04 | 0.250 | 1.76 | 0.690 | 49.63 ms | 49.78 ms |
| coverage | 66.98 | 4.01 | 0.574 | 3.86 | 0.240 | 1.93 | 0.703 | 52.18 ms | 59.31 ms |

P2AL은 +0.17, P2 기여는 +1.9%지만 tok/step은 사실상 같고 TPS는 -9.1%다.
§6의 더 짧은 3-seed gate에서는 P2 기여 +14.4%였으므로, 현재 quality
개선량은 prompt/output 분포에 민감하다. 따라서 `coverage`는 가장 안전한
tree 기준선이지만 아직 production champion은 아니다.

### 9.3 P2 forward 사이 공백 판정

새 profile에서 tree의 4-round P2는 `p2_graph_replay` 한 구간으로 보이며
p50 12.34 ms다. replay 내부의 forward→sample→tree-update→다음-forward는
하나의 CUDA graph로 실행되고, 중간 CPU readback/plan 호출은 0이다. 즉
처음 목표였던 **P2 draft forward 사이 host 공백 제거는 완료**됐다.

다만 공백 제거와 전체 P2 비용 동률은 다르다. chain은 모델 replay 네 번의
합이 약 9.2--9.9 ms이고, tree graph에는 ordered sibling sampling과 동적
topology 갱신 커널이 더 들어 있어 replay 자체가 약 2.5--3 ms 길다.
`phase2_build`도 chain 약 0.8 ms, tree 약 1.9 ms다. 이것이 target wait와
합쳐져 tree draft step이 아직 약 8--10 ms 느린 주된 이유다.

### 9.4 target 시간축과 profiler 교정

warmup 이후 profile의 p50은 다음과 같다.

| target 구간 | chain K2 | tree K2 |
|---|---:|---:|
| verify 준비 | 0.36 ms | 1.63 ms |
| 앞부분 graph | 26.72 ms | 33.76 ms |
| exit proxy 실제 side work | 0.32+0.38 ms | 1.24 ms |
| 뒷부분 graph | 9.70 ms | 13.05 ms |
| 수락/복구 | 4.40 ms | 4.30 ms |

tree K2는 chain의 recovery+4행 대신 recovery+8노드를 target이 실제로
검증하므로 앞/뒤 model graph 증가는 물리적 row 비용이다. 별도 tree mask
구성+FlashInfer plan은 verify 준비 1.63 ms 안에 포함된다. 반면 tree
`miss`에서 `exit_logits`가 p50 4.92 ms인 현상은 tree topology 검증 비용이
아니다(그 step은 tree meta가 없다). miss/JIT의 exit-replica dispatch가
hit_k1과 다른 경로를 타는지 다음 target-side 최적화에서 분리해야 한다.

두 profiler 오류도 수정했다.

1. target `exit_logits`를 default stream event로 재던 방식은 실제 side-stream
   계산과 launch 시간을 섞었다. `exit_proxy_launch`와
   `exit_proxy_side`로 분리했다.
2. draft profile이 12,000 event cap에 먼저 닿으면 target-only 후반 step을
   대표로 골라 draft 행이 빈 그림이 생성됐다. 양쪽 response marker가 있는
   공통 step 중 median을 고르도록 plotter를 수정했다.

과거 700--800 ms 막대는 약 23,000번째 CUDA event 부근에서 chain/tree 모두
서로 다른 label에 발생한 profiler 축적 stall이었다. event cap을 12,000으로
고정했다. 새 profile의 79--86 ms `verify_sample_accept` 첫-step cold start는
DUET verifier 초기화에서 K1/K2 hit/miss softmax·multinomial을 미리 실행해
제거했다(chain 최대 9.8 ms, tree 최대 13.3 ms). 최초 NCCL 송수신과 첫
request 대기는 여전히 cold-start outlier이며 정상 step의 pre/post 병목과
분리해서 본다.

### 9.5 pre-tree branch와 현재 chain

pre-tree branch `e29c4b6`은 현재 branch의 direct ancestor다. 현재 chain
route와 KV pool은 남아 있지만, tree 작업 중 split graph/warmup/profiler/
watchdog를 포함한 공통 파일도 바뀌어 byte-identical하지는 않다.

동일한 짧은 champion workload(8 prompts, output 96)를 별도 worktree에서
직접 비교했다.

| branch | TPS | target verify | draft step | P1AL | P2AL |
|---|---:|---:|---:|---:|---:|
| pre-tree `e29c4b6` | 69.52 | 46.34 ms | 46.13 ms | 3.72 | 1.79 |
| current chain | 70.86 | 46.38 ms | 46.17 ms | 3.74 | 1.99 |

짧은 stochastic quality 값은 판정용이 아니지만 두 branch의 target/draft
latency는 0.04 ms 차이로 같다. 따라서 현재 chain의 75 tok/s와 과거 80+
headline 차이를 tree 공통 코드의 명백한 latency 회귀로 볼 증거는 없다.
정확한 80+ 재현은 같은 서버 부하와 200 prompts × output 512의 반복 gate가
필요하다.

### 9.6 P1 tree는 별도 단계로 미룬다

P1 tree의 잠재 가치는 크다(P1 hit 약 0.54--0.58). 그러나 P2와 동시에
넣으면 안 된다.

- P1 시작 시점에는 최신 target proxy root 확률이 없다.
- `W1=16`, `K1=9`는 144 후보로 현재 단일 int64 ancestry 표현의 63-node
  한도를 넘는다.
- P1 cache/wire/target verify에는 아직 P2 tree metadata 계약이 없다.
- P1은 다수 hit 경로이므로 hit/AL non-regression 요구가 P2보다 엄격하다.

먼저 P2의 target verify와 남은 draft tree kernel 비용을 줄여 chain보다 TPS가
높아진 뒤, P1 tree를 별도 flag로 구현한다. 그때는 multiword ancestry,
`R1=W1=16`, `Nv1<=K1`, all-root backbone, P1-only 3-arm gate가 선결이다.
