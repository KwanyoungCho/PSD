# DUET P1 dynamic-tree audit and tuning plan

작성일: 2026-08-10

## 1. 질문과 판정 기준

이 검토는 다음 질문을 서로 섞지 않고 순서대로 답한다.

1. P1 tree의 후보, root key, topology, parent-q가 target verifier까지 정확히 전달되는가?
2. 동일한 root 집합과 동일한 draft forward-cell 예산에서 tree를 chain으로 퇴화시키면 기존 P1 chain과 같은가?
3. 실제 hit root에 충분한 node가 배분되는가? 배분됐는데도 verifier에서 잃는가?
4. 분기가 조건부 P1 AL과 전체 tokens/step을 실제로 높이는가?
5. 품질 이득이 있더라도 P1/proxy와 draft/target의 두 rendezvous를 만족해 TPS 이득으로 연결되는가?
6. threshold가 정말 필요한가? 필요하다면 품질 배분용인지 latency 절감용인지 구분할 수 있는가?

P1 tree가 직접 최적화하는 값은 **hit 이후의 conditional P1 AL**이다. 첫 cache
root 선택은 별도 정책이므로 hit rate는 목표가 아니라 두 arm의 root 모집단이 크게
달라지지 않았는지 보는 통제 지표로만 사용한다. 최종 판정은 아래를 함께 본다.

- `P1 hit rate` (통제 지표; tree 품질 목표가 아님)
- `P1 accepted length | P1 hit`
- P1 기여량: `P1_hit_rate * (P1_AL + 1)` 또는 step별 exact accepted token 수
- 전체 `tokens/step`
- target에 실제 전송된 tree의 valid node 수, depth, sibling 수
- 실제 hit root의 생성 node 수와 root rank
- 생성 tree 대비 verify-cap rerank 후 보존된 node/path
- P1 종료 시각과 target proxy 도착 시각
- draft 응답 완료 시각과 target verify 완료/다음 요청 시각
- 최종 TPS

## 2. 현재 확인된 구조적 문제

대표 formal 설정의 P1 chain root 수는 위치별 fanout
`[2,2,2,2,2,2,1,1,1,1]`의 합인 16이다. 반면 P1 tree는
`roots_per_position=2`를 10개 위치에 균일 적용해 20개 root를 만든다.

따라서 기존 비교는 topology만 바꾼 비교가 아니다.

- root coverage가 16 대 20으로 다르다.
- cache hit의 모집단이 달라져 조건부 P1 AL을 직접 비교할 수 없다.
- tree round 0은 20개 root를 모두 평가하지만 이후 round의 width는 chain fanout 합인
  16이다. 매 round 적어도 네 root는 연장되지 않는다.
- tree hit 뒤의 재귀 P1은 context 수가 커져 round-0 폭이 더 커질 수 있다.

기존 trace에서 작은 P1 tree(3--9 nodes)는 hit의 18.4%, AL 0.91이었고,
18-node tree는 81.6%, AL 4.31로 chain 4.43에 근접했다. 이는 구현 오류의 증거라기보다
실제 hit root가 짧은 tree를 받는 tail이 평균을 크게 낮춘다는 가설과 일치한다. 아래 실험은
이 가설을 직접 검증한다.

## 3. EAGLE2/SGLang과의 알고리즘 비교

로컬 SGLang 0.5.16의 기준 구현은 다음 파일을 기준으로 한다.

- `sglang/srt/speculative/spec_utils.py`: cumulative path score와 global top-k
- `sglang/srt/speculative/eagle_worker_v2.py`: round별 확장과 최종 후보 조직
- `sglang/srt/speculative/eagle_utils.py`: parent list와 verify mask 생성

공통점은 `root_prior * product(edge_q)`에 해당하는 누적 path confidence로 global
frontier를 선택하는 것이다. DUET의 captured executor도 같은 종류의 global selector를
사용하며, P1의 root prior만 `context_reach * local_root_q`이다.

그러나 목적함수는 같지 않다.

- EAGLE2는 현재 request의 하나의 proposal tree를 최적화한다.
- DUET P1은 여러 미래 cache key에 대응하는 forest를 미리 만든다.
- DUET에서는 현재 score가 낮아도 실제 다음 request에서 hit할 root의 최소 coverage가
  중요하다. 순수 global top-k는 이 root를 짧게 만들 수 있다.
- temperature 0.7 DUET은 ordered-without-replacement sibling과 lossless residual verifier를
  사용한다. 단순 greedy EAGLE tree와 sibling의 의미도 다르다.

따라서 EAGLE2 selector를 그대로 썼다는 사실만으로 P1 AL 향상이 보장되지는 않는다.
P1에는 root-coverage와 hit-conditioned value를 포함한 목적함수가 필요할 가능성이 있다.

## 4. 단계별 검증

### A. 정적 데이터 경로와 invariant audit

다음을 root id 기준으로 끝까지 추적한다.

`glue context -> root token/q/reach -> cache key -> hit row -> root-local view -> M rerank -> wire topology -> parent-q gather -> target verify walk`

강제 invariant:

- root token은 동일 위치의 returned token을 제외한 chain selector와 같은 top-k이다.
- root score는 원분포의 local q에 context reach를 곱한 값이다.
- 모든 child의 parent가 먼저 나오고 sibling order는 0부터 연속이다.
- 같은 parent의 siblings는 같은 parent-q row를 참조한다.
- cache hit row에서 계산한 phase/root index가 전송 view와 같다.
- rerank는 ancestor와 앞선 WOR sibling을 보존하고 parent/qref를 재매핑한다.
- `tree.valid == valid_k <= phase verify cap`이다.
- target의 node q는 node 자신의 logits가 아니라 parent context의 q이다.

진단 hook은 `SSD_TREE_TOPO_TRACE`, `SSD_TREE_NODE_AUDIT`,
`SSD_TREE_CALIB_TRACE`, `SSD_DUET_E0_TRACE`를 사용한다. TPS run에서는 모두 끈다.

### B. unit/CUDA contract gate

1. arena CPU/GPU parity
2. eager/CUDA-graph parity
3. variable-width mask와 multiword ancestry
4. parent-q reference와 wire validation
5. tree verify walk와 residual-ladder parity
6. serving rerank의 parent/sibling/qref remap
7. P1 전용 chain-degenerate end-to-end contract

실패 시 sweep을 진행하지 않고 원인을 고친 뒤 같은 gate를 반복한다.

### C. 공정한 chain-degenerate 실험

고정 데이터: Spec-Bench의 동일 7개 category request, raw prompt, temp 0.7, 같은 seed.
초기 gate는 output 128, 확인 gate는 output 256 이상을 사용한다.

| 군 | chain root 설정 | tree root 설정 | continuation width | 목적 |
|---|---:|---:|---:|---|
| U1 | fanout=1, 총 `K1+1` | rpp=1, 총 `K1+1` | `K1+1` | 가장 깨끗한 동일-root 계약 |
| U2 | fanout=2, 총 `2(K1+1)` | rpp=2, 총 `2(K1+1)` | `2(K1+1)` | 대표 root density에서 재검증 |

각 군에서 아래 arm을 비교한다.

- chain
- tree `C=1`: sibling이 없는 chain-degenerate topology
- tree `C=3`, threshold 0/0: 동일 forward-cell 수에서 global branching
- 같은 생성 tree를 first-child projection으로 오프라인 재검증한 counterfactual

`C=1`에서 root token, root key, cache hit, tree backbone token, parent-q, verifier 결과가
불일치하면 구현 문제다. stochastic target output은 RNG 소비 순서 차이까지 따로 기록하며,
구조 계약은 고정 noise/coin unit test로 판단한다.

### D. 문제 위치 분해

각 P1 hit를 다음 strata로 나눈다.

- hit root rank/context position
- root start score, context reach, local q
- 생성 valid nodes: 1--3, 4--6, 7--9, 10+
- 전송 valid nodes와 max depth
- sibling 사용 여부
- accepted path에서 sibling이 추가한 node 수

세 counterfactual을 계산한다.

1. **Topology value**: 동일 전송 tree의 full walk 대 first-child-only walk
2. **Rerank loss**: 생성 tree 대 M-cap subtree가 accepted path를 보존하는 비율
3. **Allocation oracle**: 실제 hit root에 forest의 spare expansion을 우선 배분했을 때의
   가능한 node/depth 상한

이로써 다음을 구분한다.

- hit root가 짧다: root allocation/score 문제
- hit root는 길지만 sibling이 없다: fanout/global selector 문제
- sibling은 있으나 target에서 사라진다: verify cap/rerank 문제
- target에 정확히 도착하지만 AL이 안 오른다: proposal quality 문제
- AL은 오르지만 TPS가 내린다: pipeline timing/target row cost 문제

### E. 품질 sweep

초기 gate를 통과한 U1을 기준으로 작은 sweep을 한다.

- `C`: 1, 2, 3
- generated `N1`: `K1`, 12, 15, 18
- verify `M1`: `K1`, 10, 12, 14 (항상 `M1 <= N1`)
- P1 start/conf threshold:
  - off: 0 / 0
  - safe: 0.001 / 0.01
  - balanced: 0.01 / 0.01
- 필요할 때 selector:
  - pure global
  - per-root minimum depth 1 또는 2 후 global
  - hit-value calibrated global score

모든 조합을 곱하지 않는다. 순차적으로 topology value가 없는 조합을 제거한 뒤 N/M,
threshold를 좁힌다. 최소 2개 seed에서 방향이 같은 조합만 timing gate로 보낸다.

### F. threshold 필요성 판정

현재 threshold는 captured graph의 round 수나 tensor width를 바꾸지 않는다. 그러므로
threshold만 높여도 draft kernel latency는 거의 줄지 않고, expansion을 다른 root로
재배분하는 효과가 중심이다.

threshold는 아래 중 하나를 만족할 때만 유지한다.

- 같은 forward-cell 수에서 실제 hit root의 useful expansion/node가 증가한다.
- 같은 M에서 accepted path 보존률 또는 전체 tokens/step이 증가한다.
- threshold로 무효화한 lane을 실제로 실행하지 않는 variable-work 경로를 별도로 구현해
  latency 감소가 측정된다.

그 외에는 threshold 0/0을 기본으로 두고 N/M 또는 root-coverage 규칙으로 제어한다.

### G. pipeline timing gate

품질 상위 2--3개 설정만 profiler를 켜고 확인한다.

1. `P1_done <= proxy_arrival`: 양수 proxy wait가 크지 않아야 한다.
2. `draft_response_done <= target_ready_for_response`: target의 spec wait가 커지지 않아야 한다.
3. tree verify의 row 증가가 `graph_pre`, `graph_post`, `verify_sample_accept`를 얼마나 늘리는지
   분리한다.

조정 순서는 다음과 같다.

1. M1을 줄여 target verify row 비용과 다음 proxy 도착을 앞당긴다.
2. N1/M1 비율을 조정해 wider search와 served tree를 분리한다.
3. K1을 7/8/9에서 좁게 조정한다. K1 감소가 단순 proxy wait 증가로 상쇄되는지 본다.
4. 마지막에 exit layer를 조정한다. early exit sweep은 P1 품질 설정이 고정된 뒤 한다.

## 5. 승격 기준

소규모 설정을 논문용 확장 실험으로 올리려면 다음을 모두 만족해야 한다.

- 모든 correctness gate 통과
- 공정한 U1 또는 U2 비교에서 P1 contribution과 전체 tokens/step이 chain 이상
- 두 seed에서 topology value의 방향이 동일
- 실제 hit root의 생성/전송 budget 손실 원인이 설명됨
- P1/proxy와 draft/target rendezvous가 timeline으로 확인됨
- TPS 손실이 있으면 AL 이득과 명확한 Pareto trade-off로 보고 가능

논문 표에는 기존 비균등-chain 대 균등-tree 결과를 공정 비교로 사용하지 않고,
`current configuration` 진단 arm으로만 남긴다.

## 6. 완료 결과 (2026-08-11)

### 6.1 correctness와 데이터 경로

- root id에서 cache row, rerank, wire parent/qref, target tree walk까지 정적 audit를
  완료했다. 선택된 root를 다른 view로 보내거나 node 자신의 q를 parent q 대신 쓰는
  오류는 발견되지 않았다.
- C=1/R=W P1 forest는 모든 root에서 정확한 K-depth chain, parent, sibling=0,
  parent-q cell을 만든다. CPU tree verifier도 같은 coin/terminal sampler에서 일반
  speculative decoding과 모든 first-reject depth에서 일치한다.
- CUDA eager/graph, variable-width mask, multiword ancestry, rerank remap을 포함한 관련
  회귀 테스트는 182개 중 166 pass, 16 model-path skip이다.
- long-context mock config의 `extend_draft_rope` 직접 접근은
  `getattr(..., False)`로 호환 수정했다.

### 6.2 실제 실패 원인

기존 global P1 trace 345회에서 모든 생성 root의 50.87%가 7 node 미만이었고,
실제로 hit된 root도 18.85%가 7 node 미만이었다. 14-node hit의 AL은 2.68이지만
3-node hit의 AL은 0.82였다. sibling은 hit의 16.4%에서 사용됐고 accepted node의
약 9.9%를 추가해 topology 자체에는 가치가 있었다. 문제는 얕은-root tail이었다.

P1 root score의 hit 판별 AUC는 0.729로 유용하지만 완전하지 않았다. EAGLE2처럼 현재
request의 한 tree에서 global frontier를 고르는 것과 달리 P1은 아직 target proxy를
받기 전에 여러 미래 cache key의 forest를 만든다. 따라서 global score나 상위 root만
깊게 만드는 방식은 seed에 따라 실제 hit root를 놓쳤다.

두 번째 구현 문제는 `backbone` 정책도 tree hit 뒤에는 root 39개에 continuation lane
27개만 예약해 실제로는 12개 root를 깊이 1에서 끊었다는 점이다. seed42에서는 우연히
score와 hit가 맞아 AL이 올랐지만 seed123에서는 P1 AL 3.42로 P2-only 4.00보다
낮아졌다. 현재 `backbone`은 모든 root에 lane을 예약하고 scheduler도 동일한 compact
cell 수를 예약한다. 대표 widest context의 round width는 `(39,27,...)`가 아니라
`(39,39,...)`다.

### 6.3 최종 작은 sweep

공통 조건은 Llama-2/LayerSkip 70B target, TinyLlama draft, raw Spec-Bench,
temperature 0.7, output 1024, P2 tree on, proxy top-k 28 고정이다.

seed42 21-request에서는 같은 K1=8의 P2-only 대비 최종 P1 tree가 다음과 같았다.

| 설정 | P1 AL | tok/step | target step | target verify | decode TPS |
|---|---:|---:|---:|---:|---:|
| P2-only, P1 chain | 4.02 | 4.39 | 67.09 ms | 61.64 ms | 66.20 |
| P1 global dynamic C3 | 3.89 | 4.22 | 71.10 ms | 62.83 ms | 60.04 |
| P1 full-backbone C3 | 4.04 | 4.21 | 71.49 ms | 62.86 ms | 59.67 |
| **P1 full-backbone C2, N14/M12** | **4.93** | **4.94** | **71.80 ms** | **63.25 ms** | **69.67** |

seed123의 동일 7-request sweep에서도 K1=8/C2/N14/M12는 P2-only의
`P1 AL 4.00 / TPS 62.74`를 `4.81 / 68.88`로 개선했다. M10은 AL 3.52,
N12/M12는 4.40으로 하락했다. C3는 seed123에서는 4.66이었지만 seed42 확대에서
4.04로 불안정해 C2를 선택했다.

K1=9 tree는 두 seed 합산 절대값이 좋아 보였지만, 같은 K1=9 P2-only와 비교하면
`P1 AL 4.92 vs 5.02`, `TPS 69.15 vs 76.81`로 졌다. K1 변화의 이득을 tree
이득으로 세지 않기 위해 최종 권장은 K1=8이다.

### 6.4 timing과 threshold 판정

K1=8/K2=4 full-backbone profile에서 P1은 proxy 도착보다 중앙값 8.60 ms 먼저
끝났고, draft cache merge는 target 다음 요청보다 1.09 ms 먼저 끝났다. 두 overlap
조건을 모두 만족한다.

P1 full-backbone은 threshold를 selector에 사용하지 않으며, fixed CUDA graph에서
threshold는 forward round/width도 줄이지 않는다. 따라서 P1 threshold는 0/0으로
두는 것이 맞다. P2 threshold 0.01/0.01은 별도 P2 정책으로 유지한다.

### 6.5 현재 권장 파라미터

```text
exit_layer=56, K1=8, K2=4
P1 fanout=3, roots_per_position=3
P1 tree=on, allocation=backbone, C=2, N1=14, M1=12
P2 tree=on, budget=15, root_count=10, N2=M2=8
proxy_top_k=28
P1 thresholds=0/0, P2 thresholds=0.01/0.01
max_model_len=4096, extend_draft_rope=true
```

이 값은 seed42 21-request와 seed123 7-request의 방향 확인까지 통과한
**확대 실험 후보**이지 아직
Spec-Bench 560 전체 논문 숫자는 아니다. 전체 실험에서는 P2-only K1=8 arm과 같은
prompt/seed로 paired 비교하고, profiler를 끈 decode TPS와 conditional P1 AL을 함께
보고해야 한다.
