# Tree 실행 최적화와 지원 범위 정리 (2026-08-07)

## 결론

- `coverage` 정책은 아직 삭제하지 않는다. 현재의 동적 `eagle` 정책은
  정상 실행되지만, 저장된 다중 seed 결과에서 P2의 실제 기여가 coverage보다
  높다는 증거가 없다. coverage는 정확성 기준이자 fallback으로 유지한다.
- 정책과 무관한 실행비용부터 줄였다. P2의 선택 확률, 자식 token, 노드 순서,
  target 수락 규칙은 바꾸지 않았다.
- B=1, temperature>0의 coverage/eagle/adaptive 실행은 같은 최적화된
  CUDA-graph 경로를 사용한다.

## 이번 변경

### P2 draft

매 라운드마다 여러 개의 작은 `scatter/gather/one_hot/cumsum` GPU kernel로
수행하던 노드 삽입과 출력 기록을 하나의 Triton kernel로 합쳤다. arena와
출력 버퍼 초기화도 하나의 kernel로 합쳤다. 확률 계산과 정렬은 기존
PyTorch 연산을 그대로 사용하므로 트리 선택 점수는 바뀌지 않는다.

입력 준비에서는 다음 임시 할당을 제거했다.

- NumPy glue를 위해 매번 만들던 임시 CUDA tensor
- block table 전체를 복사하기 전에 하던 불필요한 zero-fill
- slot/context dtype 변환용 임시 tensor
- canvas page-id용 clone

저장된 동일 구성 profile과 비교한 P2 graph 평균은 12.37ms에서 12.08ms로
약 0.29ms 감소했다. 이 수치는 짧은 profile 비교이므로 전체 TPS 개선량으로
해석하지 않는다.

### Target tree verify

Target의 tree CUDA graph는 이미 캡처되어 있으므로, replay 직전의
FlashInfer `plan()` 호출은 graph의 kernel 배치를 바꿀 수 없다. 실제로 필요한
것은 graph가 읽는 page 수, 마지막 page 길이, page id, mask buffer 갱신이다.
이를 직접 갱신하도록 바꿨다.

Mask도 `[rows, kv_len]` 불리언 행렬을 만든 뒤 pack하지 않고, 공통 prefix와
각 노드의 조상/self bit만 packed buffer에 직접 기록한다.

실모델 profile의 P2 hit 기준:

| 항목 | 이전 | 변경 후 |
|---|---:|---:|
| 전체 verify 준비 | 1.72ms | 0.99ms |
| attention 준비 | 0.75ms | 0.17ms |
| mask 준비 | 0.57ms | 0.41ms |

Page 수 1/2/3, 마지막 page 길이 1/15/16에서 직접 buffer 갱신과 기존
`plan()` 경로의 attention 출력은 비트 단위로 일치했다. 70B target + 1B
draft 실모델에서도 283회 replay, fallback 0, 오류 0으로 완주했다.

## 시작 비용

모든 P2 page bucket을 미리 준비하는 비용은 이번 실모델에서 7.18초,
약 1014MiB였다. 이 시간은 모델 초기화 중에 발생하며 decode TPS와
요청별 target/draft step 시간에는 포함되지 않는다. 다만 다음에는 포함된다.

- 프로세스 cold start 시간
- GPU 상주 메모리
- 짧게 실행하고 종료하는 benchmark의 총 wall time

따라서 서비스 steady-state 성능에는 미포함이지만 운영 비용이 없는 것은 아니다.

## 동적 정책 상태

동적 `eagle` 정책은 융합 kernel을 포함한 실모델 smoke에서 정상 완주했다
(replay 275, fallback/error 0). 이 짧은 실행의 P2 accepted length는 1.91이었고,
같은 날 coverage smoke의 2.03보다 높지 않았다. 표본과 profiling 조건이 달라
최종 우열 판정에는 쓰지 않지만, coverage를 지금 삭제할 근거도 없다.

정책 전환 조건은 다음처럼 고정한다.

1. 같은 계산량과 root 수를 사용한다.
2. 같은 seed/dataset에서 coverage와 순서를 교대한다.
3. P1 hit/accepted length를 악화시키지 않는다.
4. `P2 hit × (P2 accepted length + 1)`이 coverage보다 높아야 한다.
5. 세 seed wall-time TPS에서도 이겨야 한다.

이 조건을 통과한 뒤에만 coverage를 deprecated로 바꾸고, 한 번 더 release 동안
fallback으로 남긴 뒤 삭제한다.

## B>1 지원 설계

가능하지만 단순히 B=1 gate를 지우면 안 된다. 각 sequence마다 다음 상태가
독립적으로 필요하다.

- root와 동적 topology
- page table, slot, context length
- tree verify mask와 row 수 bucket
- cache key와 수락 경로

권장 순서는 B=1 graph를 B개 직렬 replay하는 것이 아니라, 먼저 topology와
verify row를 `[B, ...]` 고정 버퍼로 바꾸고 row 수 합계에 따른 CUDA graph
bucket을 추가하는 것이다. 그 뒤 sequence별 parent 관계를 block-diagonal
mask로 검증한다. B=2 parity부터 시작해 B=4/8로 확장한다.

## temperature=0 지원 설계

현재 비복원 확률 sampling은 temperature=0에서 draft 분포가 한 token에만
질량을 가지므로 두 번째 자식을 정의할 수 없다. 이 gate를 제거해서는 안 된다.

Greedy tree는 별도 규칙으로 지원해야 한다.

1. 각 부모에서 draft logits의 top-C 서로 다른 token을 자식으로 둔다.
2. target의 argmax token과 일치하는 자식 하나만 따라간다.
3. 일치하는 자식이 없으면 그 위치에서 target argmax를 recovery token으로 낸다.
4. 확률 residual과 형제별 coin flip은 사용하지 않는다.

이는 temperature>0의 lossless residual verifier와 다른 verifier이므로 별도
테스트와 graph bucket으로 구현한다. 우선순위는 현재 B=1 temperature>0 경로의
최종 성능 판정 이후, B>1, greedy tree 순서다.
