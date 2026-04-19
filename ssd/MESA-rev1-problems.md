# MESA Rev1 Problems

이 문서는 **기존 SSD 공통 코드 전체**가 아니라, **MESA를 구현하면서 새로 생겼거나 MESA 경로에서만 의미 있게 드러나는 문제**만 정리한다.

기준:
- 포함: `mesa_enabled` 경로, `run_mesa_verify_cudagraph`, `draft_runner._build_tree_batch_mesa`, runtime proxy layout, MESA profiling, MESA payload/selection 정책
- 제외: 기존 SSD에도 그대로 존재하던 공통 병목/버그 (`Sampler`, `ParallelLMHead`, 일반 `run_verify_cudagraph`, 공통 `__debug__` print 등)


## 요약

현재 MESA Rev1 구현의 가장 중요한 문제는 아래 5개다.

1. `proxy token selection`이 draft critical path에서 Python loop + GPU→CPU sync를 사용한다.
2. `Policy A`의 dynamic `fan_out_list`를 매 step마다 새 `TreeLayout`으로 만들면서 불필요한 GPU tensor 할당이 발생한다.
3. Rev1이 사실상 `B=1 only`인데 코드에서 강제되지 않는다.
4. `Policy A` underfill 시 일부 proxy slot이 `token id 0`으로 남을 수 있다.
5. target이 계산/전송하는 proxy payload가 현재 Policy A 구현에 비해 무겁다.

그 외에,
- MESA profiling flush가 run 경계에 연결되지 않은 점
- `phase1_build`/`phase2_build` 해석이 비대칭인 점
- MESA용 dead code / hot-path import / padding allocation
도 정리 대상이다.


## Critical

### 1. Proxy token selection: Python loop + GPU→CPU sync

대상:
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1009)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1054)

문제:
- `_select_proxy_sourced_tokens()`와 `_select_proxy_sourced_tokens_policy_a()` 둘 다
  - `.cpu().tolist()` 3회
  - Python `for b in range(B)` / `for pos in range(K)`
  - Python `set` 기반 dedup
  를 사용한다.
- 이 구간은 특히 phase2에서 `proxy_recv_work.wait()` 직후 실행되므로 draft critical path에 그대로 들어간다.

현재 코드:
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1025)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1070)

판단:
- 이 피드백은 맞다.
- `B=1`에선 당장 심각한 절대 시간은 아닐 수 있지만, Rev1 구조상 가장 먼저 줄여야 하는 draft-side CPU 병목이다.
- 특히 Policy A는 dynamic `fan_out_list`까지 들어가 있어 phase2 직후 지연을 더 직접적으로 만든다.

권장 수정:
- 최소 수정:
  - `draft_forked`, `proxy_topk_ids`, `fallback_topk`를 한 번에 하나의 CPU 텐서로 옮긴 뒤 split
- 다음 단계:
  - CPU loop는 유지하되 `.tolist()`는 없애고 tensor/numpy 기반 membership로 변경
- 최종:
  - GPU tensor mask/scatter 기반 dedup/refill로 완전 벡터화


### 2. Dynamic `fan_out_list`마다 `create_tree_layout()` 재생성

대상:
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1172)
- [tree_layout.py](/home/chokwans99/PSD/ssd/ssd/engine/helpers/tree_layout.py:28)

문제:
- Policy A가 target에서 받은 `fan_out_list`로 매 step runtime layout을 생성한다.
- `create_tree_layout()`는
  - `fan_out_t`
  - `fan_out_t_miss`
  - `fan_idx_hit`
  - `fan_idx_miss`
  - `arange_mq`
  - `step_pos_offsets`
  - `step_rope_offsets`
  를 새로 만든다.

판단:
- 이 피드백도 맞다.
- 절대 비용은 phase2 replay보다 작겠지만, dynamic layout을 도입한 Rev1에서 생긴 **순수 MESA 오버헤드**다.
- 특히 반복적으로 비슷한 `fan_out_list` 패턴이 나오는 실험이라면 캐시 이득이 분명하다.

권장 수정:
- `tuple(fan_out_list)` 키로 LRU cache 또는 dict cache
- `step_rope_offsets`, `step_pos_offsets`는 `K`와 `MQ_LEN` 의존성이 있으므로 캐시 가능
- 최소한 `TreeLayout` 생성 자체를 helper로 감싸서 재사용 가능하게 변경


### 3. Rev1은 사실상 `B=1 only`인데 코드에서 강제되지 않음

대상:
- [verifier.py](/home/chokwans99/PSD/ssd/ssd/engine/verifier.py:227)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1171)
- [config.py](/home/chokwans99/PSD/ssd/ssd/config.py:100)

문제:
- target은 `accept_probs[0]`만 사용해서 `h_i`와 `fan_out_list`를 계산한다.
- draft는 그 단일 `fan_out_list`를 배치 전체 proxy pass에 그대로 사용한다.
- 즉 `B>1`이면 첫 번째 시퀀스 기준 allocation이 모든 시퀀스에 강제로 적용된다.

판단:
- 이건 현재 Rev1 구현의 실제 correctness issue다.
- 문서상으론 `B=1 scope`였지만, 코드에서 assert가 없다.

권장 수정:
- `mesa_enabled`면 `assert max_num_seqs == 1` 또는
- 실제 MESA 경로 진입 시 `assert B == 1`


### 4. Policy A underfill 시 `token id 0` slot 가능

대상:
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1066)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1089)

문제:
- `result`를 0으로 초기화한 뒤 실제 채워진 길이만큼만 쓴다.
- `selected`가 `fo`보다 짧으면 나머지 slot은 0으로 남는다.
- 현재 fallback 폭은 `max(max(fan_out_list), async_fan_out)`까지만 뽑는다. [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1066)

왜 생기나:
- proxy 후보와 draft 후보가 크게 겹치거나
- `all-accept` 위치에 budget이 몰리거나
- 특정 position의 `fo`가 실제 unique fallback 후보 수보다 클 때

판단:
- 이건 실제 branch correctness에 직접 영향을 줄 수 있는 버그다.

권장 수정:
- 최소:
  - `assert len(selected) == fo`
- 실전:
  - fallback 폭 확대
  - 부족 시 추가 top-k 재탐색
  - 또는 `fo`를 줄여서 runtime layout과 fill count를 일치시키는 보정 필요


### 5. Target payload가 Policy A 구현에 비해 무겁다

대상:
- [verifier.py](/home/chokwans99/PSD/ssd/ssd/engine/verifier.py:197)
- [verifier.py](/home/chokwans99/PSD/ssd/ssd/engine/verifier.py:209)
- [verifier.py](/home/chokwans99/PSD/ssd/ssd/engine/verifier.py:251)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:990)

문제:
- target은 여전히
  - full-vocab `softmax(p_E)`
  - full-vocab `softmax(p_D)`
  - `topk_probs`
  를 계산하고 전송한다.
- 그런데 현재 draft의 Policy A는 사실상 `fan_out_list + topk_ids`만 사용한다.
- `topk_probs`는 unpack되지만 selection에는 쓰이지 않는다.

판단:
- MESA Rev1이 Policy A 중심이면 이건 불필요한 compute/comm overhead다.
- target-side proxy compute는 이미 MESA 전체 성능의 주요 병목 후보다.

권장 수정:
- Rev1:
  - `fan_out_list + topk_ids`만 전송
- Policy B 도입 시:
  - 그때 `topk_probs`를 다시 활성화


## Medium

### 6. MESA verify padding은 여전히 `torch.cat` 기반 임시 할당

대상:
- [cudagraph_helpers.py](/home/chokwans99/PSD/ssd/ssd/engine/helpers/cudagraph_helpers.py:1079)

문제:
- `run_mesa_verify_cudagraph()`는 padding 시
  - `input_ids`
  - `positions`
  - `slot_mapping`
  - `block_tables`
  - `context_lens`
  에 대해 `torch.cat()`으로 새 텐서를 만든다.

판단:
- 원본 피드백의 `run_verify_cudagraph` 일반론은 기존 SSD 공통 이슈라 여기서 제외한다.
- 하지만 `run_mesa_verify_cudagraph`에 동일한 패턴을 복제한 부분은 **MESA 구현 이슈**로 포함할 가치가 있다.

권장 수정:
- cat 대신 `graph_vars` 버퍼에 직접 write
- pad 영역은 zero/fill 방식으로 처리


### 7. `cache_hits_list` 기반 `fan_idx` 생성이 MESA 경로에도 반복됨

대상:
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1113)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1204)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1210)

문제:
- runtime layout을 쓰는 MESA 경로에서도 여전히
  - Python list / comprehension
  - `torch.cat([hit if ... else miss for ...])`
  패턴이 반복된다.

판단:
- `B=1`에서는 영향이 작지만,
- 이 로직이 MESA runtime layout 경로에 그대로 남아 있는 건 깔끔하지 않다.
- 공통 helper로 빼면 유지보수성과 vectorization 여지가 좋아진다.

권장 수정:
- `build_fan_idx(cache_hits, layout)` 유틸 추가
- `torch.where` 기반으로 vectorize


### 8. MESA용 dead code: `get_forked_recovery_tokens_from_logits(..., mesa_proxy=...)`

대상:
- [async_spec_helpers.py](/home/chokwans99/PSD/ssd/ssd/utils/async_helpers/async_spec_helpers.py:57)

문제:
- 이 분기는 MESA가 `_build_tree_batch_mesa()`를 도입하기 전 흔적이다.
- 현재 MESA 경로는 여기로 오지 않고, `_select_proxy_sourced_tokens_policy_a()`를 직접 사용한다. [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1178)

판단:
- 피드백이 맞다.
- 현재 기준으로는 dead code다.

권장 수정:
- 제거하거나
- 정말 future fallback 용도라면 deprecated 주석 추가


### 9. MESA hot path 내부 import가 반복됨

대상:
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:575)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1151)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1200)
- [verifier.py](/home/chokwans99/PSD/ssd/ssd/engine/verifier.py:188)

문제:
- MESA 추가 코드가 함수 내부 import를 많이 사용한다.
- Python import cache 때문에 큰 절대 비용은 아니지만, hot path 코드 품질은 떨어진다.

판단:
- 성능 issue라기보다는 유지보수 issue
- MESA 추가분에서만 반복된 패턴이므로 문서에 남길 만하다.

권장 수정:
- 파일 상단 import로 정리


### 10. MESA profiling은 helper만 있고 run-level flush wiring이 없음

대상:
- [cudagraph_helpers.py](/home/chokwans99/PSD/ssd/ssd/engine/helpers/cudagraph_helpers.py:1181)
- [llm_engine.py](/home/chokwans99/PSD/ssd/ssd/engine/llm_engine.py:326)

문제:
- `mesa_flush()`는 추가됐지만 실제 `generate()`나 `bench` 경로에는 연결되지 않았다.
- 현재 dump는 target은 `exit()`, draft는 child 종료 시점에서만 수행된다.

판단:
- 이건 MESA profiling 사용성 issue다.
- 기능은 있지만 run 단위 breakdown 수집엔 아직 직접 못 쓴다.

권장 수정:
- `LLMEngine.generate()` 끝에서 target flush
- draft runner에도 run boundary flush command 추가


### 11. `phase1_build` / `phase2_build` label 의미가 비대칭

대상:
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1154)
- [draft_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:1169)

문제:
- `phase1_build`는 사실상
  - draft token selection
  - args build
  만 포함한다.
- `phase2_build`는
  - unpack
  - dynamic layout 생성
  - Policy A token selection
  - args build
  를 모두 포함한다.

판단:
- 측정 자체는 유효하지만, 두 항목을 단순 비교하면 오해가 생긴다.

권장 수정:
- `phase2_build`를
  - `proxy_unpack`
  - `proxy_layout_create`
  - `select_proxy_tokens`
  - `phase2_args_build`
  로 쪼개기


## 현재 피드백 중 제외한 항목

아래는 이번 문서에서 **의도적으로 제외**한다. 이유는 “기존 SSD 공통 코드”이거나 “MESA와 직접 무관”하기 때문이다.

1. `speculator_async._speculation_request`의 per-seq `torch.tensor(bt)`
   - async draft 공통 이슈
   - MESA 전용 문제는 아님

2. `Sampler.forward`의 `temperatures == 0` division
   - 공통 샘플러 이슈
   - MESA 전용 아님

3. `draft_async_prefill`의 EAGLE/non-EAGLE 동일 분기
   - 기존 draft prefill 코드 정리 이슈
   - MESA 본체와 직접 관련 없음

4. 일반 `run_verify_cudagraph` padding cat
   - SSD 공통 verify 경로
   - 다만 `run_mesa_verify_cudagraph`에 동일 패턴이 복제된 부분만 본 문서에 포함

5. `verify.py`, `step.py`의 `__debug__` print
   - 공통 speculative 경로 이슈
   - MESA를 켰을 때 더 거슬릴 수는 있지만, MESA가 만든 문제는 아님

6. `ParallelLMHead.forward`의 `dist.gather + torch.cat`
   - TP logits gather 공통 구현
   - MESA 전용 아님

7. `llm_engine.exit`의 `/dev/shm/sem.*` 전수 삭제
   - 엔진 공통 종료 처리 문제
   - MESA 전용 아님


## 우선순위

### 바로 고칠 것

1. `B=1` assert 추가
2. Policy A underfill 방지
3. proxy selection CPU sync / Python loop 축소
4. `topk_probs` 제거

### 그 다음

5. dynamic `TreeLayout` cache
6. MESA verify padding cat 제거
7. `fan_idx` helper vectorization
8. dead code 제거

### 분석/품질

9. profiling flush wiring
10. `phase2_build` 세분화


## 한 줄 결론

Rev1에서 실제로 중요한 MESA 문제는 **Policy A runtime path의 CPU-side 오버헤드와 correctness guard 부족**이다.

특히:
- `B=1` 미강제
- underfill 가능성
- proxy selection Python loop
- step마다 새 layout 생성

이 네 개는 MESA Rev1의 핵심 품질 문제로 봐야 한다.
