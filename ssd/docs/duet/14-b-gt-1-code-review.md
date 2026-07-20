# 14 — B>1 구현 코드 리뷰 (M1-M6)

**날짜**: 2026-07-19. 리뷰 범위: 커밋 baa011c (M1), af93cde (M2),
73fe75a (M3), 2cd2176 (M4), 9528366 (M6)의 엔진 변경 사항을
docs/duet/13(설계 + 단계별 계획 + verdict)과 대조하여 검토했다. 대상 파일:
`ssd/engine/{verifier,draft_runner,speculator_async,scheduler}.py`,
`ssd/engine/helpers/{cudagraph_helpers,tree_layout,runner_helpers}.py`,
`utils/verify.py`, `config.py`. 라인 번호는 리뷰 시점의 HEAD 기준이다
(cd460d9의 parent인 f543c24).

**핵심 결론**: 새로운 고심각도(high-severity) 정확성 버그는 발견되지
않았다. robustness(견고성) 수정 1건을 적용했다(R1 — M6 버그를 숨겼던
것과 정확히 같은 부류인, `python -O`에서 제거되는 assert 문제).
그 외는 전부 LOW/INFO이며, 구체적인 패치를 명시한 채 보류(defer)했다.

검증: `tests/test_b_gt1_m{1,2,3,4}.py` +
`test_b_gt1_m6_verify_window.py`를 개별 실행 — 수정 적용 전과 후 모두
38/38 OK. 강화된 guard는 `python -O` 환경에서도 추가로 검증했다
(pre-M6 형태의 어긋난(misaligned) batch에서는 발화하고, 정렬된
batch에서는 통과).

## 발견 사항 표

| # | 축 | 심각도 | 위치 | 발견 내용 | 조치 |
|---|---|---|---|---|---|
| R1 | robust | **MED** | runner_helpers.py:88 | `num_cached_tokens == pos0` assert가 `python -O`에서 제거됨 — 이 guard는 조용한(silent) 출력 손상을 막는 것으로, M6 버그가 정확히 여기서 숨어 있었다 | **적용됨** (cd460d9): 명시적 `raise AssertionError`, step당 int 비교 ~B회 |
| C1 | correct | OK | 리뷰한 모든 파일 | B>1에서 도달 가능한 B=1 / seq-0 인덱싱 잔재 없음 (아래 전체 sweep 참조) | 없음 |
| C2 | correct | LOW | verifier.py:454-456 + draft_runner.py:1564-1610 | score가 0인 항목의 topk tie-break(동점 처리)가 짧은 seq의 wire에 `chosen_pos > vk_i`인 항목을 올릴 수 있음; 그 seq의 양수-score 항목이 total_budget보다 적으면 selector가 이를 채택 → phase-2 budget이 도달 불가능한 position으로 새어 나감. 성능에만 영향 (`k_idx > vk_i`인 key는 절대 요청되지 않음), B>1 혼합 batch에서만 발생 | 문서화; B>1에서 L_p2가 악화될 때만 재검토 |
| C3 | correct | INFO | verifier.py:388-392, 444-447 | 짧은 seq의 전원-실제-accept 이벤트(h 질량이 `vk_i < K`인 col에 위치)에서, 후보를 position K가 받는 `pE_K` 방식의 전체 분포 처리 대신 PADDED col `vk_i`의 residual에서 뽑음 (p_D = zero logits에서 나온 uniform; token id 0 제외). 수치적으로는 p_E의 top-k와 거의 같음 (uniform ≈ 1/V); v1에서는 허용 가능 | 문서화; v2에서 per-seq vk_i 위치의 pE gather 고려 |
| C4 | correct | OK | — | edge case 추적 결과 모두 깨끗함: 전원-short batch, 전원-miss batch, 빈 cache, B=8, step마다 B가 바뀌는 경우, preemption 재입장, B=0 (아래 상세) | 없음 |
| C5 | correct | OK | speculator_async.py:13-47, utils/verify.py:140-172 | M6 × non-DUET/EAGLE: non-DUET wire는 균일한 `valid_k = K`를 실어 나르므로 extend-by-vk_max ≡ extend-by-K로 bit 단위 동일; sync speculator는 `valid_k=None`을 전달 → clamp + `_k_real` residual 검사가 pre-M6 형태로 환원; EAGLE extend 데이터는 미변경 | 없음 |
| E1 | perf | OK | M1-M6가 건드린 영역 | sync(동기화) 감사: 문서화되지 않은 새 GPU→CPU sync 없음. `extend_seqs_for_verify`는 B>1에서 sync를 2B → 2로 오히려 감소시킴; `.max().item()` 두 곳은 문서화된 교체임 (verifier:114는 `torch.unique` 대체, draft:546은 `valid_k[0].item()` 대체) | 없음 |
| E2 | perf | LOW | verifier.py:113-116 | verify 경로의 `valid_k.max().item()` sync는 공짜로 제거 가능: speculator가 `speculations.size(1) == vk_max+1`을 보장하므로 `_step_lookahead = speculate_result.speculations.size(1) - 1`은 sync 없이 동일한 값 | 보류 (아래 패치 참조; hot path이므로 GPU A/B 측정 후 적용) |
| E3 | perf | LOW | draft_runner.py:1356 | `_irecv_duet_proxy`가 draft critical path에서 매 step 새 `2·B·wire_N` int64 버퍼를 할당 (glue 이전에 post됨) | 보류: B 변경 시 resize하는 persistent 버퍼로 |
| E4 | perf | LOW | cudagraph_helpers.py:324-328 | bucket이 아닌 B (3,5,6,7): `active_cache_hits_list` 길이 < 패딩된 `wrapper_bs` → `cache_hits[:B].tolist()` fallback으로 빠져 phase당 step-0마다 sync 1회 추가. B ∈ {1,2,4,8}은 영향 없음 | 보류: 대신 threading 지점에서 리스트를 wrapper_bs까지 패딩 |
| E5 | perf | OK | cudagraph_helpers.py:373-410 | step-0 mask 빌드는 O(B·MQ·ctx) — B에 선형, O(B²) 없음; per-seq nested glue 빌드는 `np.repeat` O(B)개짜리 리스트 (B=4에서 측정치 +0.2 ms, 예측과 일치). 참고: phase 사이 MQ_LEN이 바뀔 때 `cache.clear()`가 호출되므로 `_cached_fol`과 무관하게 glue mask는 매 phase-2 step-0마다 재빌드됨 (기존부터 있던 동작) | 없음 |
| E6 | perf | INFO | draft_runner.py:1406,1433 | fork selector들이 step마다 `[B,P,V]` logits를 `clone()` (~2.5·B MB) — 기존부터 있던 비용이나 이제 B에 비례해 커짐 | 기록만 |
| E7 | perf | INFO | draft_runner.py:421,492 | `match.float().argmax(dim=1)`이 DUET step당 두 번 계산됨 (각각 [B,T], sync 없음) — fill 블록에서 `_hit_idx`를 재사용 가능 | 보류, 사소함 |
| D1 | dead | INFO | verifier.py:421 | `proxy_fan_out_total`은 Policy-B 통합 이후 dead code (기존 코드, M1 이전부터 존재 — surgical-change 원칙에 따라 그대로 둠) | 언급만 |
| D2 | dead | INFO | verifier.py:407-415 | `cache_hits is not None and not config.jit_speculate` miss 분기는 도달 불가: `_compute_and_send_proxy`는 DUET에서만 실행되고 DUET config는 `jit_speculate=True`를 강제함 (기존 코드) | 언급만 |
| D3 | dead | INFO | draft_runner.py:258-260 | 낡은 주석 "B=1이므로 N = max_mq"가 바로 아래의 (이미 B-aware인) `max_N = max_num_seqs * max_mq` 라인과 모순됨 (기존 코드, rev1 시절) | 언급만 |
| D4 | dead | INFO | draft_runner.py:1645-1650 | `metadata_ints`의 F (`_f0`, M3의 nested-list 특수 처리 포함)는 `_decode_tree`가 unpack하지만 layout 경로에서 실제로 소비되지 않음 | 유지 (3줄; F를 제거하면 non-DUET 경로와 공유하는 payload 형태가 바뀜) |
| D5 | dead | INFO | speculator_async.py:194 | fallback `int(valid_k.max().item())`은 도달 불가 (해당 분기를 탈 때 `_vk_max`는 항상 non-None) — M6에서 남은 무해한 방어적 dead code | 언급만 |
| D6 | dead | OK | — | 엔진 코드에 M5 시절 debug print/분기 없음 (M5는 측정 전용이었음; TRACE/PROFILE 지점은 env 변수로 gate되며 기존부터 존재) | 없음 |
| D7 | dead | OK | tree_layout.py:48 + 소비자 2곳 | `fan_idx_per_seq` dual-mode의 단순화 여부 검토 — 결론: 유지 (근거는 아래) | 없음 |
| R2 | robust | LOW | model_runner.py:1111, draft_runner.py:1723 | `_step_lookahead/_step_valid_k in (K1, K2)` scalar assert가 `-O`에서 제거됨; 위반 시 잘못된 CG bucket으로 dispatch됨. 다만 R1과 달리 실패가 대체로 시끄럽게(loudly) 드러남 (CG shape mismatch) | 보류-권장: 저렴한 hard raise |
| R3 | robust | OK | draft_runner.py:1548-1550, 1577-1585, 1636-1638 | selector의 `__debug__` guard들 (chosen_pos 범위, Fix-③ per-seq `take.sum == total_budget`, fan_idx 길이)은 GPU sync이므로 debug 전용이 올바름. Fix-③의 불변식은 config 차원에서 보장됨 (아래 증명 스케치) — 이것이 잡을 조용한 per-seq 어긋남은 well-formed wire에서는 발생 불가 | debug 전용 유지 |
| R4 | robust | LOW | draft_runner.py:930 | `_glue_decode`의 B-일치 assert (`-O`에서 제거됨); 불일치는 wire 손상을 의미하지만 하류의 `.view`가 시끄럽게 실패함 | 현행 유지 |

## 적용된 수정

**R1 — 커밋 cd460d9** `fix(duet): harden verify-window pos0 guard`.
`prepare_decode_tensors_from_seqs`(verify 경로)가 이제
`seq.num_cached_tokens != pos0`일 때 명시적으로 `AssertionError`를
raise하므로, guard가 `python -O`(모든 bench 실행 조건)에서도
살아남는다. 이것이 바로 M6를 숨겼던 정확한 부류다: 어긋난 window는
*조용한* 손상(logits_p row가 밀리고 stale position에서 recovery가
일어남)이라 smoke 지표에는 보이지 않고 sweep에 가서야 드러난다.
`assert` 문 대신 명시적 raise를 쓰는 이유는, `python -O`가 `assert`를
바이트코드에서 통째로 제거해 버리는 반면 일반 `if` + `raise`는 항상
실행되기 때문이다. 예외 타입을 동일하게 유지했으므로
`test_b_gt1_m6_verify_window.test_pre_m6_extension_slides_short_row_window`
(assertRaises)는 그대로 통과한다. 비용: verify step당 seq마다 Python
int 비교 1회 — noise 수준. 검증: 적용 후 unit test 38/38 OK; `-O`
런타임 체크가 pre-M6 형태의 batch에서는 발화하고 정렬된 batch에서는
통과함을 확인.

## 축 1 상세 — B=1 가정 sweep

리뷰 대상 파일들의 모든 `[0]` / seq-0 / scalar-collapse 지점을 다음과
같이 분류했다:

- **Config로 gate되어 B>1에서 도달 불가** (M4의 `max_num_seqs > 1`
  hard `ValueError`): `policy_b_from_candidates`의 single-seq 호출
  (verifier.py:310-316), raw-proxy pack의 `exit_logits[0]`
  (verifier.py:349-373), `_policy_b_from_raw_proxy`의 `out_logits[0]`
  (draft_runner.py:1401), exit-replica / topm의 `orig_bs == 1` assert
  (cudagraph_helpers.py:1317, 1359). 런타임 assert들은 `-O`에서
  제거되지만 실질적인 방어선은 config gate다 — 올바른 계층화
  (config에서 한 번 hard-error로 막으면, 내부의 저렴한 assert는
  제거되어도 안전하다는 구조).
- **B==1 전용 profile 분기** draft_runner.py:551-562 — 명시적으로
  `B == 1`로 분기하며, batched else 분기가 존재.
- **`layout.fan_out_list[0]`** 지점들: draft_runner.py:1740/1746은
  per-seq 데이터가 아니라 seq 간 균일(UNIFORM)한 phase-1 config
  리스트를 다룸; :1648-1650은 dead metadata F(D4)에 공급.
- **Batch-shape 배선**: wire 길이 (2·B·wire_N send =
  request-meta-B irecv, 같은 step, 같은 seq들), `[B,wire_N]` unpack
  view, per-seq accept clamp, per-seq h-padding, per-seq fan_out /
  fan_idx / mask — 전부 end-to-end로 일관성 확인.

**Edge case (C4) 추적 결과**:

- *모든 seq가 short*: `vk_max = K2` → k2 glue/verify bucket +
  `split_k1_short` layout; champion의 short prefix `[2]*5`는 균일 →
  uniform selector 분기; `K_rank = K2`가 `chosen_pos ≤ K2`를 유지.
- *전원-miss batch / 빈 cache*: `_miss_vk`로 JIT-all
  (`SSD_DUET_JIT_SHORT` 하에서는 K2), overwrite pass 없음; 유일한
  stale-KV 코너 케이스 (all-short dispatch + `_jit_K = K1 > vk_max`)는
  도달 불가: JIT_SHORT 없는 miss row는 `vk_max = K1`을 강제하고,
  전원-short 전원-hit batch는 JIT를 아예 실행하지 않기 때문.
- *step마다 B가 바뀜 (preemption/finish)*: 모든 `[B,·]` stash가
  step마다 재빌드 또는 재할당됨 — speculator handshake 버퍼 +
  speculations 버퍼는 B가 바뀌면 realloc (speculator_async.py:163,
  230); draft tree cache는 spec 요청마다 reset+재빌드
  (draft_loop:1975)되므로 떠난 seq의 stale row는 최대 1 step만
  존재하고 단조 증가하는 `seq_id`로 key됨; `split_k2_layout`의 변형된
  `[B,P]` fan tensor는 모든 소비자보다 먼저 덮어써짐; preempt 후
  재입장한 seq의 요청 key는 `k_idx = -2`
  (`last_spec_step_accepted_len = -1`)라서 `fan_idx ≥ 0`과 절대 매치될
  수 없음 → clean miss 보장. CG bucket 패딩 (wrapper_bs > B)은
  cache_hits를 0으로 채우고 마지막 실제 seq의 glue block / block
  table을 재사용; 패딩된 출력은 slot_map -1로 폐기.
- *B=8*: bucket 축이 커버함 (M4의 cap ≤ 8); handshake/spec 버퍼는
  `max_num_seqs` 크기로 잡힘.
- *B=0*: `SpecDecodeStep.decode`에서 필터링됨; `extend_seqs_for_verify`
  와 slicing 블록은 방어적으로 이를 허용.

**M6 × EAGLE/non-DUET (C5)**: async response wire는 항상 `valid_k`를
실어 나른다 (non-DUET은 EAGLE 포함 균일한 `K`). 따라서 `extend by
vk_max` ≡ `extend by K`이고 `num_draft_cached_tokens += K+1` —
pre-M6와 bit 단위로 동일하면서, 대체된 per-seq sync들은 사라졌다 (E1).
sync speculator의 `SpeculateResult.valid_k`는 기본값 None → `verify()`
clamp가 생략되고 `_k_real = K`가 되어 예전의 `accept_until < K` 검사를
정확히 재현한다. EAGLE의 extend-count/acts 배선은 M6가 건드리지
않았다.

## 축 2 상세 — sync/할당 감사

건드린 영역의 step당 GPU→CPU sync, HEAD vs pre-M1:

| 지점 | pre-M1 | HEAD | 판정 |
|---|---|---|---|
| verifier lookahead | `torch.unique(valid_k)` + item | `valid_k.max().item()` | 교체 (E2: 완전 제거 가능) |
| draft dispatch scalar | `valid_k[0].item()` | `valid_k.max().item()` | 교체 |
| speculator extend | B×`.item()` + B×`.tolist()` | 1×`.tolist()` + 1×`.tolist()` | B>1에서 **개선** |
| hit/miss 분기 | `.any()`/`.all()` | 동일 | 변화 없음 |
| phase-2 layout | 1×`fan_out.tolist()` | 동일, 이제 [B,P] | 변화 없음 (sync 1회) |
| glue mask step-0 | `context_lens.tolist()` (+`cache_hits.tolist()` fallback) | 동일 | 변화 없음; E4 fallback은 non-bucket B에서만 발화 |
| selector | 없음 (debug assert로 gate됨) | 동일 | 변화 없음 |

새로 생긴 step당 할당은 전부 작고 B에 비례한다: irecv 버퍼 (E3,
B=8에서 ≤1.5 KB), `_pad_cols` arange [K], h `[B,K+1]`, per-seq
fan_idx 빌드 (`arange.repeat(B).repeat_interleave`), nested
`fan_out_list` tolist (B×≤10 int), per-seq numpy glue block. 숨은
O(B²)는 어디에도 없다 (E5): per-b 루프는 기존부터 있던
mask/kv-meta 빌드뿐이며, 각각 O(B)이고 per-seq 본문은 B와 무관하다.

### E2 보류 패치 (verifier.py:113-116)

```python
# speculations is sliced to [B, vk_max+1] by SpeculatorAsync.speculate
# (M1 slicing block) — width IS vk_max+1, no sync needed:
if speculate_result.valid_k is not None and config.duet_phase1_k is not None:
    _step_lookahead = speculate_result.speculations.size(1) - 1
```

구성상(by construction) 동일한 값이다 — `.max().item()`처럼 GPU에서
값을 읽어오는 대신, 이미 CPU에 있는 tensor의 shape 메타데이터에서 같은
정보를 얻는 방식이라 GPU 동기화 지점 자체가 사라진다. 이 패치는
B=4가 병목인 target에서 CG dispatch 직전 verify 경로의 마지막 sync를
제거한다. 미적용 사유: hot path 성능 변경은 별도의 GPU A/B 측정을
거쳐야 하고, 현재의 sync는 문서화된 budget 안에 있기 때문.

## 축 3 상세 — `fan_idx_per_seq` dual-mode (D7)

결론: **유지**. 이 flag의 setter는 정확히 하나
(`_update_phase2_layout_inplace`, 유일한 런타임 layout 변경자)이고,
소비자 분기는 2줄짜리 두 곳뿐이다 (`_build_tree_decode_args_for_layout`,
`_merge_and_populate_cache`). 두 모드를 하나로 합치려면 (a) 정적인
phase-1 layout까지 per-seq로 만들거나 — 설계상 seq 간 균일한 것을
바꾸는 순수한 churn — (b) `fan_idx_hit.shape[0] == B*MQ_LEN`으로
per-seq 여부를 추론해야 하는데, 이는 암묵적이고 B·MQ 값이 우연히
일치하는 경우 깨지기 쉽다. 즉 명시적 flag 하나가 가장 단순하면서도
올바른 인코딩이다. 또한 split_k2의 hit==miss aliasing
(`fan_idx_miss = fan_idx_hit`, `fan_out_list_miss = fan_out_list`)
덕분에 nested glue 빌드가 block 리스트 하나를 공유할 수 있다
(cudagraph_helpers.py:350-352).

## 축 4 상세 — assert 정책

이 코드에서 `-O`로 제거되는 assert들에 대한 권장 정책:

- **Hard-check (raise)**: 위반이 *조용한 손상*이면서 검사 비용이
  CPU scalar 수준인 guard. 즉, 어차피 공짜에 가까운 검사라면 `-O`에서
  사라지는 assert 대신 항상 실행되는 raise로 두는 것이 이득이다.
  적용: R1 (pos0 window). 다시 손댈 일이 생기면 후보: R2의
  bucket-dispatch scalar들 (위반이 대체로 시끄럽게 드러나므로 지금은
  미적용).
- **`__debug__` 전용 유지**: GPU를 sync시키는 모든 것
  (selector Fix-③의 per-seq 합, chosen_pos 범위, fan_idx 길이,
  `_construct_tree_decode_args`의 N 체크). GPU sync는 step당 비용이
  커서 hard-check로 승격하면 벤치마크 성능을 해치기 때문이다.
  Fix-③의 불변식은 config 크기 설계로 수학적으로 보장된다: 한
  position의 wire 항목들은 서로 다른 token을 가지므로 dedup 손실 ≤
  Σp fan_out_p = p1_sum이고, `wire_N = total_budget + p1_sum + 2`
  (+ short seq를 위한 `ceil(wire_N/(K_min+1))` top_k 하한) 덕분에
  seq당 ≥ total_budget + 2개의 유효 항목이 남는다 — 이 guard가 잡을
  B>1 어긋남은 well-formed wire에서는 나올 수 없고 wire 자체가
  malformed여야 하므로, 도달 가능한 엔진 상태가 아니다.

## 보류된 권장 사항 (우선순위순)

1. **E2** — verify 경로의 `.max().item()`을
   `speculations.size(1) - 1`로 제거 (위 패치) + B=4에서 GPU A/B.
2. **R2** — 두 곳의 `in (K1, K2)` bucket-dispatch 체크를 hard-check로
   강화 (scalar 비교; R1 정책과 대칭).
3. **E4** — threading 지점에서 `active_cache_hits_list`를
   `wrapper_bs`까지 패딩해 non-bucket B의 step-0 sync 제거
   (B ∈ {3,5,6,7} cell을 sweep할 일이 생길 때만 의미 있음).
4. **E3** — draft 쪽 persistent irecv 버퍼 (B 변경 시 resize).
5. **E7** — cache-overwrite gather에서 `_hit_idx` 재사용.
6. **C3** — v2: short row의 전원-accept 후보 집합에 대해 `vk_i`
   위치의 per-seq `pE` gather (향후 per-seq vk_max-padding 제거 —
   verdict가 지목한 B=4의 지배적 비용 — 와 자연스럽게 짝을 이룸).

기존부터 있던 dead code는 기록만 하고 의도적으로 건드리지 않았다
(surgical-change 원칙): D1 (`proxy_fan_out_total`), D2 (DUET 하의
non-jit miss 분기), D3 (낡은 B=1 버퍼 주석).
