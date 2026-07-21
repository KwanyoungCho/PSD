# overlap_profile — B=32 draft/target overlap 타임라인 검증 (2026-07-21)

**질문**: (1) draft가 target에 실제로 overlap되고 있는가 (평균이 아니라
타임라인으로), (2) matched-shape 격차 +27ms의 정확한 거처.

**설정**: B=32 ns=32 out=128, `SSD_PROFILE_DUET=1`, GPU 0-4, 포트 13600+.
셀 2개: DUET k2x2_d4p1 (265.85 tok/s), C k2f2 (289.38 tok/s). 타임라인
그림은 `plot_timeline.py`(두 프로세스의 host-monotonic wall_ns를 한 축에
정렬, 정상상태 중간 4 step 윈도우) → `figs_timeline_duet_k2x2.png`,
`figs_timeline_c_k2f2.png`. 프로파일 JSON은 프루닝 정책상 미커밋.

## 결론 1 — overlap은 실재하고 거의 완벽하다

| | draft 실작업 (윈도우) | target-busy 아래 숨은 비율 |
|---|---|---|
| DUET k2x2 | 272 ms / 4 steps (≈68 ms/step) | **98.8%** |
| C k2f2 | 137 ms / 4 steps (≈34 ms/step) | **98.7%** |

bench의 "Avg draft step time"(DUET 204 ms)은 대기 포함 수치다 — 실제
GPU 작업은 68 ms/step이고 그 99%가 target verify와 동시 실행된다.
그림에서 draft 레인의 작업 묶음(cache fill → glue → phase-1 → phase-2)
이 target의 verify 구간 안쪽에 위치하는 것으로 시각 확인.

overlap이 깨지는 유일한 지점은 step 경계의 `target_spec_wait`
(응답 wire + glue rendezvous 대기)로, **양쪽이 동일하게 ~14 ms/step
지불한다** (DUET 14.56 vs C 14.22) — B=4 시절 ~3 ms에서 커진, 양
시스템 공통의 B-스케일 비용.

## 결론 2 — +27ms의 거처: proxy 블록이 아니라 exit-이전 CG 구간

target rank0 라벨별 분해 (정상상태 중간 1/3 구간 평균, ms/step):

| 라벨 | DUET k2x2 | C k2f2 |
|---|---|---|
| verify 본체 | graph_pre **185.37** + graph_post 63.13 = 248.50 | verify_replay 223.33 |
| exit_logits | 1.30 | — |
| **proxy_compute_send** | **0.81** | — |
| final_logits | 1.06 | (replay에 포함) |
| verify_setup | 0.24 | — |
| verify_sample_accept | 2.99 | 3.10 |
| target_spec_wait | 14.56 | 14.22 |
| 합 | 269.53 | 240.72 |

레이어당 속도: C 223.33/80 = **2.79 ms/layer**. DUET exit-이후
63.13/24 = **2.63** (C보다 빠름 — 정상). DUET exit-이전 185.37/56 =
**3.31** (+19% vs C). C 속도로 환산한 exit-이전 기대치는 156.3 ms →
**초과 +29.1 ms가 전부 graph_pre 안에 있다.**

즉 4db373b 정정이 지목했던 "mid-verify proxy 블록(수집+계산+송신)"은
**0.81+1.30 ≈ 2 ms로 무죄**다. 격차의 실체는 exit-이전 CG 세그먼트가
같은 GEMM인데 레이어당 19% 느리게 도는 것이며, 후보 원인은 (a) rank0의
DUET 전용 준비/언팩으로 인한 TP rank 진입 시차가 첫 collective에서
흡수되어 graph_pre에 계상, (b) duet_verify CG family의 capture 품질
(exit hidden/residual 경계 버퍼 포함) 차이. 판별에는 rank1-3 프로파일
또는 그래프 내부 레이어 단위 계측이 필요 — 미해결로 남긴다.

**레버 함의**: exit_topm_gather/proxy_on_draft의 B>1 배치화는 ~2 ms
짜리 표적이므로 기대 회수가 작다. 27 ms를 노리려면 graph_pre 자체
(rank 진입 동기화 / capture 방식)를 파야 한다.

## 추가 (같은 날, 사용자 지적 반영) — 정식 도구 재렌더 + "여유"의 정체

지적 두 가지가 모두 옳았다: (1) 타임라인 정식 도구가 이미 있었다
(`bench/plot_duet_aligned_timeline.py` — step 단위 상세 뷰, 자식 span
+ 응답 인과 화살표 + clock-drift 보정). 본 실험의 프로파일 dump 자체는
기존 `SSD_PROFILE_DUET` 그대로이며, 새로 만든 것은 매크로(다중 step)
플롯 스크립트뿐이다. 정식 도구 렌더:
`duet_k2x2_prof/timeline_step121_mixed.png` (+ hit_k1/miss 대표),
`c_k2f2_prof/timeline_step121_mixed.png`.

(2) 첫 매크로 그림의 "draft 여유"는 과장이었다 — proxy_wait를 idle과
같은 회색으로 뭉뚱그렸기 때문. 실측 분해 (DUET k2x2 @B=32, ms/step):

| draft 레인 구성 | ms/step | 성격 |
|---|---|---|
| 실작업 (P1 29.7 + P2 14.5 + glue 17.1 + fill/prep ~12) | ~75 | target 아래 은닉 (99%) |
| **proxy_wait** | **136.7** | **구조적 블로킹** — P2는 target이 exit layer(56/80, graph_pre 185ms)에 도달해야 시작 가능. 자유 슬랙이 아님 |
| 진짜 idle (요청 사이) | ~95 | 자유 슬랙 |

step 상세(정식 도구, step 121 mixed): P1은 step 초반에 끝나고, draft는
proxy_wait로 ~140ms 블로킹, proxy 도착 후 P2(~17ms)가 graph_post와
겹쳐 돌고, 응답은 target이 다음 요청을 내기 전에 준비된다(spec_wait
14.6ms만 노출 — mixed status의 부분-JIT respond 비용; hit_k1 step은
3.0ms, miss는 54.4ms).

**함의**: (a) phase1_replay가 forward당 ~15ms (384 rows) — "draft
forward는 latency-bound라 공짜" 가정은 B=32에서 무효; (b) proxy_wait
137ms는 exit를 앞당기면(56→더 얕게) 풀리는 예산이다 — 2026-07 초의
"early-exit을 당겨 overlap을 만든다" 아이디어가 B=32에서 정량적
표적(137ms)을 얻었다. 단 proxy 품질(α̂ 정확도) 하락과의 트레이드오프.
