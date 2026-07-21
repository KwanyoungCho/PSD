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
