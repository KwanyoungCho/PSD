# Verdict 실험 — B=4 격차는 버그인가 물리인가? (2026-07-18)

**질문** (m5_sweep/RESULTS.md 정정 결론과 docs/duet/13 §M6의 후속):
M6 verify-window fix 이후에도 DUET은 B=4에서 SD-best C에 −21.5%
뒤진다(118.00 vs 150.31). 다음을 입증 또는 반박하라: "남은 B>1
버그는 없다 — 격차는 draft-shape + vk_max-padding 시간 비용이다",
그리고 인과 분해(causal decomposition)를 산출하라.

**셋업**: HEAD 9528366, GPUs 0-4 (target TP4 on 0-3, draft on 4),
ns=20 out=256 in=512 temp 0.7 seed 42, `--all`, B=4. 무관한 vLLM이
GPUs 6-7에서 전 구간 유휴(m5/m6과 동일 regime).

- **Exp1** `prof_b4/`: 정정된 챔피언 (E9K24_jit, m6_fix duet_b4
  인자) + `SSD_PROFILE_DUET=1`, port 12920. rc=0, 126.69 tok/s.
- **Exp2** `fat7/`, `fat5/`: B>1-shape 재조율 probe, PROFILE=0,
  ports 12921-2 (§3 참조). 둘 다 rc=0, Traceback 0건.
- 분석: `analyze_prof.py` (이 디렉터리; 전체 테이블은
  `analyze_prof_out.md`), 추가로 `tax_decomposition/analyze.py --base
  ../b_gt1/verdict --cells prof_b4:4` (`analyze_tax_prof_b4.md`).

## 결론 요약 (Verdict up front)

**남은 B>1 버그는 없다.** 모든 profile label이 자신의 구조적 row
모델과 노이즈 범위 안에서 일치한다(§1). 챔피언-shape의 B=4 격차는
다음과 같이 분해된다: **vk_max로 padding된 verify 폭 ≈ 17-21
ms/step (지배 항, knock-on 포함 시 시간 쪽 격차의 ~2/3) + 기존의
~−7% 토큰 결손(B-불변, B=1과 동일) − 실재하지만 작은 miss-stall
우위(DUET에 유리한 +1..+5 ms/step)**. "13번의 직렬 draft forward가
tile cliff를 넘는다"는 이야기는 병목 비용으로서는 반박(REFUTED)
되었다: draft는 B=4에서(유휴 34.5 ms) B=1(6.0 ms)보다 오히려
slack이 더 많다 — target의 verify GEMM이 draft보다 빨리 커진 것이다.

측정된 증거 (Exp2): K1 9→7 축소(fat7: verify 폭 40→32 rows = C와
정확히 같은 폭, 13→11 forwards)로 **TPS +22.6% (118.00 → 144.72,
C 대비 −3.7%)** 를 회복한다 — T_verify가 91.97 ms로 C의 91.42와
동률(parity)이 되고 DUET의 full step은 C보다 빨라진다(t=102.5 vs
106.2 ms). 더 줄이면(fat5: K1=5, verify 24 rows, 9 forwards) **C를
이긴다: 155.12 vs 150.31 (+3.2%)** — DUET의 첫 B>1 승리이며, 토큰
−14.5%를 step time −17%와 맞바꾼 결과다. **B=4 config는 fat5다**
(단일 run 주의, §4).

## 1. Exp1 — profile 포렌식 vs 구조적 ROW MODEL

1340개 step이 profile됨(warmup 이후 1240; full-batch 1229). Status
비중(draft): any-miss (mixed+miss) 56.5%, all-hit 42.7%, ramp 0.9% —
any-miss 비중은 1−hit^B = 1−0.81^4 = 0.57과 정확히 일치한다.

### 1a. Draft label (any-miss step, n=699) vs 모델

| label | B=1 (hit_k1) | B=4 | 모델 기대치 (64/40-row rows, 4/3 Marlin m-tiles) | 판정 |
|---|---|---|---|---|
| phase1_replay | 22.58 (9×2.51) | 47.31 (9×5.26) | 9 × ≤5.79 (tile-선형 bound 2.52+3×1.09) | ✓ bound 이하 (×4 rows에 ×2.13) |
| phase2_replay | 9.93 (4×2.48) | 17.81 (4×4.45) | 4 × ≤4.70 (3 tiles) | ✓ |
| draft_glue_replay | 1.78 | 3.60 | 40 rows vs 10 (3 tiles) | ✓ (×2.0) |
| glue (build 포함) | 2.70 | 4.48 | — | ✓ |
| phase1_prep | 3.23 | 4.04 | CPU prep, 9 units | ✓ (합계 +0.8) |
| phase2_prep | 1.69 | 1.95 | CPU prep, 4 units | ✓ |
| phase1_build / phase2_build | 0.78 / 0.85 | 1.00 / 0.83 | seq별 nested fan_out_list mask build (M3) | ✓ 평탄 — rebuild 폭증 없음 |
| merge_cache | 0.11 | 0.32 | B× keys | ✓ 미미 |
| draft_send_response | 0.22 | 0.47 | 4× wire | ✓ 미미 |
| hit_cache_respond (all-hit) | 0.89 | 0.89 | cache fill은 B와 무관 | ✓ 동일 |
| hit_cache_respond_mixed (JIT-all) | 8.00 (B=1 miss) | 8.64 | batched JIT, latency-bound | ✓ ×4 rows에 +8% — M2 주장 확인 |
| proxy_wait + draft_recv_cmd (IDLE) | 6.0 | 34.5 | — | draft slack이 오히려 커짐 (1c 참조) |
| top-level 합 vs wall | — | 124.49 vs 124.28 | — | ✓ 미계상 gap / sync storm 없음 |

어떤 draft label도 자신의 B×rows 기대치를 크게 웃돌지 않는다.
의심하던 숨은 비용들(step 0의 seq별 mask rebuild, nested
fan_out_list numpy 루프, sync storm)은 전부 ≤ +0.3 ms로 측정되었고
wall은 label로 완전히 설명된다. 참고로 hit_cache_respond_mixed는
mixed batch에서 모든 seq에 대해 JIT 응답을 한꺼번에 만들어 두고 hit
seq 것은 이후 정상 경로가 덮어쓰는(JIT-all 후 overwrite) 설계인데,
이 batched JIT 자체가 latency-bound라 — 즉 시간이 데이터 양이 아니라
커널 호출 지연에 지배되어 — row가 4배가 되어도 +8%만 늘어난다.

### 1b. Target label (any-miss step, target status 기준 n=1112)

| label | B=1 (hit_k1) | B=4 | 판정 |
|---|---|---|---|
| graph_pre | 31.54 | 78.94 | ✓ 물리: verify 40 rows vs 10; 한계비용 2.23 ms/row (1d 참조), tax_decomposition의 ~2.2 ms/pos verify 물리와 일치 |
| graph_post | 12.14 | 30.11 | ✓ 동일 |
| target_spec_wait | 2.68 | 3.00 all-hit / 10.84 any-miss | ✓ hit-step 대기 = B=1 baseline; miss stall은 구조적 (§2c) |
| verify_sample_accept | 3.64 | 2.40 | ✓ CPU가 더 길어진 GPU 뒤에 숨음 |
| proxy_compute_send | 1.48 | 0.50 | ✓ 오히려 줄어듦(숨겨짐) — mid-verify 블록은 커지는 B-비용이 아님 |
| exit_logits | 0.78 | 0.63 | ✓ 평탄 |
| final_logits | 0.36 | 0.58 | ✓ 미미 |
| top-level 합 vs wall | — | 121.4 vs 122.6 | ✓ 계상 완료 |

여기서 "~2.2 ms/pos verify 물리"란: verify할 위치(pos)가 하나 늘 때
target의 verify GEMM이 처리해야 할 row가 그만큼 늘어나는데, 이
모델/TP4 셋업에서 그 한계비용이 위치당 약 2.2 ms로 거의 선형이라는
뜻이다 — verify 폭이 시간 비용의 지배 항이 되는 물리적 근거다.

### 1c. B=4에서 어느 쪽이 병목인가: TARGET

| | B=1 | B=4 |
|---|---|---|
| draft wall / idle / work | 52.1 / 6.0 / 46.1 | 122.1 / 34.5 / 87.6 |
| target wall (mixed / all-hit) | 52.3 | 124.3 / 119.2 |

Draft 작업은 ×1.90 증가한 반면 target verify는 ×2.5 증가했다 —
draft의 slack은 오히려 넓어졌다(6→34.5 ms). 13번의 직렬 forward는
hit step에서 결코 병목이 아니고(spec_wait 3.0 ≈ B=1의 2.7), draft가
critical path에 닿는 것은 any-miss step의 JIT-response를 통해서
뿐이다(+7.8 ms, §2c). "더 적고 더 두꺼운 draft forward"를 1순위로
꼽았던 docs/duet/12의 랭킹은 이유가 틀린 채 절반만 맞았다: fat7의
승리는 draft forward 절약이 아니라 대부분 VERIFY 폭(K1+1) 축소에서
온다.

### 1d. vk_max 폭 분포와 padding tax

step dispatch는 graph_pre의 이봉성(bimodality)으로 측정(cut 60 ms):

| dispatch | step 비중 | graph_pre+post (ms) |
|---|---|---|
| k1 (vk_max=9, 40 rows) | 93.3% | 110.8 |
| k2 (all-short, 20 rows) | 5.8% | ~66 |
| ramp (partial batch) | 0.9% | — |

(대조용 B=1: k1 59% / k2-or-miss 41%.) all-short 확률은 이론과
일치한다: seq당 long(P1-hit) 비중 0.553 → 0.447^4 ≈ 4% 예측 vs
5.8% 측정(seq 상관). verify row 한계비용 (110.8−66)/20 ≈
**2.23 ms/row**. k1-dispatch step에서
E[short seqs | ≥1 long] ≈ 1.65 × 5 padded rows ≈ 8.3 낭비 rows →
**+18.4 ms × 93.3% ≈ 17.2 ms/step의 padding tax**. 교차 검증:
fat7의 폭 축소 40→32 rows가 T_verify를 −20.9 ms 움직였다(§3) — 두
추정치가 tax를 **17-21 ms/step**으로 양쪽에서 조인다. padding tax를
풀어 말하면: v1 설계에서는 batch에 긴 row가 하나라도 있으면 모든
seq가 최장 verify 폭(vk_max)으로 dispatch되므로, 짧아도 되는
seq들의 남는 자리를 padding으로 채워 그만큼의 verify GEMM 시간을
그냥 버리게 된다는 것이다.

## 2. 챔피언-shape B=4 격차의 인과 분해

격차: 118.00 vs 150.31 = −21.5%; R = 토큰 비율 0.910 × step-time
비율 0.863. 시간 쪽, ΔT_target = 129.8 − 113.7 = **+16.1 ms** vs C:

| 항 | ms/step | 측정 방법 |
|---|---|---|
| vk_max-padded verify 폭 | **+17.2** (17-21) | §1d row 모델; fat7 −20.9 확인 |
| mid-verify DUET 블록 (exit_logits + proxy_compute_send) | +1.1 | profile, 대부분 숨겨짐 |
| unpadded row 차이 (DUET 31.1 vs C 32 rows) | −2.2 | 2.23 ms/row |
| miss-stall 차이 (§2c) | −1.1 .. −4.7 | DUET 0.570×7.8 vs C 0.700×(8-13) |
| 직렬 forward 수 (13 vs 7) | 직접 비용 ~0 | draft는 hit step에서 병목 아님 (§1c) |
| 잔차 (vk_max 폭의 response-path glue, wire) | +1..+2 | profile |
| **합** | **+12..+16** | **측정 +16.1 ✓ 닫힘** |

토큰 쪽: 챔피언 tok/step 3.63 (PROFILE=0) vs 3.91 (PROFILE=1, 동일
인자/seed) — B=4의 단일-run 노이즈는 ±4%에 걸친다; C는 3.99.
−2..−9%의 토큰 결손은 DUET의 B=1 결손과 같다(m6: L_p2/miss-tok/P2
모두 B-불변) — **B-효과가 아니다**.

### 2c. miss-stall 증폭 항 (finding 5b의 메커니즘)

실재하며(IS present), 크기가 측정되었다:

| | B=1 | B=4 |
|---|---|---|
| DUET any-miss 비중 (1−h^B) | 0.19 | 0.570 (측정 0.565 ✓) |
| C any-miss 비중 | 0.26 | 0.700 |
| 빈도 우위 | 6 pts | **13 pts — 증폭 확인** |
| DUET의 any-miss step당 stall | 7.4 (spec_wait Δ) | 7.8 (10.84−3.00) |
| C의 any-miss step당 stall | ~12.8 (13.4 ms K7-JIT, 스케일 보정) | 8-13 (추정, C profile 없음) |
| DUET 순 우위 | ~+1.8 ms/step | **+1.1 .. +4.7 ms/step** |

이 항은 가설 그대로 B와 함께 커진다 — 그러나 B≤4에서는 한 자릿수
ms/step으로, 상쇄해야 할 padding tax보다 한 차수 작고, C의 평탄한
tok/step은 C가 자신의 stall을 상각함을 보여준다. 가설의 메커니즘은
실재하지만, 그 크기만으로는 승리를 이끌 수 없다.

## 3. Exp2 — B>1-shape 재조율 probe (B=4)

두 cell 모두: 챔피언 기본 인자, jit-short on, PROFILE=0.
fat7 = K1=7 K2=4 (k=11), uniform dfo=2 ([2]×8 = seq당 phase-1 16
rows, 11 직렬 forwards, verify 8 pos/seq = 32 rows). fat5 = K1=5
K2=4 (k=9), uniform dfo=3 ([3]×6 = seq당 18 rows, 9 forwards,
verify 24 rows; dfo<f가 되도록 --f 4 필요, pfo는 1 유지 → phase-2
budget 6).

| 지표 | champion b4 | prof_b4 (=champion+prof) | fat7 | fat5 | C b4 |
|---|---|---|---|---|---|
| Decode TPS | 118.00 | 126.69 | 144.72 | **155.12** | 150.31 |
| vs C | −21.5% | −15.7% | −3.7% | **+3.2%** | — |
| Tok/step | 3.63 | 3.91 | 3.71 | 3.41 | 3.99 |
| T_target (ms) | 129.82 | 131.74 | 108.95 | **95.35** | 113.71 |
| T_verify (ms) | 112.88 | 113.85 | 91.97 | **79.56** | 91.42 |
| T_draft (ms) | 103.70 | 105.22 | 86.99 | 75.54 | 75.28 |
| Cache hit | 0.80 | 0.81 | 0.82 | **0.84** | 0.74 |
| P1 hit / L_p1 | 0.529 / 3.50 | 0.553 / 3.84 | 0.591 / 3.42 | 0.660 / 2.83 | — |
| P2 hit / L_p2 | 0.274 / 1.63 | 0.261 / 1.83 | 0.230 / 1.69 | 0.175 / 1.67 | — |
| step t = B·tok/TPS (ms) | 123.1 | 123.5 | 102.5 | **87.9** | 106.2 |

fat7 소견: verify 폭이 C와 동률(32 rows) → T_verify 동률(91.97 vs
91.42). step time은 이제 C보다 빠르다(102.5 vs 106.2 — DUET에 남은
verify 우위 + 더 싼 stall). K1 9→7은 토큰을 거의 잃지 않았고(L_p1
3.42 vs 3.50 — 챔피언의 fo=1 꼬리 위치 8-9가 담당하던 몫은 ~0.1
tok) P1 hit rate는 오히려 올랐다(0.591). 남은 −3.7%는 전부 토큰
쪽(비율 0.930)이다 — DUET의 알려진 B=1 결손(L_p2 1.69 vs 손익분기
~2.6, docs/duet/12 finding 5a).

fat5 소견: fat 트레이드가 B=4에서 이긴다. verify 24 rows (C의 32보다
25% 아래) → T_verify 79.56 (C 대비 −13%); T_draft 75.54 ≈ C의
75.28, 직렬 forward 9번; hit rate 0.84 (any-miss 부담 1−0.84^4 =
0.50 vs C 0.70). K1=5가 chain 길이를 캡하면서 토큰은 줄지만(L_p1
2.83, tok/step 3.41 = C의 0.855) step-time 비율(t_C/t_D =
106.2/87.9 = 1.208)이 그 이상을 갚는다: R = 0.855 × 1.208 = 1.033 →
**측정 +3.2% (155.12 vs 150.31)**. fat5의 Marlin 기하 참고: phase-1
B×18 = 72 rows (4.5 tiles — tile 경계를 넘는데도 이긴다. B=4에서는
tile cliff 비용이 4개 seq에 상각되기 때문)이며, step당 토큰 비용은
세 DUET cell 중 최악이다 — 이 승리는 순수하게 step-time shape의
승리다.

## 4. 정직한 최종 진술

1. **버그 없음.** 26개 profile label 전부가 자신의 B×rows 구조
   모델과 일치한다; 두 프로세스 모두 wall이 label로 계상된다;
   hit-side cache fill과 batched JIT는 설계 그대로 동작한다
   (M2/M3/M6 모두 profile 하의 B=4에서 성립).
2. **챔피언 SHAPE는 B=1의 산물이었다.** K1=9 deep-narrow는 B=1의
   16-row tile cliff + 41% short-dispatch 혼합에 맞춰 조율된 것이다.
   B=4에서는 step의 93%가 모든 seq에 대해 K1-폭 verify를
   지불한다(v1 uniform-vk_max 설계 비용) — 17-21 ms/step. 해결책은
   코드가 아니라 shape다: **B=4에서는 fat5 (K1=5 K2=4 dfo=3
   uniform, k=9, --f 4)를 사용하라** — 155.12 tok/s, **SD-best C
   대비 +3.2%**, 최초로 측정된 B>1 DUET 승리다 (docs/duet/12
   finding 5b: v1에서 shape 재조율을 통해 부분 확인(partially
   CONFIRMED), 단일 run). fat7 (K1=7, 144.72, −3.7%)은
   토큰-보수적인 대안이다.
3. B=4에 남은 레버: DUET의 토큰 쪽 결손(tok/step이 C의 0.86-0.93,
   B-불변, L_p2≈1.7의 off-policy continuation 품질 — docs/duet/12
   finding 5a), mixed batch 내부의 잔여 padding을 회수할 seq별
   verify dispatch, 그리고 제대로 된 per-B shape 스윕(fat5는 첫
   fat-shape 추측이었다; K1∈{4,5,6} × dfo × pfo를 B∈{2,4,8}에서
   도는 것은 미탐색이고, fat5의 B∈{1,2}도 미측정 — B=1 챔피언은
   E9K24_jit 유지).

주의사항: cell당 단일 run(챔피언 tok/step이 동일 인자의 두 run에서
3.63-3.91에 걸침 — 토큰 쪽 비교는 ±4%); C 쪽 stall은 C profile 없이
추정; fat5는 --f 4 사용(다른 cell보다 넓은 miss JIT); PROFILE=1은
CUDA-event 오버헤드를 더하는데도 prof_b4가 PROFILE=0 챔피언
cell보다 빨랐다 — 이 델타는 token-draw 노이즈이지 profiling 가속이
아니다.

재현: `run_prof_b4.sh`, `run_retune.sh`, `analyze_prof.py` (이
디렉터리); raw profile JSON은 `prof_b4/` (docs/duet/12의 상시 prune
정책 적용 대상).
