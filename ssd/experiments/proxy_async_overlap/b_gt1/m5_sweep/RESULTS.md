# M5 — B ∈ {1,2,4} 스윕: DUET 챔피언 vs SD-best C (interleaved 실행)

## ⚠ 정정 (2026-07-18, M6) — 아래의 DUET 수치는 버그가 있었음 (BUGGED)

이 스윕의 원래 DUET cell들은 B>1 정합성 버그(docs/duet/13 §M6)를
안고 실행되었다: target verify(검증) 입력 window가 batch 전체에
균일한 `vk_max`를 사용한 반면, 각 seq의 토큰은 seq별 `vk_i`만큼만
연장되었다. 그 결과 MIXED batch 안의 모든 SHORT row(P2 hit /
JIT-short miss, vk=K2=4)는 verify window가 이미 알고 있는 문맥
쪽으로 5토큰 뒤로 밀렸고 — 해당 chain은 낡은(stale) 예측과 비교되어
기각되었으며, recovery 토큰은 과거 문맥을 다시 출력했다(출력 정합성
버그이며 `python -O`에서는 조용히 지나간다). "L_p2 붕괴"(1.64→0.49),
miss 토큰 붕괴(2.57→1.48), 부풀려진 P2 hit rate(0.28→0.445)를 만든
것은 바로 이 버그이지 DUET 알고리즘이 아니다. C cell들은 영향을 받지
않았다(DUET gate가 없음). DUET cell들은 수정 후 동일한 인자/GPU로
재실행되었고(ports 12911-13, `../m6_fix/duet_b{1,2,4}/`), 아래의 C
수치는 원본 그대로다.

**정정된 raw 지표** (모든 cell rc=0, Traceback 0건):

| 지표 | duet_b1 | duet_b2 | duet_b4 | (버그 당시 b1/b2/b4) |
|---|---|---|---|---|
| Decode TPS (합산) | 74.69 | 104.59 | 118.00 | 71.86 / 89.22 / 108.87 |
| Tokens/step (recovery 포함) | 3.71 | 3.89 | 3.63 | 3.62 / 3.24 / 3.27 |
| Cache hit rate | 0.81 | 0.81 | 0.80 | 0.80 / 0.82 / 0.84 |
| P1 (draft) hit rate | 0.537 | 0.544 | 0.529 | 0.523 / 0.428 / 0.392 |
| P2 (proxy) hit rate | 0.269 | 0.269 | 0.274 | 0.280 / 0.390 / 0.445 |
| L_p1 | 3.61 | 3.83 | 3.50 | 3.54 / 4.05 / 5.07 |
| L_p2 | 1.73 | 1.81 | 1.63 | 1.64 / 0.85 / 0.49 |
| hit 시 tok/step | 3.98 | 4.16 | 3.86 | 3.88 / 3.52 / 3.62 |
| miss 시 tok/step | 2.59 | 2.71 | 2.68 | 2.57 / 1.98 / 1.48 |
| T_target full step (ms) | 52.06 | 79.28 | 129.82 | 52.70 / 76.71 / 126.44 |
| T_verify (ms) | 45.88 | 68.27 | 112.88 | 46.41 / 66.39 / 110.86 |
| T_draft step (ms) | 44.67 | 66.40 | 103.70 | 45.09 / 65.29 / 102.06 |

원래 실행에서 보였던 "B에 대한 단조(monotone) 효과"는 전부
사라졌다: L_p2, miss 토큰, P2 hit rate가 이제 B와 무관(B-INVARIANT)
하다 (1.73/1.81/1.63, 2.59/2.71/2.68, 0.269/0.269/0.274) — seq별로
독립적인 rollout이라면 정확히 이렇게 나와야 하는 결과다. B=1은
run-to-run 노이즈 범위 안에서 변화가 없다(이 fix는 B=1에서는
no-op이다).

**정정된 B-스케일링 vs C** (C row는 아래 § 원본과 동일):

| B | DUET TPS | C TPS | 격차 | DUET ×B1 | C ×B1 | DUET /seq | C /seq |
|---|---|---|---|---|---|---|---|
| 1 | 74.69 | 77.90 | −4.1% | 1.00 | 1.00 | 74.7 | 77.9 |
| 2 | 104.59 | 109.86 | −4.8% | ×1.400 | ×1.410 | 52.3 | 54.9 |
| 4 | 118.00 | 150.31 | −21.5% | ×1.580 | ×1.930 | 29.5 | 37.6 |

**정정된 격차 분해** (R = 토큰 비율 × step-time 비율,
t = B·tok_step/TPS; 각 row에서 R이 측정된 TPS 비율을 재현한다):

| B | tok_D/tok_C | t_C/t_D | R = D/C | (버그 당시 R) |
|---|---|---|---|---|
| 1 | 0.937 | 1.023 | 0.959 | 0.921 |
| 2 | 0.997 | 0.954 | 0.952 | 0.813 |
| 4 | 0.910 | 0.863 | 0.785 | 0.725 |

**수정된 결론**: B=2에서 DUET은 tok/step 동률(3.89 vs 3.90)로
사실상 동급(NEAR-PARITY, −4.8%)이다 — 버그 당시의 −18.8%는 거의
전부 버그였다. B=4 격차(−21.5%)는 살아남았지만 그 구성이 바뀌었다:
B1→B4의 log-격차 확대분(−0.200) 중 ~85%가 시간(TIME) 쪽(t_C/t_D
1.023 → 0.863)이고 토큰 쪽은 ~15%뿐이다(0.937 → 0.910, B=4에서
tok/step 3.89→3.63 하락 — 단일 run이라 노이즈일 수 있음). 살아남은
설명은 §2(a)의 batched-GEMM 물리학이며, 이제 버그 없이 깨끗하게
측정되었다: T_draft가 C의 ×1.93 대비 ×2.32로 증가(B=4에서 103.7 vs
75.3 ms, +37.8% — B×16 row짜리 13번의 직렬 forward가 Marlin tile
cliff를 넘음), T_verify는 ×2.46 vs ×2.01(112.9 vs 91.4 ms, +23.5% —
vk_max padding + mid-verify DUET 블록). 토큰/hit 쪽 기계 장치는
문제가 아니다: hit rate는 0.80-0.81로 평탄하고, B=4에서 DUET이
miss로 잃는 토큰은 이제 C보다 오히려 적다(step당 결손 0.24 vs
0.30 tok).

finding 5b(증폭 가설)에 대해: 여전히 확인되지 않음(NOT confirmed) —
DUET은 어떤 B에서도 이기지 못하고 C의 스케일링이 계속 우세하다
(×1.93 vs ×1.58). B ≤ 4에서는 any-miss JIT stall이 C에게 병목이 되는
항이 아니기 때문이다. 그러나 기각(REJECTION)의 근거가 달라졌다:
DUET의 B>1 문제는 토큰 희석이 아니라(그건 버그였다) draft/verify
step-time의 SHAPE(형상)이며, 이는 앞서 순위를 매겨둔 레버들(B별로 더
적고 더 두꺼운 draft forward, per-B verify dispatch, mid-verify
블록을 critical path 밖으로)로 해결 가능한 문제다. §3의 "P2 구성
변화(composition shift)"와 "기묘한 L_p1 상승"은 버그의 산물로 완전히
설명된다(오염된 short row들이 가짜 P2 hit을 양산했고, 퇴화된 반복
텍스트가 살아남은 P1 chain을 부풀렸다).

이 선 아래의 모든 내용은 원본(버그가 있던 DUET) 작성분이며, 무엇이
측정되었고 왜 오도했는지에 대한 역사적 기록으로 그대로 남긴다.

---

**날짜**: 2026-07-18. docs/duet/13의 Stage M5. GPUs 0-4, ports
12900-12905, cell당 1 run, B별로 INTERLEAVED(duet_bB 다음 c_bB).
모든 cell: ns=20 (×4 datasets = 80 prompts), out=256, in=512,
temp 0.7, seed 42, `--all`. DUET = 챔피언 E9K24_jit (m4-smoke 인자:
K1=9 [2×6,1×4], K2=4, exit 56, pfo=1, `SSD_FORCE_SPLIT_K1K2=1
SSD_DUET_JIT_SHORT=1`). C = `--k 7 --f 6`, DUET gate 없음.
6개 cell 모두 rc=0, Traceback 0건. 무관한 vLLM이 GPUs 6-7에서 스윕
시작과 끝 모두 동일하게 떠 있었음(전 구간 동일 regime).

## 결론 요약 (Verdict up front)

**B>1 가설(docs/duet/12 finding 5b)은 v1에서 기각(REJECTED)**:
C가 모든 B에서 더 잘 스케일된다. 격차는 −7.8% → −18.8% → −27.6%로
벌어진다. DUET의 재료적 우위는 전부 실현되었는데도(B=4에서 hit 0.84
vs 0.74, seq당 더 적은 row, 더 싼 miss) 졌다. 이유는 (i) P2 희석이
실제 B-효과(L_p2 1.64 → 0.49)로 DUET의 tokens/step을 ~10% 깎는 반면
C의 tok/step은 평탄하고, (ii) DUET의 step time이 모든 축에서 C보다
빨리 증가하기 때문이다(verify ×2.39 vs ×2.01, draft ×2.26 vs
×1.93) — seq당 verify row 수가 절반인데도 그렇다. 가설이 기대던
메커니즘인 any-miss JIT stall은 B ≤ 4에서는 어느 시스템에서도 증가
항이 아닌 것으로 드러났다.

## Raw per-cell 지표

| 지표 | duet_b1 | c_b1 | duet_b2 | c_b2 | duet_b4 | c_b4 |
|---|---|---|---|---|---|---|
| Decode TPS (합산) | 71.86 | 77.90 | 89.22 | 109.86 | 108.87 | 150.31 |
| Tokens/step (recovery 포함) | 3.62 | 3.96 | 3.24 | 3.90 | 3.27 | 3.99 |
| Cache hit rate (seq당) | 0.80 | 0.74 | 0.82 | 0.73 | 0.84 | 0.74 |
| P1 (draft) hit rate | 0.523 | - | 0.428 | - | 0.392 | - |
| P2 (proxy) hit rate | 0.280 | - | 0.390 | - | 0.445 | - |
| L_p1 (P1 accepted 길이) | 3.54 | - | 4.05 | - | 5.07 | - |
| L_p2 (P2 accepted 길이) | 1.64 | - | 0.85 | - | 0.49 | - |
| hit 시 tok/step | 3.88 | 4.23 | 3.52 | 4.26 | 3.62 | 4.29 |
| miss 시 tok/step | 2.57 | 3.20 | 1.98 | 2.94 | 1.48 | 3.12 |
| T_target full step (ms) | 52.70 | 53.38 | 76.71 | 75.80 | 126.44 | 113.71 |
| T_verify (ms) | 46.41 | 45.41 | 66.39 | 58.26 | 110.86 | 91.42 |
| T_draft step (ms) | 45.09 | 39.05 | 65.29 | 62.89 | 102.06 | 75.28 |

정합성 검증: hit·tok_hit + miss·tok_miss가 모든 cell에서 tok/step을
재현한다 (예: duet_b4: 0.84·3.62 + 0.16·1.48 = 3.28 ≈ 3.27;
c_b4: 0.74·4.29 + 0.26·3.12 = 3.99).

## 1. B에 따른 스케일링

| B | DUET TPS | C TPS | DUET vs C | DUET ×B1 (효율) | C ×B1 (효율) | DUET /seq | C /seq |
|---|---|---|---|---|---|---|---|
| 1 | 71.86 | 77.90 | −7.8% | 1.00 | 1.00 | 71.9 | 77.9 |
| 2 | 89.22 | 109.86 | −18.8% | ×1.242 (62%) | ×1.410 (71%) | 44.6 | 54.9 |
| 4 | 108.87 | 150.31 | −27.6% | ×1.515 (38%) | ×1.930 (48%) | 27.2 | 37.6 |

seq당 토큰 지연 (ms/tok): DUET 13.9 → 22.4 → 36.7;
C 12.8 → 18.2 → 26.6. C가 모든 B에서 합산 처리량과 seq당 지연 모두
우세하며, B가 ×2 될 때마다 격차는 ~10포인트씩 벌어진다.

**격차 분해** (TPS 비율 R = 토큰 비율 × step-time 비율;
step time t = B·tok_step/TPS, T_target과 ~5% 이내로 일치):

| B | tok_D/tok_C | t_C/t_D | R = D/C |
|---|---|---|---|
| 1 | 0.914 | 1.008 | 0.921 |
| 2 | 0.831 | 0.978 | 0.813 |
| 4 | 0.820 | 0.884 | 0.725 |

B=1→2의 확대는 거의 순수하게 토큰 쪽이다(P2 희석, §3);
B=2→4에서는 시간 쪽이 합류한다(DUET의 verify/draft 증가, §2b).

## 2. Any-miss 증폭 — 가설 vs 데이터

관측된 seq당 miss 비율과, 거기서 유도되는 batch 수준 any-miss 부담
P(any miss) = 1 − hit^B:

| B | DUET miss | C miss | DUET 1−h^B | C 1−h^B | DUET step당 tok 결손 | C step당 tok 결손 |
|---|---|---|---|---|---|---|
| 1 | 0.20 | 0.26 | 0.20 | 0.26 | 0.26 | 0.27 |
| 2 | 0.18 | 0.27 | 0.33 | 0.47 | 0.28 | 0.36 |
| 4 | 0.16 | 0.26 | 0.50 | 0.70 | 0.34 | 0.30 |

(step당 tok 결손 = miss_share · (tok_hit − tok_miss).)

DUET의 hit 우위는 실제로 실현되었고 심지어 확대되었다(+0.06 → +0.10
절대값; 이론적 any-miss 부담은 B=4에서 0.50 vs 0.70). **그런데도
이득이 되지 않았다.** 측정으로 확인된 이유는 두 가지다:

(a) **B ≤ 4에서는 JIT stall이 step당 증가 항이 아니다.** step time이
any-miss에 결합되어 있었다면, C의 step time(any-miss 부담 0.26 →
0.70, +0.44)이 DUET(0.20 → 0.50, +0.30)보다 빨리 증가했어야 한다.
실제로는 반대였다: C의 T_target은 ×2.13, DUET은 ×2.40으로 증가했고,
C의 tok/step은 평탄하다(3.96 → 3.99) — C의 더 높은 miss 부담은
완전히 상각(amortize)된다. 실제로 지배하는 step-time 증가는 batched
GEMM 물리학이다: T_verify (D ×2.39 vs C ×2.01), T_draft (D ×2.26 vs
C ×1.93). DUET은 seq당 26-row 예산 vs C의 48인데도 두 축 모두에서 더
빨리 증가한다 — 다음과 일치하는 결과다: (i) v1 vk_max padding
비용(mixed batch는 모든 row에 대해 항상 K1 폭의 verify를 지불한다;
B=4에서는 사실상 매 step에 긴 row가 ≥1개 존재한다), (ii) DUET 전용
mid-verify 블록(exit-56 proxy + batched Policy B + 2·B·wire_N
wire)은 B=1에서는 결코 병목으로 측정되지 않았지만 B에 비례해 커지는
step당 직렬 작업이고, (iii) draft 쪽: B×16 / B×10 row짜리 13번의
직렬 forward가 B ≥ 2에서 Marlin tile cliff를 넘는다 (B=1에서 DUET을
싸게 만들었던 latency-bound 무임승차가 역전됨 — C의 7번 fat
forward는 준선형으로 batch되어 ×4 row에 ×1.93이고, C의 seq당 draft
비용은 52% 감소 vs DUET 43%; B=4의 slack T_target−T_draft: C 38.4 ms
vs DUET 24.4 ms). 쉽게 풀면: 아주 작은 행렬곱은 계산량이 아니라 커널
실행 지연(latency)에 묶여 있어 row를 조금 늘려도 시간이 거의 늘지
않지만(latency-bound), row 수가 Marlin 커널의 tile 경계를 넘는 순간
시간이 계단식으로 뛴다(tile cliff). B=1에 맞춰 조율된 DUET의 깊고
좁은 draft 형상은 B≥2에서 이 경계를 넘어버리고, vk_max padding은
batch에 긴 row가 하나라도 있으면 짧은 row까지 최장 폭에 맞춰
채워(padding) verify하느라 불필요한 계산을 낭비하는 비용이다.

(b) **잘못된 통화(currency), 그리고 그 복리 효과** (docs/duet/12의
finding 1): hit의 가치는 토큰이 아니라 회피한 stall인데 — K2=4
JIT-short로 처리되는 DUET의 miss는 토큰 상한이 낮다. miss 시
tok/step이 B와 함께 붕괴하는(2.57 → 1.98 → 1.48) 반면 C는
유지된다(3.20 → 2.94 → 3.12). B=4에서 DUET은 miss가 10포인트 더
적은데도 miss로 잃는 tokens/step이 C보다 많다(0.34 vs 0.30). 싼
miss는 B=1에서는 tail-latency 이점이었지만 B>1에서는 토큰 부채가
된다.

(a)에 대한 주의: 이 스윕은 SSD_PROFILE_DUET=0으로 실행되어 status별
step 타이밍이 없고, 따라서 t_step(hit-only)과 t_step(any-miss)을
직접 나눌 수 없다. 위의 귀속은 시스템 간 증가율 비교와 C의 평탄한
tok/step에 근거한 것이지, step별 타임라인에 근거한 것이 아니다.

## 3. M4의 P2-구성(composition) 플래그 — 실제 B-효과, ns=20에서 확인

| B | P1 hit | P2 hit | L_p1 | L_p2 |
|---|---|---|---|---|
| 1 | 0.523 | 0.280 | 3.54 | 1.64 |
| 2 | 0.428 | 0.390 | 4.05 | 0.85 |
| 4 | 0.392 | 0.445 | 5.07 | 0.49 |

M4 플래그는 유지되며 ns=20에서 B에 대해 단조다: hit 구성이
draft-소스에서 proxy-소스로 이동하고(hit 중 P2 비중 35% → 53%), P2
accepted 길이는 1.64 → 0.85 → 0.49로 붕괴한다 — B=4에서 proxy-소스
hit 하나의 가치는 토큰 반 개다. 크기에 대한 노트: M4의 ns=8
tok/step −24%는 과장이었고(ns 노이즈), 실제 효과는 B=2와 B=4에서
tok/step ~−10%다(3.62 → 3.24/3.27) — 그러나 실재하고, 단조적이며,
C의 tok/step이 평탄하므로 격차 확대의 토큰 쪽 전부를 이것 하나가
설명한다(§1 분해). 기묘하게도 P1 hit rate가 떨어지는데 L_p1은
오른다(3.54 → 5.07) — 살아남는 draft-소스 hit들이 더 깊다는 뜻이다.
메커니즘은 미해결이다(seq당 예산은 설계상 상수이므로 seq 간 예산
분할 문제가 아니다; 후보: step time과 함께 커지는 proxy hint의
off-policy 신선도 저하, 어떤 row가 P1/P2에 도달하는지의 선택 효과).
future work로 플래그해 둔다.

## 4. 결론 + 주의사항

| 주장 (docs/duet/12 finding 5b) | 측정 결과 |
|---|---|
| B>1은 DUET의 구조적 홈그라운드 | **NO (v1)** — 격차 −7.8% → −27.6%, B에 대해 단조 |
| draft forward는 batch해도 ~공짜 (latency-bound) | B≥2에서 NO — B×16 row가 tile cliff를 넘음; T_draft ×2.26 vs C ×1.93 |
| 26 rows/seq는 48보다 깊게 batch됨 | verify rows/seq 우위는 나타나지 않음: T_verify ×2.39 vs C ×2.01 (vk_max padding + mid-verify 블록) |
| 높은 hit + 싼 miss는 B와 함께 복리 | hit 우위는 확대(0.84 vs 0.74)됐지만 B≤4에서 any-miss stall은 병목 항이 아님; 싼 miss는 토큰 부채가 됨(miss 시 1.48 tok) |
| P2 구성 플래그 (M4) | 실제 B-효과: L_p2 1.64 → 0.49, hit 중 P2 비중 35% → 53% |

DUET이 B>1에서 이기려면 무엇이 바뀌어야 하나(future work, 기대
레버리지 순): (1) P2 희석 수정 — B=2 토큰 격차의 전부다(B별로 K2 /
P1-P2 예산 분할 재조율; 먼저 구성 변화부터 이해할 것); (2) B>1에서
더 적고 더 두꺼운(fat) draft forward — 13-forward 깊고-좁은 형상은
B=1 tile cliff에 맞춘 것으로 B≥2에서 역전된다; (3) per-B verify
dispatch(two-bucket 또는 seq별 폭)로 mixed batch의 short row가 K1
폭을 지불하는 것을 중단; (4) mid-verify DUET 블록을 critical path
밖으로(B=1의 null probe 결과는 B>1로 이전되지 않는다).

주의사항: cell당 단일 run(오차 막대 없음; B=1의 DUET-C 격차 −7.8%
vs 5-rep 헤드라인 +0.5%는 out=256/ns=20이 baseline을 C에 유리하게
움직임을 보여준다 — 헤드라인 관례는 out=512/ns=50); interleave는
했으나 반복은 안 함(느린 drift 미통제); 무관한 vLLM이 GPUs 6-7에서
전 구간 유휴 상태로 존재(모든 cell 동일 regime, M1-M4 smoke regime과
동일); `--b 4` cell은 20 seq를 4개씩 wave로 실행(꼬리 wave는 부분
채움일 수 있음); SSD_PROFILE_DUET=0(status별 타이밍 없음, §2 주의
참조).

재현: `run_all.sh` (이 디렉터리); cell별 로그는
`{duet,c}_b{1,2,4}/run.log`; `extract.py`가 raw 테이블을 재생성한다.
