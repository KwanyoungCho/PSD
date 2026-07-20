# pb_sweep — per-B DUET shape 스윕 + confirm (2026-07-18/19)

**질문** (`../verdict/RESULTS.md` §4.3의 후속): fat5/fat7은 첫
fat-shape 추측이었다. K1/K2/dfo/pfo가 B별로 실제 최적인가?
B ∈ {2,4}별로 진짜 grid를 돌리고, 승자를 SD-best C (k=7 f=6) 상대로
multi-rep으로 confirm하라.

**셋업**: HEAD f543c24, GPUs 0-4 (target TP4 on 0-3, draft on 4),
in=512 out=256 temp 0.7 seed 42 `--all`, jit-short on, exit=56,
`SSD_FORCE_SPLIT_K1K2=1`, PROFILE=0. Scan: ns=12, cell당 1 run,
ports 12930-12948 (`run_scan.sh` + `run_fixup.sh`). Confirm: ns=20,
B별 3-rep interleaved DUET/C, ports 12950+ (`run_confirm.sh`).
무관한 vLLM이 GPUs 6-7에서 전 구간 유휴(m5/verdict와 동일 regime).
cell 명명 규칙: `kAxB_dCpD` = K1=A, K2=B, dfo=C, pfo=D (k=K1+K2,
f=dfo+pfo, uniform phase-1 fan-out list [dfo]×(K1+1)).

## 결론 요약 (Verdict up front)

**두 승자 모두 C를 band-clear(최악 DUET rep > 최고 C rep)로 이긴다:**

- **B=4: k3x3_d4p1** (K1=3 K2=3 dfo=4 pfo=1, k=6 f=5) —
  **169.42 vs C 147.53 (+14.8%)**, 범위 167.24-171.89 vs
  142.48-151.28.
- **B=2: k6x5_d3p1** (K1=6 K2=5 dfo=3 pfo=1, k=11 f=4) —
  **114.09 vs C 106.73 (+6.9%)**, 범위 112.82-115.77 vs
  105.45-108.36.

B=1 헤드라인(+0.5%, docs/duet/12)과 함께 보면 승리는 **B와 함께
증폭된다: +0.5% → +6.9% → +14.8%** — docs/duet/12 finding 5b(B>1이
DUET의 regime)가 확인(CONFIRMED)되었다. 단, speculation shape를
B별로 재조율해야 한다는 조건이 붙는다(batch가 깊어질수록 최적
shape는 더 얕고 더 fat해진다). fat5는 B=4 최적이 아니었다: "shape가
최적인가?"에 대한 scan의 답은 **아니오** — 표면(surface)은 K1이
grid 가장자리(K1=3)까지 내려가는 동안 계속 오르며, ns=12 기준 fat5
대비 +10.6%다.

## 1. Scan 테이블 (ns=12, cell당 1 run)

### 1a. B=4 grid (TPS 내림차순; C anchor k7 f6)

| cell | k | f | TPS | tok/step | t_step (ms) | hit | P1/P2 hit | L_p1 | L_p2 | T_target | T_verify | T_draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3x3_d4p2 | 6 | 6 | 166.27 | 2.84 | 68.3 | 0.89 | .750/.135 | 1.95 | 1.52 | 73.71 | 60.08 | 56.86 |
| **k3x3_d4p1** | 6 | 5 | **165.50** | 2.82 | 68.2 | 0.86 | .733/.129 | 2.00 | 1.44 | 73.59 | 60.11 | 53.66 |
| k4x4_d3p1 | 8 | 4 | 156.74 | 3.23 | 82.4 | 0.84 | .681/.163 | 2.51 | 1.62 | 86.74 | 71.40 | 69.07 |
| k4x3_d3p1 | 7 | 4 | 154.68 | 3.13 | 80.9 | 0.84 | .690/.150 | 2.46 | 1.48 | 86.10 | 71.56 | 64.47 |
| k5x4_d3p2 | 9 | 5 | 153.78 | 3.47 | 90.3 | 0.85 | .665/.189 | 2.86 | 1.74 | 96.88 | 80.36 | 81.08 |
| k6x5_d3p1 | 11 | 4 | 153.76 | 3.75 | 97.6 | 0.84 | .641/.194 | 3.22 | 1.99 | 105.20 | 87.08 | 86.72 |
| **C (k7 f6)** | 7 | 6 | 152.11 | 4.08 | 107.3 | 0.74 | — | — | — | 111.38 | 89.27 | 73.22 |
| k5x5_d3p1 | 10 | 4 | 149.89 | 3.46 | 92.3 | 0.83 | .654/.175 | 2.75 | 1.89 | 97.13 | 79.68 | 80.32 |
| k5x4_d3p1 (fat5) | 9 | 4 | 149.72 | 3.33 | 89.0 | 0.82 | .650/.170 | 2.75 | 1.59 | 94.77 | 78.52 | 75.43 |
| k5x4_d4p1 | 9 | 5 | 149.70 | 3.36 | 89.8 | 0.84 | .693/.146 | 2.66 | 1.81 | 95.94 | 79.80 | 76.01 |

### 1b. B=2 grid (TPS 내림차순; C anchor k7 f6)

| cell | k | f | TPS | tok/step | t_step (ms) | hit | P1/P2 hit | L_p1 | L_p2 | T_target | T_verify | T_draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **k6x5_d3p1** | 11 | 4 | **114.35** | 3.63 | 63.5 | 0.84 | .653/.188 | 3.02 | 1.82 | 66.85 | 56.33 | 56.80 |
| k5x4_d3p1 | 9 | 4 | 114.22 | 3.41 | 59.7 | 0.84 | .665/.176 | 2.80 | 1.71 | 63.72 | 53.53 | 52.62 |
| k7x6_d2p1 | 13 | 3 | 112.33 | 3.76 | 66.9 | 0.81 | .557/.248 | 3.37 | 2.02 | 71.33 | 59.29 | 63.42 |
| k7x4_d2p1 (fat7) | 11 | 3 | 110.92 | 3.58 | 64.6 | 0.79 | .579/.214 | 3.26 | 1.69 | 68.82 | 58.27 | 55.69 |
| k6x5_d2p1 | 11 | 3 | 110.40 | 3.52 | 63.8 | 0.81 | .566/.239 | 3.02 | 1.88 | 67.46 | 56.54 | 57.34 |
| **C (k7 f6)** | 7 | 6 | 106.66 | 3.84 | 72.0 | 0.72 | — | — | — | 75.81 | 60.79 | 62.32 |

14개 DUET cell + 2개 C anchor 전부: rc=0, Traceback 0건 — config
assert에 걸린 cell 없음(grid 전체가 v1 제약 집합 K2≤K1, dfo<f,
k≤13 안에 있다).

## 2. Response surface 이야기

response surface(반응 표면)란 K1/K2/dfo/pfo 같은 조율
손잡이(knob)들을 축으로 놓았을 때 TPS가 그리는 곡면을 말한다 — 어느
knob이 성능을 어느 방향으로 얼마나 움직이는지의 지도다.

**K1(verify 폭)이 B=4의 지배적 knob이며, 순수한 시간 효과다.**
T_verify는 거의 K1만의 함수다: 60.1 (K1=3) / 71.4-71.6 (K1=4) /
78.5-80.4 (K1=5) / 87.1 (K1=6) ms — K1 한 단계당 +~9 ms = B × 2.25
ms/row로, verdict profile이 측정한 verify-row 한계비용(2.23
ms/row)과 정확히 일치한다. 고정 K1에서 K2는 T_verify를 ≤1.2 ms만
움직인다. K1 한 단계는 ~+0.3 tok/step(~+9%)만 사주는 대신 step
time을 ~+12% 요구하므로, K1이 내려갈수록 TPS는 단조 상승한다:
149.7 (K1=5) → ~155 (K1=4) → 165.5 (K1=3). 승자는 B×(K1+1) = 16
rows를 verify한다 — C의 32의 절반(HALF) 폭으로 step이 44%
빠르며(ns=20에서 67.4 vs 106.9 ms) — 그러면서 hit rate는
0.86-0.89로 오르고(짧은 chain은 맞히기 쉽다; K1=3의 P1 hit .733 vs
K1=5의 .650), 토큰 비율은 C의 0.72에서 유지된다. 단, 표면이 K1에
대해 전역적으로 단조인 것은 아니다: k6x5 (153.76)는 모든 K1=5
cell을 이긴다 — K2가 K1을 따라가면 토큰(tok/step 3.75, L_p2 1.99)이
깊이 비용을 다시 일부 갚는다 — 그러나 k3x3을 위협하지는 못한다.

**K2=K1 (valid_k gap 0)은 공짜 토큰이다.** 고정 K1에서 K1−K2 gap을
닫으면 verify 비용 ~0으로 phase-2 깊이가 추가된다(T_verify 동일):
k4x4 156.74 > k4x3 154.68; k5x5 149.89 ≈ k5x4 149.72; B=2에서는
k7x6 112.33 > k7x4 110.92. K2=K1이면 모든 dispatch가 같은 폭이라 v1
vk_max padding 항이 구조적으로 0이 된다 — batch 안 모든 row의
verify 길이가 같아지므로 짧은 row를 긴 폭에 맞춰 채우는(padding)
낭비 자체가 사라진다는 뜻이며, verdict의 "작은 K1−K2 gap이
padding을 중화한다" 메커니즘을 극한까지 밀어붙인 것이다. B=4 승자는
uniform-폭 K1=K2=3 shape다.

**pfo는 grid 중간에서는 실재하는 2차 knob이고, 승자 지점에서는
중립이다.** k5x4_d3p2 vs _d3p1: +4.1 TPS (+2.7%) — +0.14 tok/step에
t_step은 +1.3 ms뿐이다; 추가 proxy fan-out은 draft 유휴가
부담한다(T_draft +5.7 ms지만 여전히 T_target 아래) — "draft 유휴가
pfo 비용을 댈 수 있다"는 가설을 방향적으로 확인. 승자 shape에서는
효과가 포화된다: k3x3_d4p2 166.27 vs _d4p1 165.50 (+0.5%, 단일-run
노이즈 이내 — 추가 row의 토큰 이득 2.84 vs 2.82가 더 이상 그 draft
비용 T_draft +3.2 ms를 감당하지 못한다).

**dfo: B=4에서는 평탄, B=2에서는 주 knob.** k5x4_d4p1 ≡ k5x4_d3p1
(149.70 vs 149.72). 그러나 B=2에서 dfo 2→3은 +3.6%의
가치이고(k6x5: 114.35 vs 110.40) dfo=3 두 cell이 B=2 grid 상위를
차지한다 — B=2에서는 draft에 아직 slack이 있고(phase-1 rows
2×dfo×(K1+1)가 Marlin tile 근처에 머문다) fat한 phase-1이 hit
rate를 0.81 → 0.84로 올린다.

**B=2는 절벽이 아니라 평평한 능선(ridge)이다.** B=2 grid 전체가
±1.8% (110.4-114.35)에 들어오고 모든 cell이 C_b2 (106.66)를
이긴다 — B=2에서는 verify 폭 항의 기울기가 절반(B×2.25 ms/row)이라
K1 5→6은 무승부이고(114.22 vs 114.35, ns=12에서 tie) dfo=3 이외의
shape 선택은 거의 무의미하다. k6x5_d3p1을 k5x4_d3p1 대신 승자로
부른 것은 0.1% 차이 — ns=12에서는 분해 불가; robustness를 위해
토큰이 더 높은 cell을 골라 그것을 confirm했으므로, 권고는
"k6x5_d3p1 (confirmed) 또는 k5x4_d3p1 (scan에서 통계적으로 구별
불가)"이다.

## 3. Confirm 단계 (ns=20, B별 3-rep interleaved DUET/C)

### 3a. B=4 — k3x3_d4p1 vs C

| rep | DUET TPS | C TPS |
|---|---|---|
| r1 | 171.89 | 142.48 |
| r2 | 169.12 | 151.28 |
| r3 | 167.24 | 148.84 |
| **평균 ± 범위** | **169.42** (167.24-171.89) | **147.53** (142.48-151.28) |

**+14.8%, band-clear** (최악 DUET 167.24 > 최고 C 151.28, +10.5%).
메커니즘 (3-rep 평균): tok/step 2.85 vs 3.94 (비율 0.723) × t_step
67.4 vs 106.9 ms (비율 1.586) → R = 1.147 ✓. T_verify 60.5 vs 92.4
ms, T_draft 54.0 vs 75.0 ms, hit 0.87 vs 0.73 (any-miss 부담
1−0.87⁴ ≈ 0.43 vs 0.72). DUET rep 편차 ±1.4%, C ±3.0%.

### 3b. B=2 — k6x5_d3p1 vs C

| rep | DUET TPS | C TPS |
|---|---|---|
| r1 | 113.67 | 108.36 |
| r2 | 112.82 | 106.39 |
| r3 | 115.77 | 105.45 |
| **평균 ± 범위** | **114.09** (112.82-115.77) | **106.73** (105.45-108.36) |

**+6.9%, band-clear** (최악 DUET 112.82 > 최고 C 108.36, +4.1%).
메커니즘: tok/step 3.60 vs 3.79 (0.950) × t_step 63.1 vs 71.2
(1.128) → R = 1.071 ✓. T_verify 56.8 vs 58.4 (거의 동률), T_draft
57.4 vs 62.6, hit 0.83 vs 0.73. 양쪽 편차 모두 ±1.5%.

12개 confirm cell 전부 rc=0, Traceback 0건.

### 3c. 증폭 곡선 (finding 5b)

| B | 최적 DUET shape | vs C | 근거 |
|---|---|---|---|
| 1 | E9K24_jit (K1=9 K2=4, deep-narrow list) | **+0.5%** | 5-rep 헤드라인, docs/duet/12 (out=512 ns=50; out=256 ns=20 regime에서는 −4.1%) |
| 2 | k6x5_d3p1 (K1=6 K2=5 dfo=3 pfo=1) | **+6.9% band-clear** | 3-rep interleaved, 이 스윕 |
| 4 | k3x3_d4p1 (K1=3 K2=3 dfo=4 pfo=1) | **+14.8% band-clear** | 3-rep interleaved, 이 스윕 |

최적 shape는 B가 커질수록 더 얕고 더 fat해진다(K1 9 → 6 → 3,
f 3 → 4 → 5). verify-폭 비용은 B에 비례해 커지는 반면 draft
forward는 latency-bound로 남기 때문이다: B=1은 tile cliff 위에서
토큰을 최적화하고, B=4는 verify GEMM 위에서 step time을 최적화한다.
**finding 5b는 per-B shape 재조율을 전제로 확인(CONFIRMED)되었다** —
B>1은 DUET이 이기는 regime이며, 승리는 B와 함께 커진다.

## 4. 정직한 주의사항

1. **Scan은 ns=12, cell당 1 run이다.** 이 길이의 단일-run 노이즈는
   ±3-4%다(fat5가 여기서는 149.72, verdict의 ns=20 run에서는
   155.12로 측정됨). grid 중간의 순위(k4x4 vs k4x3, k6x5 vs k5x5,
   B=2 상위 2개의 tie)는 분해 불가하고, 승자 마진(B=4 차점 shape
   대비 +9.5%, B=2 dfo=2 cell 대비 +7-8)은 분해 가능하다.
   multi-rep 근거를 가진 것은 confirm된 두 승자뿐이다.
2. **C scan anchor와 confirm 단계는 DUET scan cell보다 ~20시간 뒤에
   실행되었다** (원래 scan의 C cell들이 run-script argparse 버그로
   crash — 떠돌이 positional 인자, 모델 로드 전 rc=2;
   `run_scan.sh`에서 수정, `run_fixup.sh`로 재실행). 날짜 간
   drift가 scan-테이블의 DUET-vs-C 격차를 왜곡할 수는 있지만
   confirm 결론은 아니다: confirm 단계는 내부적으로
   interleave되었다(DUET/C 교대, 같은 세션). ns=12의 C_b4
   (152.11)와 confirm C_b4 평균(147.53)이 verdict의 150.31을
   사이에 끼운다(bracket) — regime 이동은 보이지 않는다.
3. **K1=3은 grid 가장자리다.** 표면은 얕은 끝에서 여전히 오르는
   중이었다; K1=2 (verify 12 rows)는 미측정이고, K2>K1(v1 제약으로
   배제)과 B=8도 마찬가지다. k3x3_d4p2 (166.27, 1 run)는 승자
   shape에서 pfo=2가 최악의 경우에도 중립임을 시사한다 — 미확인.
4. **Regime**: out=256 ns=20 temp 0.7, in=512, 프롬프트 세트 1개
   (--all, seed 42). B=1 헤드라인(+0.5%)은 out=512 ns=50 regime의
   것이다; 이 regime에서 B=1 DUET은 −4.1%로 측정되었다(m6_fix) —
   증폭 곡선의 B=1 점은 regime 의존적이고 B∈{2,4} 점들은
   아니다(여기서 interleave로 측정).
5. **토큰 가격**: B=4 승자는 tok/step 2.85 vs C의 3.94를 지불한다 —
   승리는 100% step-time(67 vs 107 ms)이다. 토큰당 target 비용을
   올리는 것은 무엇이든(더 긴 context, 더 큰 모델) frontier를 더
   깊은 shape 쪽으로 되돌린다; per-B 최적은 보편적이지 않다.
6. GPUs 6-7은 모든 m5/verdict run과 같은 무관한 유휴 vLLM을 실었고,
   GPUs 0-5는 그 외에 비어 있었다(scan 시작 시 확인).

재현: `run_scan.sh` (grid), `run_fixup.sh` (C anchors + k3x3_d4p2),
`run_confirm.sh` (승자, env-파라미터화), `extract.py` (테이블);
cell별 raw run.log는 `<cell>/`과 `confirm/<cell>/` (미커밋,
docs/duet/12의 상시 prune 정책).
