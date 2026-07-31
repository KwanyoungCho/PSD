# jit_subset — mixed batch에서 miss 행만 JIT하는 A/B (B=16, 2026-07-31)

**질문** (사용자 제안): JIT-all-then-overwrite는 hit 행까지 JIT 연산을
굴린다(응답 내용은 캐시가 덮어써서 무관 — 연산만 낭비). miss 행만
gather→compact JIT→scatter로 돌리면(`SSD_DUET_JIT_SUBSET=1`, 커밋
b0f42da) 더 빠른가?

**사전 예측(기록해 둔 것)**: JIT forward는 seq당 1토큰 decode라 비용이
폭(B)이 아니라 깊이(K2회 직렬)에 있음 → 이득 ~0 가능.

**설정**: B=16 ns=16 out=256 seed 42 temp 0.7, PROFILE=0, exit=56,
jit-short on, 3-rep 인터리브(base/sub 교대), ports 13800+. 형상 2종:
k1x1_d5p1(K2=1, JIT 1fwd — B=16 확정 승자)과 k2x2_d4p1(K2=2, JIT 2fwd).

## 결과

| 형상 | JIT-all(base) 3런 → 평균 | subset 3런 → 평균 | Δ | 라운드 |
|---|---|---|---|---|
| k1x1_d5p1 | 259.04/261.07/261.08 → **260.40** | 261.20/262.17/262.27 → **261.88** | **+0.6%** | sub 3/3승 (밴드 0.12 차로 분리) |
| k2x2_d4p1 | 252.92/246.44/248.04 → **249.13** | 244.17/246.30/241.76 → **244.08** | **−2.0%** | base 3/3승 (sub 최고 246.30 < base 최저 246.44) |

## 판정 — 게이트는 기본 OFF 유지 (일반 이득 없음)

형상 의존적으로 갈렸고 순효과는 없음:

- **K2=1**: +0.6% — 미세하지만 3/3 일관. JIT이 forward 1회뿐이라
  subset의 고정 비용(nonzero 동기화 + gather/scatter)을 근소하게 상회.
- **K2=2**: −2.0% — 역효과가 더 크고 역시 3/3 일관. 해석: subset의
  n(miss 수, 보통 2~5)이 CG bucket 정렬에서 벗어나 forward가 작은
  bucket/eager 경로로 떨어지면, per-forward 발사 비용이 16행 bucket
  replay보다 비싸질 수 있고, K2=2는 그 비용을 두 번 지불한다. 깊이가
  비용이라는 사전 예측과 정합.

**결론**: JIT-all-then-overwrite의 "여분 행은 공짜" 가정은 B=16에서도
유효(폭-latency-bound)하며, subset화는 일반적 이득이 없고 K2≥2에서
오히려 손해다. M2 설계 재검증. 게이트는 A/B 재현용으로 코드에 남기되
(기본 OFF), 승격하지 않는다.

주의: B=32 미측정(신호가 없어 확장 불요 판단); n이 bucket 크기와
일치하는 레짐(예: miss가 정확히 4/8개)에서는 결과가 다를 수 있으나
miss 수는 확률적이라 제어 불가.

## 추가 — JIT 구간 격리 측정 (프로파일 쌍, 같은 날)

TPS는 토큰 샘플링 운(±1.5~2.5 tok/s)이 섞이므로, 사용자 지적대로
respond 구간(`hit_cache_respond_*` span)만 격리한 프로파일 쌍을 떴다
(k2x2_d4p1@B=16, out=128, PROFILE=1 — 메커니즘 측정 전용, TPS 판정
아님):

| 지표 (mixed step) | base (JIT-all) | subset | Δ |
|---|---|---|---|
| respond span 평균 | 5.43 ms (n=195) | **5.02 ms** (n=194) | **−0.41 ms** |
| respond span p95 | 5.70 | 5.09 | −0.61 |
| target spec_wait 평균 | 9.07 ms | 8.57 ms | −0.50 ms |

**정정**: subset은 JIT 구간에서 실제로 **~0.4ms/step 빠르다** (일관,
p95도 개선). 본 A/B에서 k2x2가 −2.0%로 나왔던 것은 JIT 메커니즘이
아니라 **토큰 샘플링 노이즈**였다고 보는 것이 옳다 (이 프로파일 쌍의
TPS도 sub가 +1.3%로 반대 방향; 앞서 추정한 "bucket/eager 발사 비용"
가설은 span 데이터로 반증). 다만 0.4ms는 step(~160ms)의 0.25%라
**게이트 기본 OFF 결론은 불변** — 이득이 실재하지만 승격 기준 미달.

## 부록 — B=16의 miss 통계 (질문 답)

프로파일 status 집계 (out=128, 두 arm 합산 ~486 step):

| step 유형 | 비율 | spec_wait |
|---|---|---|
| mixed (일부 miss → JIT 실행) | **88.5~91.4%** | 8.6~9.1 ms |
| all-hit (JIT 생략) | 8.2~10.7% | 2.3~4.3 ms |
| all-miss | 0.4~0.8% | 37~78 ms |

즉 **B=16에서 step의 ~90%가 JIT을 실행**한다 (B=1 시절 miss율
15~18% → B=32에선 97.9%). miss 개수는 per-step 로그가 없어 hit율
0.87에서 이항 근사: **평균 ≈ 2.1개/step**, P(0)≈11% (실측 all-hit
10.7%와 일치 — 독립 근사 유효), P(1)≈25%, P(2)≈28%, P(3)≈20%,
P(≥4)≈16%.
