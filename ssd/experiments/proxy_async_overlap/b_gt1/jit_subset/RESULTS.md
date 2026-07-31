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
