# 15 — P2 동적 트리 설계 (design-first; 구현 착수 전)

**작성일**: 2026-08-02. **상태: 설계 단계 — 구현 금지.** 사용자 지시:
"모든 건 완벽하게 설계가 이루어진 이후에 시작. 충분한 검증·설계·실험이
진행된 이후에 시작할 수 있도록" — 본 문서의 선행 실험(E0-E2)이 전부
green이고 설계가 승인되기 전에는 엔진 코드를 만지지 않는다.

## 1. 목표

P2 hit 시 accepted length가 낮다 (per-position 수락 0.5~0.6, L_p2 ≤
K2에서 조기 사망). seed 이후의 rollout이 무분기 체인이기 때문. **proxy
score × confidence 기반 동적 트리**로 2번째 이후 토큰을 헤징해
E[AL|P2 hit]을 올린다. **타겟 레짐 B=1** (B>1은 구현 호환만, 최적화는
추후 — B≥16은 대부분 miss라 async 구조 이점이 소멸된 레짐).

## 2. 확정된 설계 결정 (사용자, 2026-08-02)

- **D1 — shape 완전 고정, 재배분 동적**: 매 forward 입력 행수는 CG
  재사용을 위해 상수. seed별 확장 수는 점수로 재배분 (seed 1이 2개,
  seed 2가 0개 식). → **가능함이 확인됨**: Phase-2 seed 레벨이 이미
  정확히 이 기계다 (fan_out 합=상수, fan_idx/mask는 값만 변동, 유일
  .tolist() 1회). 이 기계를 레벨마다 반복 적용하면 된다. EAGLE-2의
  레벨별 top-k 확장과 동형 (shape-static / content-dynamic).
- **D2 — score = proxy score × confidence**: 단, proxy score의 hit
  예측력(calibration)을 E0로 먼저 실증한다. "budget 때문에 들어간
  낮은 점수 seed가 실제로도 hit 확률이 낮은가"가 검증 질문.
- **D3 — log 공간**: EAGLE-2 원논문은 raw 곱 V_i = ∏c_j를 사용하며
  log는 등장하지 않음 (원문 확인). log-합은 top-k 선택에 **동치**
  (단조변환)이고 수치적으로 안전하며 proxy 점수와의 합성이 깔끔:
  `value(n) = log s_seed + Σ log c_j`. 우리 깊이(K2 ≤ 4)에선 fp32
  underflow 위험이 낮지만 비용이 0이므로 log 채택. (beam search
  관례와 동일.)
- **D4 — 동적 선택 오버헤드 최소**: 레벨별 선택은 GPU topk 한 번 —
  신규 GPU→CPU 동기화 0회 원칙 유지 (M1-M3 독트린).

## 3. 사실 확인 결과 (2026-08-02 코드/논문 검증)

- **F1 — "verify tree-attention 이미 구현"은 절반만 참**:
  `get_custom_mask`(트리 어텐션)는 **draft tree-decode 전용**이다
  (cudagraph_helpers의 draft 경로에서만 사용). target의
  `capture_duet_verify_cudagraph`/`capture_verify_cudagraph`는
  cu_seqlens 기반 **causal varlen 체인 전용** — custom mask 버퍼가
  capture에 없다. 트리 verify는 draft 쪽 packed-mask 기계를 verify
  CG에 이식하는 신규 공사다 (기계는 있으니 이식이지 발명은 아님).
- **F2 — 수락 알고리즘 신규 필요**: utils/verify.py는 체인 순차
  ratio test (위치당 후보 1개, 첫 기각에서 정지 + residual recovery).
  트리 무손실 수락은 **같은 위치의 sibling들에 대한 순차 기각 샘플링
  (기각될 때마다 q를 without-replacement 보정)** — SpecInfer/EAGLE
  계열 다항 검증을 새로 구현해야 한다. 이것이 무손실성의 핵심.
- **F3 — KV slot 경합**: 같은 depth의 sibling verify 행들이 같은
  position slot을 경합한다. 후보안: (a) scratch slot에서 트리 verify
  후 accepted path KV만 canonical로 복사, (b) `kvcache_block_size ≥
  2k+2` 여유로 sibling별 slot 예약. — 설계 결정 D5로 보류.
- **F4 — 응답/wire 확장**: 현행 체인 [B,K]tok + [B,K,V]logits →
  트리 [B,N_v]tok + [B,N_v]parent + [B,N_v,V]logits. B=1, N_v~10이면
  logits ~640KB/step (현행 K=4 기준 ~256KB의 2.5배). spec_wait 예산
  내인지 E2로 실측.
- **F5 — 캐시/응답 크기 균일화 필요**: seed별 서브트리 크기가 점수
  재배분으로 달라지므로, hit 응답의 트리 노드 수도 상수 N_v로
  절단/패딩해야 CG가 산다 (설계 결정 D6: N_v 값과 절단 규칙).
- **F6 — 기존 SSD_TRACE_SPLIT_K1K2는 shape만 로깅** — E0에는 신규
  덤프 게이트가 필요하다 (§6, 승인 대상).

## 4. 트리 구성 알고리즘 (제안)

- **seed 레벨** (현행 유지): Policy B가 budget B_s개의 (pos, tok)
  seed 선택. seed 점수 s_i = ĥ_i × P_iv (이미 계산됨).
- **rollout 레벨 l = 1..K2**: 레벨 폭 W_l 고정. 이전 레벨 생존
  노드의 자식 후보(draft logits top-c)에 대해
  `value = log s_seed + Σ_path log c_j`를 매기고 **레벨 전체에서
  top-W_l 선택** → seed 간 재배분이 자동으로 일어남 (높은 seed가
  여러 자식, 낮은 seed 0개 — D1). fan_out/fan_idx/mask 갱신은 기존
  Phase-2 기계 재사용.
- **형상 파라미터**: {B_s, W_1..W_K2}, 총 노드 N = B_s + ΣW_l 상수.
  B>1 호환: per-seq N 상수 = 기존 budget-합-상수 불변량의 일반화.
- **EAGLE-2와의 차이**: ① 루트가 단일이 아니라 proxy-선택 seed 숲,
  ② seed 사전점수(proxy)가 draft confidence보다 강한 신호(E0로
  검증), ③ 레벨 폭이 고정이므로 EAGLE-2의 rerank 단계 불필요.

## 5. verify 측 설계 (개요; F1-F3 해소 방안)

hit 시 응답 = 명중 seed의 서브트리(N_v 노드, 상수). target verify
rows = N_v+1, packed custom mask(F1 이식), 깊이별 sibling 순차 기각
샘플링(F2), accepted path만 KV 커밋(F3-D5). 비용 모델: B=1 verify
행당 한계비용 ~1.9ms (frontier finding) — 트리가 행당 기대 토큰으로
이를 이겨야 하며, 그 판정이 E1이다. 균형조건 ①②는 그대로 적용:
트리 rollout 증가는 proxy_wait 예산(조건① 위반분)의 소비처가 된다.

## 6. 선행 실험 (구현 전 관문 — 전부 green이어야 착수)

- **E0 — proxy score calibration** (승인 필요: 신규 덤프 게이트
  `SSD_DUET_SCORE_TRACE=1`, target 쪽 ~30줄): step마다 {chosen
  (pos,tok), score}와 다음 step의 실제 outcome(reject pos, rec tok)을
  JSONL로 덤프 → offline join으로 (a) score 분위별 hit율 곡선,
  (b) seed rank vs hit, (c) score와 hit 후 L_p2의 상관. **판정**:
  곡선이 단조면 D2 채택, 평평하면 confidence 단독으로 후퇴.
- **E1 — offline 트리 시뮬레이션** (엔진 무수정): E0 덤프 + HF
  draft/target 재생(top-level 분석 스크립트 전통)으로 "체인 대신
  트리였다면 E[AL|hit]이 얼마"를 형상 {B_s, W_l} 후보별로 계산.
  **판정**: ΔE[AL|hit] × P2 hit율 × 토큰 가치 > 추가 verify 행 ×
  1.9ms — 이 부등식이 성립하는 형상이 존재해야 착수.
- **E2 — 마이크로벤치**: 응답 payload 2.5×의 wire 시간, packed-mask
  verify capture 프로토타입의 replay 시간.

## 7. 구현 마일스톤 (설계 승인 + E0-E2 green 이후에만)

T0 덤프 게이트 → T1 draft 트리 rollout+캐시 → T2 응답/wire →
T3 verify mask+트리 수락 → T4 B=1 E2E + A/B → T5 B>1 호환 게이트.
각 단계 유닛테스트 + B=1 회귀 스모크 + 커밋 (M1-M6 방식). 미결
설계 결정: D5 (KV slot 방식), D6 (N_v 값/절단 규칙), 형상 초기값
(E1이 결정).
