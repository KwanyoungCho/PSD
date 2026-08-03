# 20 — P2-tree 구현 로드맵·진행 (T1 → 최종 산출물까지)

**전권 위임 (사용자 2026-08-03)**: 끊김 없이 최종 구현까지 연속 진행.
모든 이슈는 본 문서(+17/19번)에 기록. 완료 후: timeline 기반 파라미터
sweep → AR/async-SD(C)/기존 DUET 대비 종합 지표 (cache hit, L_p1/L_p2,
TPS 등) → **정확한 timeline 그래프** (정식 도구
`bench/plot_duet_aligned_timeline.py`, 수 step 구간) → 논문/발표용
"기법 동작 원리 + 결과" 절 정리.

## 전체 로드맵

| 국면 | 내용 | 산출물 |
|---|---|---|
| T1 (진행 중) | draft rollout: 노브/예산/샘플러/루프/캐시 | 아래 단계표 |
| T2 | 트리 응답: wire 스키마(max-padded+헤더) + 캐시 뷰 + outcome(node id) + score 관통 | wire 왕복 테스트 |
| T3 | target: attention mode 계약 + rank별 wrapper + tree-verify CG(bucket) + 잔차 사다리 보행 + TP4 commit-ack + glue 트리화 + P1 fork 일반화 + tree Policy B | **작은 vocab 전수 분포-일치(하드 게이트)** + 체인 퇴화 bit-identical |
| T4 | E2E 통합 + **timeline 기반 sweep** (R/N_v/C_tensor/β/정책 × 균형 2조건 확인) | sweep 결과표 (CPU 한가 시간대 실측) |
| T5 | 비교분석: AR / async-SD(C-opt) / DUET-체인(champion) / DUET-tree — 5-rep 인터리브, 종합 지표 + timeline 그래프 (몇 step 정밀) | 비교표 + PNG + 논문용 절 |
| 마감 | NCCL A/B (연기분), 논문/발표용 기법 설명 + 결과 정리 | 21번(가칭) 최종 리포트 |

# T1 상세 (draft 측)

**전제**: 설계 v6 (15번) 확정 결정 ①~⑤ 준수. T1 범위 = draft 측
rollout(정책 스위치) + 비복원 샘플(D8/D11) + 사전 예산(⑤v2) + 형제
순서 기록 + 캐시 구조. **응답 wire(T2)·tree verify(T3)는 범위 밖** —
따라서 T1 동안 tree ON은 E2E로 돌지 않으며, 검증은 unit-수준(CPU 참조
동일성)과 tree OFF 회귀(bit-identical)로 한다.

**게이트 설계 (안전 원칙)**: `--duet_tree_policy {off, level, frontier}`
— 기본 **off** = 기존 경로 코드 그대로 (분기 1회). off에서 기존과
bit-identical이 T1의 hard 게이트.

## 단계 분해 (단계별 커밋 + 테스트)

| 단계 | 내용 | 검증 |
|---|---|---|
| T1.1 | config/CLI 노브 (`duet_tree_policy`, `duet_tree_c_tensor`(=C_tensor, 기본 3), `duet_tree_nv`(=N_v, 기본 8)) + OFF 게이트 배선 | config 동등성 (off ≡ 현행 필드), 기존 44개 회귀 |
| T1.2 | 사전 예산 b_x: root별 P_iv-비례 (β 지수 노브, 기본 0.5 — E1 근거) + 같은-forward 부모 priority-정렬 prefix 배분 + root 예산 소진 관리 — **pure 함수** | CPU 유닛: 합=예산, D10 준수(정체 무관), 몰빵/균등 경계 사례 |
| T1.3 | 비복원 샘플러: C_tensor 일괄 exponential-race + 앞 b_x만 valid + **뽑힌 순서 기록**; temp==0 게이트(체인 폴백); q_eff는 verify.py 분포 빌더 공유(`build_sampling_probs` 추출) | CPU 유닛: 분포 정합(카이제곱), 순서 재현성, fanout=1이 기존 단일 샘플과 **RNG 소비까지 동일** |
| T1.4 | rollout 루프: pool 텐서(고정 용량) + 정책 스위치 선택(level=최신 depth 마스크 / frontier=전체 topk) + 동적 조상 mask 주입(기존 draft packed-bitmask 기계에 내용만) + 셀 주소 규칙(§6 v6) | CPU 참조 구현(파이썬 트리) 대비 topology 동일성; mask 정합(조상 집합) |
| T1.5 | 캐시 구조: root별 서브트리 뷰 (tok/parent/순서/logits, U_max=N_v pad) 저장 — 응답은 아직 체인(기존) | 뷰 스키마 유닛; tree OFF 회귀 44개 + champion 스모크(OFF, 정확성만) |

## 진행 로그

**T1.1 완료**: config 필드 4종 (`duet_tree_policy` off|level|frontier
기본 off / `duet_tree_c_tensor` 3 / `duet_tree_nv` 8 /
`duet_tree_beta` 0.5) + -O 생존 검증 + CLI 4종. off 기본이라 기존
경로 무변경 — 동등성 6/6 + 회귀 44/44 green.

**T1.2 완료**: `ssd/engine/helpers/p2_tree.py` — `alloc_root_budgets`
(P_iv^β largest-remainder, cap=N_v, 결정론) + `alloc_fanouts` (같은
forward 부모들 priority-정렬 prefix 배분 — 리뷰 4차 초과 반례 방지).
**D10을 시그니처로 보장** (자식 정체 입력 없음 — 테스트로 고정).
유닛 11/11 (합 보존/cap/균등·비례 경계/결정론/무변이).
