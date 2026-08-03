# 20 — T1 구현 계획·진행 (P2-tree rollout, draft 측)

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
