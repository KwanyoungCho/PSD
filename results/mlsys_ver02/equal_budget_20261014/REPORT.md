# 동일 draft 입력 예산의 tree AL 비교

최종 확인: 2026-10-14T22:38:21+09:00

**상태: 14/14 실행·검증·최종 분석 완료. [최종 분석 보고서](FINAL_REPORT.md)와 [그림 PDF](figs/ALL_FIGURES.pdf) 참조.**

- 모델: LayerSkip-Llama2-7B + AMD-Llama-135m, full BF16.
- B=1, T=0.7, raw Spec-Bench 480문항/560턴, 입력 절단 없음, 출력 상한1024, 문맥 상한2048.
- MT-Bench 두 턴은 기존 논문 실행처럼 독립 raw prompt. 문항별 턴 평균 후 문항 평균.
- K1=K2=8, P1/P2 root 각각8개, forward 폭8개. Phase당 물리 입력64행.
- Calibration 수락률 없이 root 점수 × draft 경로 확률. P1/P2 root 점수는 모든 비교군에서 동일 규칙.
- Chain 형태(C1), 마지막 깊이 경쟁(C3/C8), 전체 미확장 말단 경쟁(C3/C8); 검증상한32. C8은 상한64도 진단.
- 모든 비교군은 같은 unified tree 실행기를 실제 B1로 사용. C1은 chain 모양의 대조군이며 chain 전용 fast path 성능 비교가 아님.
- 로컬 seed16101, eslab18 seed26101. 각 서버/seed 안에서 모든 비교군을 비교. 서버와 seed 효과를 별도로 추정하는 설계는 아님.
- 주 비교 5개 + 검증상한 진단2개 × 2seed = 총14개 full 실행. B>1은 후속 검증.

아래 표는 최초 실패·외부 간섭 기록도 보존한 대기열 이력이다. 재실행 완료분으로 대체한 최종 유효 결과는 양 서버 각각7개다.
| 서버 | GPU | 실행 대기열 | 완료/계획 | 실행 기록 |
|---|---|---|---:|---|
| 로컬4090 | 4,5 | local_45_paper | 3/3 | 대기/완료 |
| 로컬4090 | 6,7 | local_67_paper | 3/4 | last_c8_m32: complete_external_excluded / PAPER_GROUP last_c8_m32 measured_0 544 |
| 로컬4090 | 4,5 | local_45_retry_external | 1/1 | 대기/완료 |
| eslab18 3090 | 2,3 | nova_23_paper | 2/3 | chain_m32: failed / PAPER_GROUP chain_m32 measured_0 544 |
| eslab18 3090 | 4,5 | nova_45_paper | 2/2 | 대기/완료 |
| eslab18 3090 | 6,7 | nova_67_paper | 1/2 | last_c8_m64: failed / PAPER_GROUP last_c8_m64 measured_0 528 |
| eslab18 3090 | 2,3 | nova_23_retry_disk | 1/1 | 대기/완료 |
| eslab18 3090 | 6,7 | nova_67_retry_disk | 1/1 | 대기/완료 |

## 검증 및 해석 기준

- 최종 비교는 560턴 UID 순서·입력token ID·출력상한·종료상태·동일 source hash를 검증한 full 결과만 포함.
- 외부 GPU 프로세스가 관측되거나 실행이 실패한 결과는 제외. 사용 중인 GPU의 외부 작업을 종료하지 않음.
- AL은 확정 root/target token 1개 포함. 상한 도달·잘린 step은 제외하며 graph 준비 step은 AL에는 포함(TPS에서는 제외).
- 전체/P1-hit/P2-hit/miss AL 및 분모, 6개 하위과제, 실제 hit tree 깊이·폭, 생성 대비 검증상한 손실을 함께 분석.
- 신뢰구간은 문항 단위 paired bootstrap. P1/P2 조건부 표본이 정책별로 달라짐을 명시. 같은 방문 문맥을 고정한 인과 실험은 아님.
- 생성 입력 행수가 같아도 샘플링 자식 수·호스트 작업·verification 노드 수·실측 latency까지 같다는 뜻은 아님.
- eslab18 최초 full 7개 중5개는 완료. Chain 및 last C8 M64는 저장공간 소진으로 각각545/529턴 raw만 남아 실패. 실패 원본을 보존하고 같은 runtime/seed로 전체560턴 재실행 완료. 잘린 campaign manifest도 원본을 남기고 실패 상태로 복구.
- 검증상한64 첫 smoke는 NCCL 메모리 할당 실패. 동일 runtime 재실행은 6턴/262 forests 모두 통과(phase당 active64행). 최초 실패 원인은 미확정이며 실패 raw를 보존.
- 기존 회귀283개, 입력 프로토콜22개, 입력검사4개 통과 기록은 각각 소스시점이 다름. 최종 압축기록 변경은 위 M64 smoke와 분석검사3개로 추가 검증.

## 완료된 비교 결과


### local — complete

| 방법 | 전체 AL | P1-hit AL | P2-hit AL | miss AL | hit(%) | AL 문항 수 |
|---|---:|---:|---:|---:|---:|---:|
| chain_m32 | 3.4899 | 4.0836 | 2.8779 | 2.5037 | 82.9894 | 480 |
| last_c3_m32 | 3.6510 | 4.4870 | 3.5131 | 2.5928 | 78.2091 | 480 |
| frontier_c3_m32 | 3.6761 | 4.5524 | 3.5863 | 2.5930 | 76.9026 | 480 |
| frontier_c8_m32 | 2.9396 | 3.3483 | 2.7563 | 2.6018 | 80.0509 | 480 |
| last_c8_m64 | 3.5868 | 4.5248 | 3.3729 | 2.6569 | 77.4526 | 480 |
| frontier_c8_m64 | 3.7101 | 4.6798 | 3.4496 | 2.6690 | 77.8103 | 480 |
| last_c8_m32 | 2.9008 | 3.3025 | 2.6986 | 2.5711 | 80.9299 | 480 |

방법별로 방문한 문맥과 hit 조건부 모집단이 다르다. 통계·구조·한계 해석은 FINAL_REPORT.md 참조.

### remote — complete

| 방법 | 전체 AL | P1-hit AL | P2-hit AL | miss AL | hit(%) | AL 문항 수 |
|---|---:|---:|---:|---:|---:|---:|
| frontier_c3_m32 | 3.7089 | 4.5034 | 3.5852 | 2.6400 | 77.9377 | 480 |
| last_c3_m32 | 3.6292 | 4.4972 | 3.5708 | 2.6176 | 76.6984 | 480 |
| frontier_c8_m32 | 2.9099 | 3.3424 | 2.7308 | 2.5958 | 79.8491 | 480 |
| last_c8_m32 | 2.9074 | 3.3271 | 2.7383 | 2.6308 | 80.6400 | 480 |
| frontier_c8_m64 | 3.6785 | 4.6421 | 3.4311 | 2.7119 | 77.1199 | 480 |
| chain_m32 | 3.3839 | 3.9826 | 2.9204 | 2.4833 | 80.7618 | 480 |
| last_c8_m64 | 3.6326 | 4.6469 | 3.3807 | 2.5992 | 76.2150 | 480 |

방법별로 방문한 문맥과 hit 조건부 모집단이 다르다. 통계·구조·한계 해석은 FINAL_REPORT.md 참조.

## 파일 및 재현

- `PLAN.md`, `PAPER_PROTOCOL.json`, `SPEC_BENCH_FULL.json`: 설계·논문 입력조건 근거.
- `SOURCE_SHA256.json`, `MODEL_SHA256.json`: 양 서버에 같은 소스/모델을 복제했는지 대조.
- `deployment_plan.py`, `local_*_paper.json`, `nova_*_paper.json`: GPU쌍별 순차실행 계획.
- `status.json`: 자동 갱신 진행상태. `RESULTS.json`: 완료 full arm의 질문별/phase별/구조별 집계 및 CI.
- `observe.py`: 진행 확인, 완료된 원격 raw의 로컬 복제, 집계·본 문서 갱신. Runtime 수정이나 외부 프로세스 종료 없음.
- 로컬 raw: `/data/chokwans99/equal-tree-paper-full-20261014`. 원격 raw: `/home/chokwans99/equal_budget_results`.
- Runtime/bench는 실행 중 고정. 이번 실험 변경은 아직 로컬 연구 브랜치에 commit/push하지 않았음. 원격 c91a1f4는 복제 소스의 snapshot commit.
