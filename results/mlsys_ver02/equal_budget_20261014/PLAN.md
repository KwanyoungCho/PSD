# 동일 생성 입력 예산의 tree 비교 — 2026-10-14

상태: 14/14 full 실행 및 최종 분석 완료. 결과·한계·후속 권고는 [FINAL_REPORT.md](FINAL_REPORT.md) 참조. 아래는 실험 당시 합의한 계획이다. 기준 `MLSys-ver0.2@76c98f9`.

## 사용자와 합의한 조건

- Calibration alpha/gain을 사용하지 않는다. 기본 tree 점수를 q_path로 되돌린다.
- P1/P2 각각 요청당 root R=8, 매 round 입력 W=8, K1=K2=8.
- P1은 context의 draft 경로 확률 × root q, P2는 현재 위치·토큰 결합 점수.
- 같은 phase의 정책 비교에서는 root 생성 규칙과 예산이 같다. 온라인 실행의 방문 문맥까지 같다는 뜻은 아니다.
- Chain 형태(C=1, root별 한 경로), 마지막 깊이 경쟁(C=3,8), 전체 미확장 경쟁(C=3,8).
- `q_global`은 마지막 깊이의 root 간 경쟁을 뜻한다. 전체 깊이 경쟁과 혼동하지 않는다.
- 생성 후 점수 rerank는 이번 주 비교에 넣지 않는다. Native ordered-WOR proposal/verification을 유지한다.
- Root별 생성 상한 R*K*C는 한 root가 가질 수 있는 전체 생성 수보다 작지 않으므로 탐색을 막지 않는다.
- 검증은 생성 순서 prefix M=32를 우선 비교한다. C8에서 M=64 영향은 별도 진단한다. 모든 방법에 같은 검증 상한을 주며 점수 rerank는 하지 않는다.
- B=1(사용자 확정; B>1은 후속 검증), T=.7, exit21, full BF16 LayerSkip-Llama2-7B + AMD-Llama-135m 단일 조합. 사용자 최신 지시에 따라 Llama3 시도는 중단하고 결과에서 제외한다. 로컬 GPU4–7 및 eslab18 GPU2–7을 이용하며, 각 서버 안에 모든 비교군을 배치하고 GPU pair별로 나누어 실행한다. 로컬 seed16101, 원격 seed26101이며 서버 간 절대 latency는 비교하지 않는다.
- Spec-Bench 전체 480문항/560턴, 입력 절단 없음, 출력 상한 1024, 문맥 상한 2048. PDF §5.1과 논문 브랜치 run_full_arm.sh를 대조했다. PAPER_PROTOCOL.json에 근거를 보존했다. MT 턴은 기존 논문 입력처럼 독립 raw prompt이며 앞선 생성 응답을 붙이지 않는다.
- Miss는 기존 JIT chain K2=8. Mixed-miss AR, proxy replica/side stream, target packed 변경 없음.

## 동일하다고 말할 수 있는 것

P1/P2 생성 본문의 물리 입력 행은 각각 요청당 R*K=64, 합계 128행이다. B=1에서는 이것이 step당 입력 수이다.
자식 샘플 수, 선택 비용, glue 입력, 실제 활성 슬롯과 target 검증 입력은 별도다.
`같은 생성 입력 예산`을 `동일한 실측 latency`로 표현하지 않는다.
Chain 대조는 같은 tree 실행기에 C=1을 넣은 chain 형태이다. 기존 chain fast 실행기 성능 비교가 아니다.

## 실험 adapter

`ssd/bench/mlsys_equal_budget_adapter.py`에서만 forest 구성과 root 선택을 바꾼다.
P1 각 context에서 top-R 후보를 구한 후 전체 top-R을 선택한다. P2는 선택된 P1 root와 중복을 제거하고 top-R을 선택한다.
P1 후보 수가 contexts*R로 증가하지 않게 실제 forest root/round 폭을 R로 고정한다.
KV 예약은 glue+R*K의 실제 고정 입력 배치에 맞춘다. 모델 context 상한과 scheduler 보호는 유지한다.
Proxy wire는 최대 R개의 P1 중복을 제거할 수 있게 2R+2 후보를 보낸다.
호스트 export는 전체 큰 생성 arena 대신 native와 동일한 생성 순서 prefix M만 전송한다. 점수로 pruning하지 않는다.
실험 코드 SHA256, runtime SHA256, 실제 forest 폭 로그, GPU telemetry와 raw JSON을 보존한다.

## 결과와 진단

- 전체 AL, P1-hit AL, P2-hit AL, miss AL 및 각 표본 수·비중·신뢰구간.
- AL은 native accepted_len(추가 target token 포함), accepted draft 수는 AL-1로 따로 표기.
- 실제 hit tree 노드 수·최대 깊이·depth ceiling 도달 비율.
- Root별 생성 노드·최대 깊이·forward 처리 노드·검증 prefix 크기.
- 메인 full은 AL 비교이며 상세 CUDA event profiler를 껐다. Step/verification wall time은 보조 지표로 남긴다. P1/P2/glue의 세부 latency는 사전 실행 또는 별도 후속 프로파일에서만 해석한다.
- 검증 cap/output cap/부분 batch/문맥 보호 종료의 분모를 공개한다.
- 같은 후보 수를 유지해도 P1과 P2 hit 표본은 다른 조건부 모집단이다. 조건부 AL만으로 전체 우위를 선언하지 않는다.
- Forward/target 실제 입출력과 proposal q, eager/graph parity를 C=1/3/8, K=8에서 검사한다.

## 선행연구의 위치

- EAGLE-2/OPT-Tree: 마지막 깊이 확장과 최종 검증 예산의 분리.
- SpecExec/SwiftSpec: 서로 다른 깊이의 미확장 후보를 확률로 선택하는 근거.
- DySpec: 깊이/형제 추가의 비교는 고정 C 실험 뒤 검토한다.
- DDTree/Tsubame: diffusion의 분포 재사용 비용은 autoregressive draft에 그대로 적용할 수 없다. 사후 선택과 stochastic 검증의 법칙을 분리해야 한다.

이번 주장은 일반적인 frontier 탐색의 발명이 아니라, 아직 확정되지 않은 여러 verification outcome의 후속 계산을 early-exit 후보 점수로 배분하는 효과이다. 수락률 보정이나 q를 정확한 target 수락률로 해석하는 주장은 하지 않는다.

원문:
- https://arxiv.org/html/2406.16858v2
- https://aclanthology.org/2025.tacl-1.8.pdf
- https://papers.neurips.cc/paper_files/paper/2024/file/1d91d5689e251d27993a3c2182dddcf7-Paper-Conference.pdf
- https://arxiv.org/html/2506.11309v1
- https://www.wict.pku.edu.cn/docs/20250717175203347541.pdf
- https://arxiv.org/html/2604.12989v1
- https://arxiv.org/html/2609.33652v1

## 입력 조건 정정 기록

- 이전 16문항/32출력 실행은 기능 검사다. 480 첫 턴·512입력·128출력 full 계획은 폐기했고, 시작했던 nova M64 chain도 중단했다. 이 결과를 논문 조건의 AL 결론에 사용하지 않는다.
- 전체 560턴 중 141턴은 512토큰을 넘으므로 입력 절단이 특히 요약/RAG에 영향을 줄 수 있다.
- 논문 토크나이저와 7B 기본 토크나이저는 동일 vocab/ID 대응이지만 전처리가 다르다. 논문 토크나이저로 입력 ID를 고정했고 과거 560 raw의 입력 토큰 수가 모두 일치한다. 모델 가중치나 분포를 바꾸지 않는다.
- P1/P2/miss AL은 같은 정의(확정 root/target 토큰 1개 포함)를 쓴다. 문항별 턴 평균 후 480문항 평균을 headline으로 보고, 단계 가중 평균과 6개 subtask도 함께 남긴다.
- 로컬/원격 서버는 같은 checkpoint를 사용하며 모델 및 source SHA256을 대조한다. 원격 git commit은 snapshot commit이고 원본 브랜치 commit으로 표현하지 않는다.
